use crate::{
    compiler::{Instruction, Program},
    tensor::{self, Op, Tensor},
};
use std::collections::{HashMap, HashSet};
#[derive(Clone, Debug)]
struct Node {
    value: Tensor,
    gradient: Option<Vec<f64>>,
    edge: Option<(Op, usize, Option<usize>)>,
    tracked: bool,
}
#[derive(Debug)]
pub struct Execution {
    pub output: Vec<Tensor>,
    pub variables: HashMap<String, Tensor>,
    pub peak_tensor_bytes: usize,
    pub checkpoint: crate::checkpoint::Checkpoint,
}
struct Vm<'a> {
    backend: &'a dyn crate::backend::Backend,
    nodes: Vec<Node>,
    slots: Vec<Option<usize>>,
    stack: Vec<usize>,
    parameters: HashSet<usize>,
    output: Vec<Tensor>,
    peak: usize,
    backward_ready: bool,
    adam: crate::optimizer::Adam,
    restore: HashMap<usize, Tensor>,
    inference: bool,
}
impl Vm<'_> {
    fn measure(&mut self) {
        let bytes = self
            .nodes
            .iter()
            .map(|n| 8 * (n.value.data.len() + n.gradient.as_ref().map_or(0, Vec::len)))
            .sum::<usize>()
            + self.output.iter().map(|t| 8 * t.data.len()).sum::<usize>();
        let moments = self
            .adam
            .states
            .values()
            .map(|s| 8 * (s.first.len() + s.second.len()))
            .sum::<usize>();
        self.peak = self.peak.max(bytes + moments);
    }
    fn push(
        &mut self,
        value: Tensor,
        tracked: bool,
        edge: Option<(Op, usize, Option<usize>)>,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            value,
            gradient: None,
            edge,
            tracked,
        });
        self.measure();
        id
    }
    fn pop(&mut self) -> Result<usize, String> {
        self.stack
            .pop()
            .ok_or_else(|| "invalid bytecode: stack underflow".into())
    }
    fn backward(&mut self, id: usize) -> Result<(), String> {
        if !self.nodes[id].value.shape.is_empty() {
            return Err("backward requires a scalar".into());
        }
        if !self.nodes[id].tracked {
            return Err("loss is not connected to any parameter".into());
        }
        for n in &mut self.nodes {
            n.gradient = None;
        }
        self.nodes[id].gradient = Some(vec![1.0]);
        for i in (0..=id).rev() {
            let Some((op, a, b)) = self.nodes[i].edge else {
                continue;
            };
            let Some(g) = self.nodes[i].gradient.as_ref() else {
                continue;
            };
            let (ga, gb) = self.backend.vjp(
                op,
                &self.nodes[a].value,
                b.map(|j| &self.nodes[j].value),
                &self.nodes[i].value,
                g,
            );
            for (j, delta) in std::iter::once((a, ga)).chain(b.zip(gb)) {
                if self.nodes[j].tracked {
                    let dst = self.nodes[j]
                        .gradient
                        .get_or_insert_with(|| vec![0.0; delta.len()]);
                    for (x, d) in dst.iter_mut().zip(delta) {
                        *x += d;
                    }
                }
            }
        }
        if self
            .nodes
            .iter()
            .filter_map(|n| n.gradient.as_ref())
            .flatten()
            .any(|v| !v.is_finite())
        {
            return Err("non-finite gradient".into());
        }
        self.backward_ready = true;
        self.measure();
        Ok(())
    }
    // Loop boundaries and optimizer steps explicitly truncate the autodiff graph.
    // Preserve aliases while dropping dead intermediates and previous gradients.
    fn compact(&mut self) {
        let parameter_nodes: HashSet<usize> = self
            .parameters
            .iter()
            .filter_map(|slot| self.slots[*slot])
            .collect();
        let mut remap = HashMap::new();
        let mut nodes = vec![];
        for slot in &mut self.slots {
            if let Some(old) = *slot {
                let next = *remap.entry(old).or_insert_with(|| {
                    let id = nodes.len();
                    let n = &self.nodes[old];
                    nodes.push(Node {
                        value: n.value.clone(),
                        gradient: None,
                        edge: None,
                        tracked: !self.inference && parameter_nodes.contains(&old),
                    });
                    id
                });
                *slot = Some(next);
            }
        }
        self.nodes = nodes;
        self.backward_ready = false;
        self.measure();
    }
    fn code(&mut self, code: &[Instruction]) -> Result<(), String> {
        for ins in code {
            match ins {
                Instruction::Constant(t) => {
                    let id = self.push(t.clone(), false, None);
                    self.stack.push(id);
                }
                Instruction::Random(shape, seed) => {
                    let mut state = *seed;
                    let mut data = Vec::with_capacity(tensor::elements(shape)?);
                    for _ in 0..tensor::elements(shape)? {
                        state = state.wrapping_add(0x9e3779b97f4a7c15);
                        let mut z = state;
                        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                        z ^= z >> 31;
                        data.push(((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0);
                    }
                    let id = self.push(Tensor::new(shape.clone(), data)?, false, None);
                    self.stack.push(id);
                }
                Instruction::Load(slot) => self
                    .stack
                    .push(self.slots[*slot].ok_or("invalid bytecode: uninitialized slot")?),
                Instruction::Store(slot, param) => {
                    let mut id = self.pop()?;
                    if *param {
                        let value = if let Some(saved) = self.restore.remove(slot) {
                            if saved.shape != self.nodes[id].value.shape {
                                return Err(
                                    "checkpoint parameter shape does not match declaration".into(),
                                );
                            }
                            saved
                        } else {
                            self.nodes[id].value.clone()
                        };
                        id = self.push(value, !self.inference, None);
                        self.parameters.insert(*slot);
                    }
                    self.slots[*slot] = Some(id);
                }
                Instruction::Apply { op, shape } => {
                    let binary = matches!(
                        op,
                        Op::Add
                            | Op::Sub
                            | Op::Mul
                            | Op::Div
                            | Op::BatchMatmul
                            | Op::Matmul
                            | Op::BceLogits
                            | Op::CrossEntropy
                    );
                    let b = if binary { Some(self.pop()?) } else { None };
                    let a = self.pop()?;
                    let t = self.backend.forward(
                        *op,
                        &self.nodes[a].value,
                        b.map(|i| &self.nodes[i].value),
                        shape,
                    )?;
                    if &t.shape != shape {
                        return Err("invalid bytecode: operation result shape mismatch".into());
                    }
                    let tracked = self.nodes[a].tracked || b.is_some_and(|i| self.nodes[i].tracked);
                    let id = self.push(t, tracked, if tracked { Some((*op, a, b)) } else { None });
                    self.stack.push(id);
                }
                Instruction::Detach => {
                    let a = self.pop()?;
                    let id = self.push(self.nodes[a].value.clone(), false, None);
                    self.stack.push(id);
                }
                Instruction::ClipGradNorm => {
                    let limit = self.pop()?;
                    let limit = self.nodes[limit].value.data[0];
                    if !self.backward_ready {
                        return Err(
                            "clip_grad_norm requires backward since the last step or loop boundary"
                                .into(),
                        );
                    }
                    let mut slots: Vec<_> = self.parameters.iter().copied().collect();
                    slots.sort_unstable();
                    let gradients = slots.iter().filter_map(|slot| {
                        self.nodes[self.slots[*slot].unwrap()].gradient.as_deref()
                    });
                    let transform = crate::optimizer::clip_transform(gradients, limit)?;
                    if let Some((divisor, magnitude)) = transform {
                        for slot in slots {
                            if let Some(g) = &mut self.nodes[self.slots[slot].unwrap()].gradient {
                                for value in g {
                                    *value = (*value / divisor) * magnitude;
                                }
                            }
                        }
                    }
                }
                Instruction::Gradient => {
                    let a = self.pop()?;
                    let n = &self.nodes[a];
                    let g = n
                        .gradient
                        .clone()
                        .ok_or("grad requires backward on a connected value first")?;
                    let t = Tensor::new(n.value.shape.clone(), g)?;
                    let id = self.push(t, false, None);
                    self.stack.push(id);
                }
                Instruction::Backward => {
                    let id = self.pop()?;
                    self.backward(id)?;
                }
                Instruction::Step => {
                    let id = self.pop()?;
                    let rate = self.nodes[id].value.data[0];
                    if rate <= 0.0 || !rate.is_finite() {
                        return Err("learning rate must be positive and finite".into());
                    }
                    if !self.backward_ready {
                        return Err(
                            "step requires backward since the last step or loop boundary".into(),
                        );
                    }
                    // Validate all updates before mutating parameters.
                    let mut updates = vec![];
                    for slot in &self.parameters {
                        let id = self.slots[*slot].unwrap();
                        let n = &self.nodes[id];
                        if let Some(g) = &n.gradient {
                            let data: Vec<_> = n
                                .value
                                .data
                                .iter()
                                .zip(g)
                                .map(|(v, g)| v - rate * g)
                                .collect();
                            updates.push((id, Tensor::new(n.value.shape.clone(), data)?));
                        }
                    }
                    for (id, t) in updates {
                        self.nodes[id].value = t;
                    }
                    self.compact();
                }
                Instruction::AdamStep { custom } => {
                    let config = if *custom {
                        let e = self.pop()?;
                        let b2 = self.pop()?;
                        let b1 = self.pop()?;
                        crate::optimizer::AdamConfig {
                            beta1: self.nodes[b1].value.data[0],
                            beta2: self.nodes[b2].value.data[0],
                            epsilon: self.nodes[e].value.data[0],
                        }
                    } else {
                        crate::optimizer::AdamConfig::default()
                    };
                    let rate = self.pop()?;
                    let rate = self.nodes[rate].value.data[0];
                    if !self.backward_ready {
                        return Err("adam_step requires backward since the last optimizer step or loop boundary".into());
                    }
                    let inputs: Vec<_> = self
                        .parameters
                        .iter()
                        .filter_map(|slot| {
                            let n = &self.nodes[self.slots[*slot].unwrap()];
                            n.gradient
                                .as_ref()
                                .map(|g| (*slot, n.value.clone(), g.clone()))
                        })
                        .collect();
                    let updates = self.adam.update(&inputs, rate, config)?;
                    for (slot, t) in updates {
                        let id = self.slots[slot].unwrap();
                        self.nodes[id].value = t;
                    }
                    self.compact();
                }
                Instruction::Print => {
                    let id = self.pop()?;
                    self.output.push(self.nodes[id].value.clone());
                    self.measure();
                }
                Instruction::AssertClose => {
                    let tolerance = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let eps = self.nodes[tolerance].value.data[0];
                    if eps < 0.0 || !eps.is_finite() {
                        return Err("assert_close tolerance must be nonnegative and finite".into());
                    }
                    if self.nodes[a]
                        .value
                        .data
                        .iter()
                        .zip(&self.nodes[b].value.data)
                        .any(|(x, y)| (x - y).abs() > eps)
                    {
                        return Err(format!(
                            "assert_close failed (absolute tolerance {eps}): {:?} versus {:?}",
                            self.nodes[a].value.data, self.nodes[b].value.data
                        ));
                    }
                }
                Instruction::Repeat {
                    count,
                    body,
                    locals,
                } => {
                    for _ in 0..*count {
                        self.code(body)?;
                        if !self.stack.is_empty() {
                            return Err("invalid bytecode: nonempty stack at loop boundary".into());
                        }
                        for slot in locals {
                            self.slots[*slot] = None;
                        }
                        self.compact();
                    }
                }
            }
        }
        Ok(())
    }
}
pub fn execute(program: &Program) -> Result<Execution, String> {
    execute_with_backend(program, &crate::backend::CpuBackend)
}
pub fn execute_with_backend(
    program: &Program,
    backend: &dyn crate::backend::Backend,
) -> Result<Execution, String> {
    execute_with_options(program, backend, None, false)
}
pub fn execute_with_checkpoint(
    program: &Program,
    checkpoint: &crate::checkpoint::Checkpoint,
) -> Result<Execution, String> {
    execute_with_options(
        program,
        &crate::backend::CpuBackend,
        Some(checkpoint),
        false,
    )
}
/// Execute with an explicit backend, optional model state and inference effects.
pub fn execute_with_options(
    program: &Program,
    backend: &dyn crate::backend::Backend,
    checkpoint: Option<&crate::checkpoint::Checkpoint>,
    inference: bool,
) -> Result<Execution, String> {
    if inference {
        check_inference(&program.code)?;
    }
    let mut restore = HashMap::new();
    let mut adam = crate::optimizer::Adam::default();
    if let Some(saved) = checkpoint {
        saved.validate()?;
        if saved.parameters.len() != program.parameters.len()
            || saved
                .parameters
                .keys()
                .any(|name| !program.parameters.contains_key(name))
        {
            return Err("checkpoint parameter names must exactly match the program".into());
        }
        for (name, t) in &saved.parameters {
            restore.insert(program.parameters[name], t.clone());
        }
        if !inference {
            adam.config = saved.adam_config;
            for (name, state) in &saved.adam {
                adam.states.insert(program.parameters[name], state.clone());
            }
        }
    }
    let mut vm = Vm {
        backend,
        nodes: vec![],
        slots: vec![None; program.slots],
        stack: vec![],
        parameters: HashSet::new(),
        output: vec![],
        peak: 0,
        backward_ready: false,
        adam,
        restore,
        inference,
    };
    vm.code(&program.code)?;
    let variables = program
        .bindings
        .iter()
        .filter_map(|(name, slot)| {
            vm.slots[*slot].map(|id| (name.clone(), vm.nodes[id].value.clone()))
        })
        .collect();
    let mut saved = crate::checkpoint::Checkpoint {
        adam_config: vm.adam.config,
        ..Default::default()
    };
    for (name, slot) in &program.parameters {
        if let Some(id) = vm.slots[*slot] {
            saved
                .parameters
                .insert(name.clone(), vm.nodes[id].value.clone());
            if let Some(state) = vm.adam.states.get(slot) {
                saved.adam.insert(name.clone(), state.clone());
            }
        }
    }
    Ok(Execution {
        checkpoint: saved,
        output: vm.output,
        variables,
        peak_tensor_bytes: vm.peak,
    })
}

pub fn execute_inference(
    program: &Program,
    checkpoint: Option<&crate::checkpoint::Checkpoint>,
) -> Result<Execution, String> {
    execute_with_options(program, &crate::backend::CpuBackend, checkpoint, true)
}
fn check_inference(code: &[Instruction]) -> Result<(), String> {
    for instruction in code {
        match instruction {
            Instruction::Gradient
            | Instruction::Backward
            | Instruction::ClipGradNorm
            | Instruction::Step
            | Instruction::AdamStep { .. } => {
                return Err("inference mode rejects gradient and optimizer operations".into())
            }
            Instruction::Repeat { body, .. } => check_inference(body)?,
            _ => {}
        }
    }
    Ok(())
}
