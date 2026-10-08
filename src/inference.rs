//! Forward-only SSA interpreter with last-use reclamation; no buffer reuse or fusion.
use crate::{
    backend::{Backend, CpuBackend},
    checkpoint::Checkpoint,
    ir::{Action, InferenceProgram, Region, ValueId},
    tensor::Tensor,
    vm::Execution,
};
use std::collections::HashMap;
struct Runner<'a> {
    program: &'a InferenceProgram,
    backend: &'a dyn Backend,
    checkpoint: Option<&'a Checkpoint>,
    values: Vec<Option<Tensor>>,
    output: Vec<Tensor>,
    active_bytes: usize,
    peak_bytes: usize,
}
impl Runner<'_> {
    fn get(&self, id: ValueId) -> Result<&Tensor, String> {
        self.values[id]
            .as_ref()
            .ok_or("inference plan referenced a released value".into())
    }
    fn insert(&mut self, id: ValueId, t: Tensor) -> Result<(), String> {
        if t.shape != self.program.shapes[id] {
            return Err("inference operation result shape mismatch".into());
        }
        if self.values[id].is_some() {
            return Err("inference region overwrote a live SSA value".into());
        }
        self.active_bytes += 8 * t.data.len();
        self.peak_bytes = self.peak_bytes.max(self.active_bytes);
        self.values[id] = Some(t);
        Ok(())
    }
    fn region(&mut self, r: &Region) -> Result<(), String> {
        for node in &r.nodes {
            let result = match &node.action {
                Action::Constant(t) => Some(t.clone()),
                Action::Random(seed) => {
                    let id = node.result.unwrap();
                    let n = crate::tensor::elements(&self.program.shapes[id])?;
                    let mut state = *seed;
                    let mut values = Vec::with_capacity(n);
                    for _ in 0..n {
                        state = state.wrapping_add(0x9e3779b97f4a7c15);
                        let mut z = state;
                        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                        z ^= z >> 31;
                        values.push(((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0);
                    }
                    Some(Tensor::new(self.program.shapes[id].clone(), values)?)
                }
                Action::Parameter { name, initializer } => {
                    let original = self.get(*initializer)?;
                    let value = if let Some(saved) = self.checkpoint {
                        let value = &saved.parameters[name];
                        if value.shape != original.shape {
                            return Err(
                                "checkpoint parameter shape does not match declaration".into()
                            );
                        }
                        value
                    } else {
                        original
                    };
                    Some(value.clone())
                }
                Action::Apply { op, a, b } => Some(self.backend.forward(
                    *op,
                    self.get(*a)?,
                    b.map(|id| self.get(id)).transpose()?,
                    &self.program.shapes[node.result.unwrap()],
                )?),
                Action::Print(id) => {
                    let t = self.get(*id)?.clone();
                    self.active_bytes += 8 * t.data.len();
                    self.peak_bytes = self.peak_bytes.max(self.active_bytes);
                    self.output.push(t);
                    None
                }
                Action::Assert { a, b, tolerance } => {
                    let a = self.get(*a)?;
                    let b = self.get(*b)?;
                    let eps = self.get(*tolerance)?.data[0];
                    if !eps.is_finite() || eps < 0.0 {
                        return Err("assert_close tolerance must be nonnegative and finite".into());
                    }
                    if a.data.iter().zip(&b.data).any(|(x, y)| (x - y).abs() > eps) {
                        return Err(format!(
                            "assert_close failed (absolute tolerance {eps}): {:?} versus {:?}",
                            a.data, b.data
                        ));
                    }
                    None
                }
                Action::Repeat { count, body, .. } => {
                    for _ in 0..*count {
                        self.region(body)?;
                    }
                    None
                }
            };
            if let Some(result) = result {
                self.insert(node.result.unwrap(), result)?;
            }
            for &id in &node.release {
                if let Some(value) = self.values[id].take() {
                    self.active_bytes -= 8 * value.data.len();
                }
            }
        }
        Ok(())
    }
}
pub fn execute(
    program: &InferenceProgram,
    checkpoint: Option<&Checkpoint>,
) -> Result<Execution, String> {
    execute_with_backend(program, checkpoint, &CpuBackend)
}
pub fn execute_with_backend(
    program: &InferenceProgram,
    checkpoint: Option<&Checkpoint>,
    backend: &dyn Backend,
) -> Result<Execution, String> {
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
    }
    let mut runner = Runner {
        program,
        backend,
        checkpoint,
        values: vec![None; program.shapes.len()],
        output: vec![],
        active_bytes: 0,
        peak_bytes: 0,
    };
    runner.region(&program.region)?;
    let mut variables = HashMap::new();
    let mut saved = Checkpoint::default();
    for (name, &id) in &program.bindings {
        variables.insert(name.clone(), runner.get(id)?.clone());
    }
    for (name, &id) in &program.parameters {
        saved
            .parameters
            .insert(name.clone(), runner.get(id)?.clone());
    }
    Ok(Execution {
        variables,
        checkpoint: saved,
        output: runner.output,
        peak_tensor_bytes: runner.peak_bytes,
    })
}
