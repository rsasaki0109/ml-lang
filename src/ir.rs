//! Typed region SSA for forward-only execution. Loops have immutable captures,
//! no carried values; each body definition is instantiated once per iteration.
use crate::{
    compiler::{Instruction, Program},
    tensor::{self, Op, Tensor},
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write;
pub(crate) type ValueId = usize;
#[derive(Clone, Debug)]
pub(crate) enum Action {
    Constant(Tensor),
    Random(u64),
    Parameter {
        name: String,
        initializer: ValueId,
    },
    Apply {
        op: Op,
        a: ValueId,
        b: Option<ValueId>,
    },
    Print(ValueId),
    Assert {
        a: ValueId,
        b: ValueId,
        tolerance: ValueId,
    },
    Repeat {
        count: usize,
        body: Region,
        captures: Vec<ValueId>,
    },
}
#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub result: Option<ValueId>,
    pub action: Action,
    pub release: Vec<ValueId>,
}
impl Node {
    fn operands(&self) -> Vec<ValueId> {
        match &self.action {
            Action::Constant(_) | Action::Random(_) => vec![],
            Action::Parameter { initializer, .. } => vec![*initializer],
            Action::Apply { a, b, .. } => std::iter::once(*a).chain(*b).collect(),
            Action::Print(a) => vec![*a],
            Action::Assert { a, b, tolerance } => vec![*a, *b, *tolerance],
            Action::Repeat { captures, .. } => captures.clone(),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub(crate) struct Region {
    pub nodes: Vec<Node>,
}
impl Region {
    fn defined(&self) -> BTreeSet<ValueId> {
        self.nodes.iter().filter_map(|n| n.result).collect()
    }
    fn captures(&self) -> Vec<ValueId> {
        let defined = self.defined();
        self.nodes
            .iter()
            .flat_map(Node::operands)
            .filter(|v| !defined.contains(v))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
    fn plan(&mut self, roots: &BTreeSet<ValueId>) {
        let defined = self.defined();
        let mut uses = HashMap::<ValueId, usize>::new();
        for value in roots {
            *uses.entry(*value).or_default() += 1;
        }
        for node in &self.nodes {
            for value in node.operands() {
                *uses.entry(value).or_default() += 1;
            }
        }
        for node in &mut self.nodes {
            for value in node.operands() {
                let remaining = uses.get_mut(&value).unwrap();
                *remaining -= 1;
                if *remaining == 0 && defined.contains(&value) {
                    node.release.push(value);
                }
            }
            if let Some(value) = node.result {
                if uses.get(&value).copied().unwrap_or(0) == 0 {
                    node.release.push(value);
                }
            }
            node.release.sort_unstable();
            node.release.dedup();
            if let Action::Repeat { body, .. } = &mut node.action {
                body.plan(&BTreeSet::new());
            }
        }
    }
}
/// Validated, CPU f64 inference IR. Construction is only through `lower`.
#[derive(Clone, Debug)]
pub struct InferenceProgram {
    pub(crate) region: Region,
    pub(crate) shapes: Vec<Vec<usize>>,
    pub(crate) bindings: BTreeMap<String, ValueId>,
    pub(crate) parameters: BTreeMap<String, ValueId>,
}
impl InferenceProgram {
    pub fn value_count(&self) -> usize {
        self.shapes.len()
    }
    /// Deterministic dump with explicit typed definitions, captures and last-use releases.
    pub fn display(&self) -> String {
        fn region(out: &mut String, r: &Region, shapes: &[Vec<usize>], depth: usize) {
            for node in &r.nodes {
                let indent = "  ".repeat(depth);
                write!(out, "{indent}").unwrap();
                if let Some(id) = node.result {
                    write!(out, "%{id}: Tensor<f64,cpu>{:?} = ", shapes[id]).unwrap();
                }
                match &node.action {
                    Action::Constant(t) => {
                        writeln!(out, "constant ({} elements)", t.data.len()).unwrap()
                    }
                    Action::Random(seed) => writeln!(out, "random seed={seed}").unwrap(),
                    Action::Parameter { name, initializer } => {
                        writeln!(out, "parameter {name} initializer=%{initializer}").unwrap()
                    }
                    Action::Apply { op, a, b } => writeln!(
                        out,
                        "{op:?} %{a}{}",
                        b.map_or(String::new(), |id| format!(", %{id}"))
                    )
                    .unwrap(),
                    Action::Print(a) => writeln!(out, "print %{a}").unwrap(),
                    Action::Assert { a, b, tolerance } => {
                        writeln!(out, "assert_close %{a}, %{b}, %{tolerance}").unwrap()
                    }
                    Action::Repeat {
                        count,
                        body,
                        captures,
                    } => {
                        writeln!(out, "repeat {count} captures={captures:?} {{").unwrap();
                        region(out, body, shapes, depth + 1);
                        writeln!(out, "{indent}}}").unwrap();
                    }
                }
                if !node.release.is_empty() {
                    writeln!(out, "{indent}release {:?}", node.release).unwrap();
                }
            }
        }
        let mut out = "inference SSA (CPU f64; last-use plan)\n".to_string();
        region(&mut out, &self.region, &self.shapes, 0);
        writeln!(
            out,
            "bindings {:?}\nparameters {:?}",
            self.bindings, self.parameters
        )
        .unwrap();
        out
    }
}
struct Builder {
    shapes: Vec<Vec<usize>>,
    slots: Vec<Option<ValueId>>,
    written: HashSet<usize>,
    stack: Vec<ValueId>,
    parameter_names: HashMap<usize, String>,
    parameters: BTreeMap<String, ValueId>,
}
fn binary(op: Op) -> bool {
    matches!(
        op,
        Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::BatchMatmul
            | Op::Matmul
            | Op::BceLogits
            | Op::CrossEntropy
    )
}
impl Builder {
    fn pop(&mut self) -> Result<ValueId, String> {
        self.stack
            .pop()
            .ok_or("invalid bytecode: stack underflow".into())
    }
    fn define(&mut self, r: &mut Region, shape: Vec<usize>, action: Action) -> ValueId {
        let id = self.shapes.len();
        self.shapes.push(shape);
        r.nodes.push(Node {
            result: Some(id),
            action,
            release: vec![],
        });
        id
    }
    fn effect(r: &mut Region, action: Action) {
        r.nodes.push(Node {
            result: None,
            action,
            release: vec![],
        });
    }
    fn region(&mut self, code: &[Instruction], in_loop: bool) -> Result<Region, String> {
        let mut r = Region::default();
        for instruction in code {
            match instruction {
                Instruction::Constant(t) => {
                    Tensor::new(t.shape.clone(), t.data.clone())?;
                    let id = self.define(&mut r, t.shape.clone(), Action::Constant(t.clone()));
                    self.stack.push(id);
                }
                Instruction::Random(shape, seed) => {
                    tensor::elements(shape)?;
                    let id = self.define(&mut r, shape.clone(), Action::Random(*seed));
                    self.stack.push(id);
                }
                Instruction::Load(slot) => {
                    let value = self
                        .slots
                        .get(*slot)
                        .copied()
                        .flatten()
                        .ok_or("invalid bytecode: uninitialized slot")?;
                    self.stack.push(value);
                }
                Instruction::Store(slot, parameter) => {
                    if *slot >= self.slots.len() {
                        return Err("invalid bytecode: slot out of bounds".into());
                    }
                    if !self.written.insert(*slot) {
                        return Err("inference SSA requires single-assignment slots".into());
                    }
                    let mut value = self.pop()?;
                    if *parameter {
                        if in_loop {
                            return Err(
                                "inference parameters cannot be declared inside loops".into()
                            );
                        }
                        let name = self
                            .parameter_names
                            .get(slot)
                            .cloned()
                            .ok_or("invalid bytecode: unnamed parameter")?;
                        if self.parameters.contains_key(&name) {
                            return Err("duplicate parameter definition".into());
                        }
                        value = self.define(
                            &mut r,
                            self.shapes[value].clone(),
                            Action::Parameter {
                                name: name.clone(),
                                initializer: value,
                            },
                        );
                        self.parameters.insert(name, value);
                    }
                    self.slots[*slot] = Some(value);
                }
                Instruction::Apply { op, shape } => {
                    let b = if binary(*op) { Some(self.pop()?) } else { None };
                    let a = self.pop()?;
                    tensor::elements(shape)?;
                    let actual = if *op == Op::Reshape {
                        if tensor::elements(&self.shapes[a])? != tensor::elements(shape)? {
                            return Err("invalid reshape size".into());
                        }
                        shape.clone()
                    } else {
                        tensor::output_shape(
                            *op,
                            &self.shapes[a],
                            b.map(|id| self.shapes[id].as_slice()),
                        )?
                    };
                    if &actual != shape {
                        return Err("invalid bytecode: result shape mismatch".into());
                    }
                    let id = self.define(&mut r, shape.clone(), Action::Apply { op: *op, a, b });
                    self.stack.push(id);
                }
                Instruction::Detach => {
                    // All values are immutable in this IR; an alias is already a snapshot.
                    let id = self.pop()?;
                    self.stack.push(id);
                }
                Instruction::Print => {
                    let id = self.pop()?;
                    Self::effect(&mut r, Action::Print(id));
                }
                Instruction::AssertClose => {
                    let tolerance = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    if self.shapes[a] != self.shapes[b] || !self.shapes[tolerance].is_empty() {
                        return Err("invalid assertion shapes".into());
                    }
                    Self::effect(&mut r, Action::Assert { a, b, tolerance });
                }
                Instruction::Repeat { count, body, .. } => {
                    if !self.stack.is_empty() {
                        return Err("invalid bytecode: stack at loop entry".into());
                    }
                    let outer = self.slots.clone();
                    let body = self.region(body, true)?;
                    self.slots = outer;
                    let captures = body.captures();
                    Self::effect(
                        &mut r,
                        Action::Repeat {
                            count: *count,
                            body,
                            captures,
                        },
                    );
                }
                _ => return Err("inference SSA rejects gradient and optimizer operations".into()),
            }
        }
        if !self.stack.is_empty() {
            return Err("invalid bytecode: nonempty region stack".into());
        }
        Ok(r)
    }
}
pub fn lower(program: &Program) -> Result<InferenceProgram, String> {
    let mut parameter_names = HashMap::new();
    for (name, &slot) in &program.parameters {
        if slot >= program.slots || parameter_names.insert(slot, name.clone()).is_some() {
            return Err("invalid parameter slots".into());
        }
    }
    let mut builder = Builder {
        shapes: vec![],
        slots: vec![None; program.slots],
        written: HashSet::new(),
        stack: vec![],
        parameter_names,
        parameters: BTreeMap::new(),
    };
    let mut region = builder.region(&program.code, false)?;
    if builder.parameters.len() != program.parameters.len() {
        return Err("missing parameter definitions".into());
    }
    let mut bindings = BTreeMap::new();
    for (name, &slot) in &program.bindings {
        bindings.insert(
            name.clone(),
            builder
                .slots
                .get(slot)
                .copied()
                .flatten()
                .ok_or("invalid final binding")?,
        );
    }
    let roots = bindings
        .values()
        .chain(builder.parameters.values())
        .copied()
        .collect();
    region.plan(&roots);
    Ok(InferenceProgram {
        region,
        shapes: builder.shapes,
        bindings,
        parameters: builder.parameters,
    })
}
