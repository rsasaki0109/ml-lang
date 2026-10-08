use crate::{
    syntax::{Ast, Dim, Expr, Function, Shape, Stmt},
    tensor::{self, Op, Tensor},
};
use std::collections::{HashMap, HashSet};
#[derive(Clone, Debug)]
pub enum Instruction {
    Constant(Tensor),
    Load(usize),
    Store(usize, bool),
    Apply {
        op: Op,
        shape: Vec<usize>,
    },
    Random(Vec<usize>, u64),
    Gradient,
    Detach,
    ClipGradNorm,
    Backward,
    Step,
    AdamStep {
        custom: bool,
    },
    Print,
    AssertClose,
    Repeat {
        count: usize,
        body: Vec<Instruction>,
        locals: Vec<usize>,
    },
}
#[derive(Debug)]
pub struct Program {
    pub code: Vec<Instruction>,
    pub slots: usize,
    pub bindings: HashMap<String, usize>,
    pub parameters: HashMap<String, usize>,
}
struct Compiler {
    vars: HashMap<String, (usize, Vec<usize>)>,
    functions: HashMap<String, Function>,
    next: usize,
    calls: Vec<String>,
    in_loop: bool,
    dimensions: HashMap<String, usize>,
    parameters: HashMap<String, usize>,
}
pub(crate) fn literal(e: &Expr) -> Result<Tensor, String> {
    match e {
        Expr::Number(x) => Ok(Tensor::scalar(*x)),
        Expr::Binary('*', a, b) if matches!(**a,Expr::Number(x) if x == -1.0) => {
            let mut t = literal(b)?;
            for x in &mut t.data {
                *x = -*x;
            }
            Ok(t)
        }
        Expr::Array(xs) => {
            if xs.is_empty() {
                return Err("empty tensor literals are unsupported".into());
            }
            let tensors: Vec<_> = xs.iter().map(literal).collect::<Result<_, _>>()?;
            let sub = &tensors[0].shape;
            if tensors.iter().any(|t| &t.shape != sub) {
                return Err("ragged tensor literal".into());
            }
            let mut shape = vec![xs.len()];
            shape.extend(sub);
            Tensor::new(shape, tensors.into_iter().flat_map(|t| t.data).collect())
        }
        _ => Err("tensor literals must contain numeric constants".into()),
    }
}
fn integer(e: &Expr) -> Result<usize, String> {
    if let Expr::Number(x) = e {
        if *x > 0.0 && x.fract() == 0.0 && *x <= 1_000_000.0 {
            return Ok(*x as usize);
        }
    }
    Err("expected positive constant integer".into())
}
pub(crate) fn row_start(e: &Expr) -> Result<usize, String> {
    if let Expr::Number(x) = e {
        if *x >= 0.0 && x.fract() == 0.0 && *x <= 1_000_000.0 {
            return Ok(*x as usize);
        }
    }
    Err("rows start must be a nonnegative constant integer".into())
}
pub(crate) fn axis_index(e: &Expr) -> Result<usize, String> {
    row_start(e).map_err(|_| "swap_axes requires nonnegative constant integer axes".into())
}
impl Compiler {
    fn resolve(&self, shape: &Shape) -> Result<Vec<usize>, String> {
        let concrete = shape
            .iter()
            .map(|d| match d {
                Dim::Fixed(n) => Ok(*n),
                Dim::Named(s) => self
                    .dimensions
                    .get(s)
                    .copied()
                    .ok_or_else(|| format!("unbound shape dimension '{s}'")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        tensor::elements(&concrete)?;
        Ok(concrete)
    }
    fn dimension(&self, e: &Expr) -> Result<usize, String> {
        if let Expr::Var(name) = e {
            self.dimensions
                .get(name)
                .copied()
                .ok_or_else(|| format!("unknown shape dimension '{name}'"))
        } else {
            integer(e)
        }
    }
    fn expr(&mut self, e: &Expr, code: &mut Vec<Instruction>) -> Result<Vec<usize>, String> {
        match e {
            Expr::Number(_) | Expr::Array(_) => {
                let t = literal(e)?;
                let s = t.shape.clone();
                code.push(Instruction::Constant(t));
                Ok(s)
            }
            Expr::Var(name) => {
                let (i, s) = self
                    .vars
                    .get(name)
                    .ok_or_else(|| format!("undefined variable '{name}'"))?;
                code.push(Instruction::Load(*i));
                Ok(s.clone())
            }
            Expr::Binary(c, a, b) => {
                let a = self.expr(a, code)?;
                let b = self.expr(b, code)?;
                let op = match c {
                    '+' => Op::Add,
                    '-' => Op::Sub,
                    '*' => Op::Mul,
                    '/' => Op::Div,
                    '@' => Op::Matmul,
                    _ => unreachable!(),
                };
                let s = tensor::output_shape(op, &a, Some(&b))?;
                code.push(Instruction::Apply {
                    op,
                    shape: s.clone(),
                });
                Ok(s)
            }
            Expr::Call(name, args) => {
                if let Some(f) = self.functions.get(name).cloned() {
                    if self.calls.contains(name) || self.calls.len() >= 32 {
                        return Err("recursive functions are unsupported".into());
                    }
                    if args.len() != f.args.len() {
                        return Err(format!("{name} expects {} arguments", f.args.len()));
                    }
                    let mut dimensions = HashMap::new();
                    let mut inputs = vec![];
                    for (e, (_, expected)) in args.iter().zip(&f.args) {
                        let actual = self.expr(e, code)?;
                        if actual.len() != expected.len() {
                            return Err(format!("{name}: rank mismatch"));
                        }
                        for (dim, &size) in expected.iter().zip(&actual) {
                            match dim {
                                Dim::Fixed(n) if *n != size => {
                                    return Err(format!(
                                        "{name}: expected dimension {n}, got {size}"
                                    ))
                                }
                                Dim::Named(symbol) => {
                                    if let Some(previous) = dimensions.insert(symbol.clone(), size)
                                    {
                                        if previous != size {
                                            return Err(format!("{name}: dimension {symbol} must be {previous}, got {size}"));
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        let slot = self.next;
                        self.next += 1;
                        code.push(Instruction::Store(slot, false));
                        inputs.push((slot, actual));
                    }
                    let saved = self.vars.clone();
                    self.vars.clear();
                    let saved_dimensions = std::mem::replace(&mut self.dimensions, dimensions);
                    for ((arg, _), (slot, shape)) in f.args.iter().zip(inputs) {
                        self.vars.insert(arg.clone(), (slot, shape));
                    }
                    self.calls.push(name.clone());
                    let expected = self.resolve(&f.result);
                    let result = self.function_body(&f, code);
                    self.calls.pop();
                    self.vars = saved;
                    self.dimensions = saved_dimensions;
                    let actual = result?;
                    let expected = expected?;
                    if actual != expected {
                        return Err(format!(
                            "{name}: return shape {actual:?} does not match {expected:?}"
                        ));
                    }
                    return Ok(actual);
                }
                match name.as_str() {
                    "zeros" | "ones" | "rand" => {
                        let (dims, seed) = if name == "rand" {
                            if args.len() < 2 {
                                return Err(
                                    "rand(dim..., seed) requires dimensions and a seed".into()
                                );
                            }
                            let last = integer(args.last().unwrap())? as u64;
                            (&args[..args.len() - 1], last)
                        } else {
                            (args.as_slice(), 0)
                        };
                        if dims.is_empty() {
                            return Err(format!("{name} requires dimensions"));
                        }
                        let shape = dims
                            .iter()
                            .map(|e| self.dimension(e))
                            .collect::<Result<Vec<_>, _>>()?;
                        let n = tensor::elements(&shape)?;
                        if name == "rand" {
                            code.push(Instruction::Random(shape.clone(), seed));
                        } else {
                            code.push(Instruction::Constant(Tensor::new(
                                shape.clone(),
                                vec![if name == "ones" { 1.0 } else { 0.0 }; n],
                            )?));
                        }
                        Ok(shape)
                    }
                    "gelu" | "tanh" | "sigmoid" | "softmax" | "causal_softmax" | "layer_norm"
                    | "transpose" | "transpose_last" | "relu" | "sum" | "mean" | "grad" => {
                        if args.len() != 1 {
                            return Err(format!("{name} expects one argument"));
                        }
                        let a = self.expr(&args[0], code)?;
                        if name == "grad" {
                            code.push(Instruction::Gradient);
                            return Ok(a);
                        }
                        let op = match name.as_str() {
                            "tanh" => Op::Tanh,
                            "gelu" => Op::Gelu,
                            "sigmoid" => Op::Sigmoid,
                            "softmax" => Op::Softmax,
                            "causal_softmax" => Op::CausalSoftmax,
                            "layer_norm" => Op::LayerNorm,
                            "transpose" => Op::Transpose,
                            "transpose_last" => Op::TransposeLast,
                            "relu" => Op::Relu,
                            "sum" => Op::Sum,
                            _ => Op::Mean,
                        };
                        let s = tensor::output_shape(op, &a, None)?;
                        code.push(Instruction::Apply {
                            op,
                            shape: s.clone(),
                        });
                        Ok(s)
                    }
                    "detach" => {
                        if args.len() != 1 {
                            return Err("detach expects one argument".into());
                        }
                        let shape = self.expr(&args[0], code)?;
                        code.push(Instruction::Detach);
                        Ok(shape)
                    }
                    "swap_axes" => {
                        if args.len() != 3 {
                            return Err("swap_axes expects tensor and two constant axes".into());
                        }
                        let first = axis_index(&args[1])?;
                        let second = axis_index(&args[2])?;
                        let input = self.expr(&args[0], code)?;
                        let op = Op::SwapAxes { first, second };
                        let shape = tensor::output_shape(op, &input, None)?;
                        code.push(Instruction::Apply {
                            op,
                            shape: shape.clone(),
                        });
                        Ok(shape)
                    }
                    "rows" => {
                        if args.len() != 3 {
                            return Err("rows expects tensor, constant start and count".into());
                        }
                        let start = row_start(&args[1])?;
                        let count = integer(&args[2])?;
                        let input = self.expr(&args[0], code)?;
                        let op = Op::Rows { start, count };
                        let shape = tensor::output_shape(op, &input, None)?;
                        code.push(Instruction::Apply {
                            op,
                            shape: shape.clone(),
                        });
                        Ok(shape)
                    }
                    "reshape" => {
                        if args.is_empty() {
                            return Err("reshape requires an input tensor".into());
                        }
                        let input = self.expr(&args[0], code)?;
                        let shape = args[1..]
                            .iter()
                            .map(|e| self.dimension(e))
                            .collect::<Result<Vec<_>, _>>()?;
                        if shape.len() > 8 {
                            return Err("reshape maximum rank is 8".into());
                        }
                        if tensor::elements(&input)? != tensor::elements(&shape)? {
                            return Err(format!(
                                "reshape cannot change element count: {input:?} to {shape:?}"
                            ));
                        }
                        code.push(Instruction::Apply {
                            op: Op::Reshape,
                            shape: shape.clone(),
                        });
                        Ok(shape)
                    }
                    "bce_logits" | "cross_entropy" | "bmm" => {
                        if args.len() != 2 {
                            return Err(format!("{name} expects two arguments"));
                        }
                        let a = self.expr(&args[0], code)?;
                        let b = self.expr(&args[1], code)?;
                        let op = if name == "bmm" {
                            Op::BatchMatmul
                        } else if name == "cross_entropy" {
                            Op::CrossEntropy
                        } else {
                            Op::BceLogits
                        };
                        let shape = tensor::output_shape(op, &a, Some(&b))?;
                        code.push(Instruction::Apply {
                            op,
                            shape: shape.clone(),
                        });
                        Ok(shape)
                    }
                    "mse" => {
                        if args.len() != 2 {
                            return Err("mse expects two arguments".into());
                        }
                        let a = self.expr(&args[0], code)?;
                        let b = self.expr(&args[1], code)?;
                        if a != b {
                            return Err(format!("mse shapes must match: {a:?} versus {b:?}"));
                        }
                        code.push(Instruction::Apply {
                            op: Op::Sub,
                            shape: a.clone(),
                        });
                        let slot = self.next;
                        self.next += 1;
                        code.push(Instruction::Store(slot, false));
                        code.push(Instruction::Load(slot));
                        code.push(Instruction::Load(slot));
                        code.push(Instruction::Apply {
                            op: Op::Mul,
                            shape: a,
                        });
                        code.push(Instruction::Apply {
                            op: Op::Mean,
                            shape: vec![],
                        });
                        Ok(vec![])
                    }
                    _ => Err(format!("unknown function '{name}'")),
                }
            }
        }
    }
    fn function_body(
        &mut self,
        function: &Function,
        code: &mut Vec<Instruction>,
    ) -> Result<Vec<usize>, String> {
        if function
            .locals
            .iter()
            .any(|stmt| !matches!(stmt, Stmt::Bind { param: false, .. }))
        {
            return Err("function bodies allow only let declarations and a final return".into());
        }
        self.body(&function.locals, code)?;
        self.expr(&function.body, code)
    }
    fn body(&mut self, stmts: &[Stmt], code: &mut Vec<Instruction>) -> Result<(), String> {
        for stmt in stmts {
            match stmt {
                Stmt::Bind {
                    name,
                    shape,
                    value,
                    param,
                } => {
                    if self.vars.contains_key(name) {
                        return Err(format!("duplicate binding '{name}'"));
                    }
                    if *param && self.in_loop {
                        return Err("parameters must be declared outside repeat blocks".into());
                    }
                    let actual = self.expr(value, code)?;
                    if let Some(annotation) = shape {
                        let expected = self.resolve(annotation)?;
                        if expected != actual {
                            return Err(format!("{name}: annotation Tensor{expected:?} does not match Tensor{actual:?}"));
                        }
                    }
                    let slot = self.next;
                    self.next += 1;
                    self.vars.insert(name.clone(), (slot, actual));
                    if *param {
                        self.parameters.insert(name.clone(), slot);
                    }
                    code.push(Instruction::Store(slot, *param));
                }
                Stmt::Repeat(count, stmts) => {
                    let saved = self.vars.clone();
                    let first = self.next;
                    let was = self.in_loop;
                    self.in_loop = true;
                    let mut body = vec![];
                    self.body(stmts, &mut body)?;
                    self.in_loop = was;
                    self.vars = saved;
                    code.push(Instruction::Repeat {
                        count: *count,
                        body,
                        locals: (first..self.next).collect(),
                    });
                }
                Stmt::Call(name, args) => {
                    let expected = match name.as_str() {
                        "print" | "backward" | "step" | "clip_grad_norm" => 1,
                        "adam_step" => {
                            if args.len() == 4 {
                                4
                            } else {
                                1
                            }
                        }
                        "assert_close" => 3,
                        _ => return Err(format!("unknown statement '{name}'")),
                    };
                    if args.len() != expected {
                        return Err(format!("{name} expects {expected} arguments"));
                    }
                    let mut shapes = vec![];
                    for e in args {
                        shapes.push(self.expr(e, code)?);
                    }
                    match name.as_str() {
                        "print" => code.push(Instruction::Print),
                        "backward" => {
                            if !shapes[0].is_empty() {
                                return Err("backward requires a scalar loss".into());
                            }
                            code.push(Instruction::Backward);
                        }
                        "adam_step" => {
                            if shapes.iter().any(|shape| !shape.is_empty()) {
                                return Err("adam_step arguments must be scalars".into());
                            }
                            code.push(Instruction::AdamStep {
                                custom: args.len() == 4,
                            });
                        }
                        "clip_grad_norm" => {
                            if !shapes[0].is_empty() {
                                return Err("clip_grad_norm requires a scalar maximum norm".into());
                            }
                            code.push(Instruction::ClipGradNorm);
                        }
                        "step" => {
                            if !shapes[0].is_empty() {
                                return Err("step requires scalar learning rate".into());
                            }
                            code.push(Instruction::Step);
                        }
                        _ => {
                            if shapes[0] != shapes[1] || !shapes[2].is_empty() {
                                return Err(
                                    "assert_close needs equal shapes and scalar tolerance".into()
                                );
                            }
                            code.push(Instruction::AssertClose);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
pub fn compile(ast: Ast) -> Result<Program, String> {
    let mut functions = HashMap::new();
    let reserved = [
        "adam_step",
        "detach",
        "rows",
        "swap_axes",
        "clip_grad_norm",
        "softmax",
        "causal_softmax",
        "layer_norm",
        "gelu",
        "cross_entropy",
        "sigmoid",
        "transpose",
        "transpose_last",
        "bmm",
        "reshape",
        "bce_logits",
        "zeros",
        "ones",
        "rand",
        "tanh",
        "relu",
        "sum",
        "mean",
        "grad",
        "mse",
        "print",
        "step",
        "backward",
        "assert_close",
    ];
    for f in ast.functions {
        if reserved.contains(&f.name.as_str()) || functions.contains_key(&f.name) {
            return Err(format!("duplicate or reserved function '{}'", f.name));
        }
        let mut names = HashSet::new();
        for (n, s) in &f.args {
            if !names.insert(n) {
                return Err(format!("duplicate argument '{n}'"));
            }
            crate::typecheck::validate_shape(s)?;
        }
        crate::typecheck::validate_shape(&f.result)?;
        functions.insert(f.name.clone(), f);
    }
    crate::typecheck::check_functions(&functions)?;
    let mut c = Compiler {
        vars: HashMap::new(),
        functions,
        next: 0,
        calls: vec![],
        in_loop: false,
        dimensions: HashMap::new(),
        parameters: HashMap::new(),
    };
    let mut code = vec![];
    c.body(&ast.body, &mut code)?;
    Ok(Program {
        code,
        slots: c.next,
        parameters: c.parameters,
        bindings: c.vars.into_iter().map(|(n, (i, _))| (n, i)).collect(),
    })
}
