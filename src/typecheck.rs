//! Structural symbolic shape checking: functions are valid before specialization.
use crate::{
    syntax::{Dim, Expr, Function, Shape, Stmt},
    tensor::{self, Op},
};
use std::collections::{HashMap, HashSet};

pub(crate) fn validate_shape(shape: &Shape) -> Result<(), String> {
    let minimum: Vec<_> = shape
        .iter()
        .map(|d| match d {
            Dim::Fixed(n) => *n,
            Dim::Named(_) => 1,
        })
        .collect();
    tensor::elements(&minimum)?;
    Ok(())
}
fn names(shape: &Shape) -> impl Iterator<Item = &String> {
    shape
        .iter()
        .filter_map(|d| if let Dim::Named(n) = d { Some(n) } else { None })
}
fn resolved(shape: &Shape, bindings: &HashMap<String, Dim>) -> Result<Shape, String> {
    shape
        .iter()
        .map(|d| match d {
            Dim::Fixed(_) => Ok(d.clone()),
            Dim::Named(n) => bindings
                .get(n)
                .cloned()
                .ok_or_else(|| format!("unbound shape dimension '{n}'")),
        })
        .collect()
}
fn product(shape: &Shape) -> Result<(usize, Vec<String>), String> {
    let mut constant = 1usize;
    let mut symbols = vec![];
    for d in shape {
        match d {
            Dim::Fixed(n) => constant = constant.checked_mul(*n).ok_or("shape size overflow")?,
            Dim::Named(n) => symbols.push(n.clone()),
        }
    }
    symbols.sort();
    Ok((constant, symbols))
}
fn output(op: Op, a: &Shape, b: Option<&Shape>) -> Result<Shape, String> {
    match op {
        Op::BatchMatmul => {
            let b = b.unwrap();
            if a.len() < 3
                || !(b.len() == 2 || b.len() == a.len())
                || a[a.len() - 1] != b[b.len() - 2]
                || (b.len() != 2 && a[..a.len() - 2] != b[..b.len() - 2])
            {
                return Err("bmm requires [...,M,K] and shared [K,N] or matching [...,K,N]".into());
            }
            let mut shape = a.clone();
            *shape.last_mut().unwrap() = b[b.len() - 1].clone();
            Ok(shape)
        }
        Op::TransposeLast => {
            if a.len() < 2 {
                return Err("transpose_last requires rank >= 2".into());
            }
            let mut shape = a.clone();
            let rank = shape.len();
            shape.swap(rank - 2, rank - 1);
            Ok(shape)
        }
        Op::Matmul => {
            let b = b.unwrap();
            if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
                return Err(format!(
                    "matmul requires matching symbolic inner dimensions: {a:?}, {b:?}"
                ));
            }
            Ok(vec![a[0].clone(), b[1].clone()])
        }
        Op::Transpose => {
            if a.len() != 2 {
                return Err("transpose requires rank 2".into());
            }
            Ok(vec![a[1].clone(), a[0].clone()])
        }
        Op::CausalSoftmax => {
            if a.len() < 2 || a[a.len() - 2] != a[a.len() - 1] {
                return Err("causal_softmax requires matching final square dimensions".into());
            }
            Ok(a.clone())
        }
        Op::LayerNorm => {
            if a.is_empty() {
                return Err("layer_norm requires rank >= 1".into());
            }
            Ok(a.clone())
        }
        Op::Softmax => {
            if a.is_empty() {
                return Err("softmax requires rank >= 1".into());
            }
            Ok(a.clone())
        }
        Op::BceLogits | Op::CrossEntropy => {
            if Some(a) != b || (op == Op::CrossEntropy && a.len() != 2) {
                return Err("loss argument shapes must match (cross_entropy needs rank 2)".into());
            }
            Ok(vec![])
        }
        Op::Sum | Op::Mean => Ok(vec![]),
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            let b = b.unwrap();
            let rank = a.len().max(b.len());
            let mut shape = vec![];
            for i in 0..rank {
                let x = if i + a.len() >= rank {
                    a[i + a.len() - rank].clone()
                } else {
                    Dim::Fixed(1)
                };
                let y = if i + b.len() >= rank {
                    b[i + b.len() - rank].clone()
                } else {
                    Dim::Fixed(1)
                };
                shape.push(if x == y || y == Dim::Fixed(1) {
                    x
                } else if x == Dim::Fixed(1) {
                    y
                } else {
                    return Err(format!(
                        "cannot prove broadcast of dimensions {x:?} and {y:?}"
                    ));
                });
            }
            Ok(shape)
        }
        _ => Ok(a.clone()),
    }
}
struct Checker<'a> {
    functions: &'a HashMap<String, Function>,
    vars: HashMap<String, Shape>,
    dimensions: HashMap<String, Dim>,
}
impl Checker<'_> {
    fn dimension(&self, e: &Expr) -> Result<Dim, String> {
        match e {
            Expr::Var(n) => self
                .dimensions
                .get(n)
                .cloned()
                .ok_or_else(|| format!("unbound shape dimension '{n}'")),
            Expr::Number(n) if *n > 0.0 && n.fract() == 0.0 && *n <= 1_000_000.0 => {
                Ok(Dim::Fixed(*n as usize))
            }
            _ => Err("dimension must be a positive constant or bound shape name".into()),
        }
    }
    fn expr(&self, e: &Expr) -> Result<Shape, String> {
        let shape = match e {
            Expr::Number(_) | Expr::Array(_) => crate::compiler::literal(e)?
                .shape
                .into_iter()
                .map(Dim::Fixed)
                .collect(),
            Expr::Var(n) => self
                .vars
                .get(n)
                .cloned()
                .ok_or_else(|| format!("undefined variable '{n}'"))?,
            Expr::Binary(c, a, b) => {
                let a = self.expr(a)?;
                let b = self.expr(b)?;
                output(
                    match c {
                        '+' => Op::Add,
                        '-' => Op::Sub,
                        '*' => Op::Mul,
                        '/' => Op::Div,
                        '@' => Op::Matmul,
                        _ => unreachable!(),
                    },
                    &a,
                    Some(&b),
                )?
            }
            Expr::Call(n, args) => {
                if let Some(f) = self.functions.get(n) {
                    if args.len() != f.args.len() {
                        return Err(format!("{n}: wrong argument count"));
                    }
                    let mut dimensions = HashMap::new();
                    for (e, (_, pattern)) in args.iter().zip(&f.args) {
                        let actual = self.expr(e)?;
                        if actual.len() != pattern.len() {
                            return Err(format!("{n}: rank mismatch"));
                        }
                        for (expected, actual) in pattern.iter().zip(actual) {
                            match expected {
                                Dim::Fixed(_) if expected != &actual => {
                                    return Err(format!("{n}: dimension mismatch"))
                                }
                                Dim::Named(s) => {
                                    if let Some(old) = dimensions.insert(s.clone(), actual.clone())
                                    {
                                        if old != actual {
                                            return Err(format!(
                                                "{n}: inconsistent dimension '{s}'"
                                            ));
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    resolved(&f.result, &dimensions)?
                } else {
                    match n.as_str() {
                        "zeros" | "ones" | "rand" => {
                            let dims = if n == "rand" {
                                if args.len() < 2 {
                                    return Err("rand needs dimensions and constant seed".into());
                                }
                                if !matches!(self.dimension(args.last().unwrap())?, Dim::Fixed(_)) {
                                    return Err("random seed must be constant".into());
                                }
                                &args[..args.len() - 1]
                            } else {
                                args.as_slice()
                            };
                            if dims.is_empty() {
                                return Err("initializer requires dimensions".into());
                            }
                            dims.iter()
                                .map(|e| self.dimension(e))
                                .collect::<Result<_, _>>()?
                        }
                        "detach" => {
                            if args.len() != 1 {
                                return Err("detach expects one argument".into());
                            }
                            self.expr(&args[0])?
                        }
                        "swap_axes" => {
                            if args.len() != 3 {
                                return Err("swap_axes expects tensor and two constant axes".into());
                            }
                            let mut shape = self.expr(&args[0])?;
                            let first = crate::compiler::axis_index(&args[1])?;
                            let second = crate::compiler::axis_index(&args[2])?;
                            if first >= shape.len() || second >= shape.len() {
                                return Err("swap_axes axis out of bounds".into());
                            }
                            shape.swap(first, second);
                            shape
                        }
                        "rows" => {
                            if args.len() != 3 {
                                return Err("rows expects tensor, constant start and count".into());
                            }
                            let mut shape = self.expr(&args[0])?;
                            let start = crate::compiler::row_start(&args[1])?;
                            let Dim::Fixed(count) = self.dimension(&args[2])? else {
                                return Err("rows count must be constant".into());
                            };
                            let Some(Dim::Fixed(size)) = shape.first() else {
                                return Err(
                                    "rows needs a statically known leading dimension".into()
                                );
                            };
                            if start.checked_add(count).is_none_or(|end| end > *size) {
                                return Err("rows range is out of bounds".into());
                            }
                            shape[0] = Dim::Fixed(count);
                            shape
                        }
                        "reshape" => {
                            if args.is_empty() {
                                return Err("reshape requires an input".into());
                            }
                            let a = self.expr(&args[0])?;
                            let s = args[1..]
                                .iter()
                                .map(|e| self.dimension(e))
                                .collect::<Result<Shape, _>>()?;
                            if s.len() > 8 || product(&a)? != product(&s)? {
                                return Err(
                                    "reshape must preserve symbolic element count and rank limit"
                                        .into(),
                                );
                            }
                            s
                        }
                        "gelu" | "tanh" | "relu" | "sigmoid" | "softmax" | "causal_softmax"
                        | "layer_norm" | "transpose" | "transpose_last" | "sum" | "mean"
                        | "grad" => {
                            if args.len() != 1 {
                                return Err(format!("{n}: expected one argument"));
                            }
                            let a = self.expr(&args[0])?;
                            let op = match n.as_str() {
                                "tanh" => Op::Tanh,
                                "gelu" => Op::Gelu,
                                "relu" => Op::Relu,
                                "sigmoid" => Op::Sigmoid,
                                "softmax" => Op::Softmax,
                                "causal_softmax" => Op::CausalSoftmax,
                                "layer_norm" => Op::LayerNorm,
                                "transpose" => Op::Transpose,
                                "transpose_last" => Op::TransposeLast,
                                "sum" => Op::Sum,
                                "mean" => Op::Mean,
                                _ => Op::Reshape,
                            };
                            output(op, &a, None)?
                        }
                        "mse" | "bce_logits" | "cross_entropy" | "bmm" => {
                            if args.len() != 2 {
                                return Err(format!("{n}: expected two arguments"));
                            }
                            let a = self.expr(&args[0])?;
                            let b = self.expr(&args[1])?;
                            output(
                                if n == "bmm" {
                                    Op::BatchMatmul
                                } else if n == "cross_entropy" {
                                    Op::CrossEntropy
                                } else {
                                    Op::BceLogits
                                },
                                &a,
                                Some(&b),
                            )?
                        }
                        _ => return Err(format!("unknown function '{n}'")),
                    }
                }
            }
        };
        validate_shape(&shape)?;
        Ok(shape)
    }
}
fn calls(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Call(n, args) => {
            out.insert(n.clone());
            for e in args {
                calls(e, out);
            }
        }
        Expr::Binary(_, a, b) => {
            calls(a, out);
            calls(b, out);
        }
        Expr::Array(xs) => {
            for x in xs {
                calls(x, out);
            }
        }
        _ => {}
    }
}
fn visit(
    name: &str,
    functions: &HashMap<String, Function>,
    active: &mut Vec<String>,
    done: &mut HashMap<String, usize>,
) -> Result<usize, String> {
    if active.iter().any(|n| n == name) || active.len() >= 32 {
        return Err("recursive or excessively nested functions are unsupported".into());
    }
    if let Some(height) = done.get(name) {
        return Ok(*height);
    }
    active.push(name.into());
    let f = &functions[name];
    let mut dependencies = HashSet::new();
    calls(&f.body, &mut dependencies);
    for local in &f.locals {
        if let Stmt::Bind { value, .. } = local {
            calls(value, &mut dependencies);
        }
    }
    let mut height = 1usize;
    for dependency in dependencies {
        if functions.contains_key(&dependency) {
            height = height.max(1 + visit(&dependency, functions, active, done)?);
        }
    }
    active.pop();
    if height > 32 {
        return Err("function call depth exceeds 32".into());
    }
    done.insert(name.into(), height);
    Ok(height)
}
pub(crate) fn check_functions(functions: &HashMap<String, Function>) -> Result<(), String> {
    let mut done = HashMap::new();
    for name in functions.keys() {
        visit(name, functions, &mut vec![], &mut done)?;
    }
    for f in functions.values() {
        let bound: HashSet<String> = f.args.iter().flat_map(|(_, s)| names(s).cloned()).collect();
        for n in names(&f.result) {
            if !bound.contains(n) {
                return Err(format!("{}: unbound return dimension '{n}'", f.name));
            }
        }
        let mut checker = Checker {
            functions,
            vars: f.args.iter().cloned().collect(),
            dimensions: bound
                .into_iter()
                .map(|n| (n.clone(), Dim::Named(n)))
                .collect(),
        };
        for local in &f.locals {
            let Stmt::Bind {
                name,
                shape,
                value,
                param: false,
            } = local
            else {
                return Err("function bodies allow only let declarations".into());
            };
            if checker.vars.contains_key(name) {
                return Err(format!("duplicate binding '{name}'"));
            }
            let inferred = checker.expr(value)?;
            if let Some(annotation) = shape {
                let expected = resolved(annotation, &checker.dimensions)?;
                if inferred != expected {
                    return Err(format!("{name}: local shape annotation mismatch"));
                }
            }
            checker.vars.insert(name.clone(), inferred);
        }
        let actual = checker.expr(&f.body)?;
        if actual != f.result {
            return Err(format!(
                "{}: return shape {actual:?} does not match {:?}",
                f.name, f.result
            ));
        }
    }
    Ok(())
}
