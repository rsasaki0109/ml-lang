//! Dense row-major f64 CPU tensors and differentiable primitive kernels.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}
pub const MAX_ELEMENTS: usize = 16_777_216;
pub fn elements(shape: &[usize]) -> Result<usize, String> {
    let n = shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or("tensor size overflow")?;
    if shape.contains(&0) || n > MAX_ELEMENTS {
        return Err(format!("tensor must have 1..={MAX_ELEMENTS} elements"));
    }
    Ok(n)
}
impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Result<Self, String> {
        if elements(&shape)? != data.len() {
            return Err("tensor data length does not match shape".into());
        }
        if data.iter().any(|x| !x.is_finite()) {
            return Err("tensor contains non-finite values".into());
        }
        Ok(Self { shape, data })
    }
    pub fn scalar(x: f64) -> Self {
        Self {
            shape: vec![],
            data: vec![x],
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Matmul,
    BatchMatmul,
    Tanh,
    Gelu,
    Sigmoid,
    Transpose,
    TransposeLast,
    SwapAxes { first: usize, second: usize },
    Reshape,
    Rows { start: usize, count: usize },
    BceLogits,
    Softmax,
    CausalSoftmax,
    LayerNorm,
    CrossEntropy,
    Relu,
    Sum,
    Mean,
    Neg,
}
pub fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let rank = a.len().max(b.len());
    let mut out = vec![1; rank];
    for (i, d) in out.iter_mut().enumerate() {
        let x = if i + a.len() >= rank {
            a[i + a.len() - rank]
        } else {
            1
        };
        let y = if i + b.len() >= rank {
            b[i + b.len() - rank]
        } else {
            1
        };
        if x != y && x != 1 && y != 1 {
            return Err(format!("cannot broadcast Tensor{a:?} and Tensor{b:?}"));
        }
        *d = x.max(y);
    }
    elements(&out)?;
    Ok(out)
}
pub fn output_shape(op: Op, a: &[usize], b: Option<&[usize]>) -> Result<Vec<usize>, String> {
    match op {
        Op::BatchMatmul => {
            let b = b.ok_or("bmm needs two inputs")?;
            if a.len() < 3
                || !(b.len() == 2 || b.len() == a.len())
                || a[a.len() - 1] != b[b.len() - 2]
                || (b.len() != 2 && a[..a.len() - 2] != b[..b.len() - 2])
            {
                return Err(format!(
                    "bmm requires [...,M,K] with shared [K,N] or matching [...,K,N], got {a:?}, {b:?}"
                ));
            }
            let mut shape = a.to_vec();
            *shape.last_mut().unwrap() = b[b.len() - 1];
            elements(&shape)?;
            Ok(shape)
        }
        Op::SwapAxes { first, second } => {
            if first >= a.len() || second >= a.len() {
                return Err("swap_axes axis out of bounds".into());
            }
            let mut shape = a.to_vec();
            shape.swap(first, second);
            Ok(shape)
        }
        Op::TransposeLast => {
            if a.len() < 2 {
                return Err("transpose_last requires rank >= 2".into());
            }
            let mut shape = a.to_vec();
            let rank = shape.len();
            shape.swap(rank - 2, rank - 1);
            Ok(shape)
        }
        Op::Rows { start, count } => {
            if a.is_empty() || count == 0 || start.checked_add(count).is_none_or(|end| end > a[0]) {
                return Err("rows requires a nonempty in-bounds leading-axis range".into());
            }
            let mut shape = a.to_vec();
            shape[0] = count;
            elements(&shape)?;
            Ok(shape)
        }
        Op::Matmul => {
            let b = b.ok_or("matmul needs two inputs")?;
            if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
                return Err(format!(
                    "matmul requires [m,k] @ [k,n], got {a:?} and {b:?}"
                ));
            }
            let s = vec![a[0], b[1]];
            elements(&s)?;
            Ok(s)
        }
        Op::Transpose => {
            if a.len() != 2 {
                return Err("transpose requires a rank-2 tensor".into());
            }
            Ok(vec![a[1], a[0]])
        }
        Op::CausalSoftmax => {
            if a.len() < 2 || a[a.len() - 2] != a[a.len() - 1] {
                return Err("causal_softmax requires rank >= 2 with square final axes".into());
            }
            Ok(a.to_vec())
        }
        Op::LayerNorm => {
            if a.is_empty() {
                return Err("layer_norm requires rank >= 1".into());
            }
            Ok(a.to_vec())
        }
        Op::Softmax => {
            if a.is_empty() {
                return Err("softmax requires rank >= 1".into());
            }
            Ok(a.to_vec())
        }
        Op::CrossEntropy => {
            let b = b.ok_or("cross_entropy requires logits and targets")?;
            if a.len() != 2 || a != b {
                return Err(format!(
                    "cross_entropy requires equal [batch,classes] shapes, got {a:?} and {b:?}"
                ));
            }
            Ok(vec![])
        }
        Op::BceLogits => {
            let b = b.ok_or("bce_logits requires logits and targets")?;
            if a != b {
                return Err(format!("bce_logits shapes must match: {a:?} versus {b:?}"));
            }
            Ok(vec![])
        }
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            broadcast(a, b.ok_or("binary operation needs two inputs")?)
        }
        Op::Sum | Op::Mean => Ok(vec![]),
        _ => Ok(a.to_vec()),
    }
}
fn index(mut flat: usize, out: &[usize], input: &[usize]) -> usize {
    let offset = out.len() - input.len();
    let (mut idx, mut stride) = (0, 1);
    for axis in (0..out.len()).rev() {
        let coord = flat % out[axis];
        flat /= out[axis];
        if axis >= offset {
            let d = input[axis - offset];
            if d != 1 {
                idx += coord * stride;
            }
            stride *= d;
        }
    }
    idx
}
pub fn forward(op: Op, a: &Tensor, b: Option<&Tensor>) -> Result<Tensor, String> {
    let shape = output_shape(op, &a.shape, b.map(|x| x.shape.as_slice()))?;
    let n = elements(&shape)?;
    let mut data = vec![0.0; n];
    match op {
        Op::SwapAxes { first, second } => {
            swap_data(&a.data, &mut data, &shape, first, second);
        }

        Op::LayerNorm => {
            let width = *a.shape.last().unwrap();
            for (row, result) in a.data.chunks_exact(width).zip(data.chunks_exact_mut(width)) {
                let (scale, mean, denominator) = layer_norm_stats(row);
                for (dst, &x) in result.iter_mut().zip(row) {
                    *dst = (x / scale - mean) / denominator;
                }
            }
        }

        Op::BatchMatmul => {
            let b = b.unwrap();
            let rank = a.shape.len();
            let (batch, m, k, n) = (
                a.shape[..rank - 2].iter().product::<usize>(),
                a.shape[rank - 2],
                a.shape[rank - 1],
                *b.shape.last().unwrap(),
            );
            for batch_index in 0..batch {
                let ao = batch_index * m * k;
                let bo = if b.shape.len() == 2 {
                    0
                } else {
                    batch_index * k * n
                };
                let co = batch_index * m * n;
                for i in 0..m {
                    for p in 0..k {
                        let value = a.data[ao + i * k + p];
                        for j in 0..n {
                            data[co + i * n + j] += value * b.data[bo + p * n + j];
                        }
                    }
                }
            }
        }
        Op::TransposeLast => {
            let rank = a.shape.len();
            let (rows, cols) = (a.shape[rank - 2], a.shape[rank - 1]);
            let width = rows * cols;
            for offset in (0..a.data.len()).step_by(width) {
                for i in 0..rows {
                    for j in 0..cols {
                        data[offset + j * rows + i] = a.data[offset + i * cols + j];
                    }
                }
            }
        }
        Op::Rows { start, count } => {
            let width = a.data.len() / a.shape[0];
            data.copy_from_slice(&a.data[start * width..(start + count) * width]);
        }
        Op::Matmul => {
            let b = b.unwrap();
            let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
            for i in 0..m {
                for p in 0..k {
                    let v = a.data[i * k + p];
                    for j in 0..n {
                        data[i * n + j] += v * b.data[p * n + j];
                    }
                }
            }
        }
        Op::Transpose => {
            let (rows, cols) = (a.shape[0], a.shape[1]);
            for i in 0..rows {
                for j in 0..cols {
                    data[j * rows + i] = a.data[i * cols + j];
                }
            }
        }
        Op::Reshape => return Err("reshape requires a target shape; use tensor::reshape".into()),
        Op::Softmax | Op::CausalSoftmax => {
            let classes = *a.shape.last().unwrap();
            for (index, (row, result)) in a
                .data
                .chunks_exact(classes)
                .zip(data.chunks_exact_mut(classes))
                .enumerate()
            {
                let visible = if op == Op::CausalSoftmax {
                    index % classes + 1
                } else {
                    classes
                };
                for (value, log_probability) in result[..visible]
                    .iter_mut()
                    .zip(log_probabilities(&row[..visible]))
                {
                    *value = log_probability.exp();
                }
            }
        }
        Op::CrossEntropy => {
            let b = b.unwrap();
            let (batch, classes) = (a.shape[0], a.shape[1]);
            for (row, target) in a
                .data
                .chunks_exact(classes)
                .zip(b.data.chunks_exact(classes))
            {
                if target.iter().any(|y| !(0.0..=1.0).contains(y))
                    || (target.iter().sum::<f64>() - 1.0).abs() > 1e-9
                {
                    return Err("cross_entropy target rows must be probabilities in [0,1] summing to 1 (tolerance 1e-9)".into());
                }
                for (&y, log_probability) in target.iter().zip(log_probabilities(row)) {
                    // Skip zero targets to avoid 0 * infinity for unrepresentable gaps.
                    if y != 0.0 {
                        data[0] -= y * log_probability / batch as f64;
                    }
                }
            }
        }
        Op::BceLogits => {
            let b = b.unwrap();
            if b.data.iter().any(|y| !(0.0..=1.0).contains(y)) {
                return Err("bce_logits targets must be in [0,1]".into());
            }
            data[0] = a
                .data
                .iter()
                .zip(&b.data)
                .map(|(&z, &y)| z.max(0.0) - z * y + (-z.abs()).exp().ln_1p())
                .sum::<f64>()
                / a.data.len() as f64;
        }
        Op::Sum | Op::Mean => {
            data[0] = a.data.iter().sum::<f64>();
            if op == Op::Mean {
                data[0] /= a.data.len() as f64;
            }
        }
        Op::Gelu | Op::Tanh | Op::Sigmoid | Op::Relu | Op::Neg => {
            for (o, &v) in data.iter_mut().zip(&a.data) {
                *o = match op {
                    Op::Tanh => v.tanh(),
                    Op::Gelu => gelu(v).0,
                    Op::Sigmoid => sigmoid(v),
                    Op::Relu => v.max(0.0),
                    _ => -v,
                };
            }
        }
        _ => {
            let b = b.unwrap();
            for (i, o) in data.iter_mut().enumerate() {
                let x = a.data[index(i, &shape, &a.shape)];
                let y = b.data[index(i, &shape, &b.shape)];
                *o = match op {
                    Op::Add => x + y,
                    Op::Sub => x - y,
                    Op::Mul => x * y,
                    Op::Div => x / y,
                    _ => unreachable!(),
                };
            }
        }
    }
    Tensor::new(shape, data)
}
pub fn vjp(
    op: Op,
    a: &Tensor,
    b: Option<&Tensor>,
    out: &Tensor,
    g: &[f64],
) -> (Vec<f64>, Option<Vec<f64>>) {
    let mut ga = vec![0.0; a.data.len()];
    let mut gb = b.map(|b| vec![0.0; b.data.len()]);
    match op {
        Op::SwapAxes { first, second } => {
            swap_data(g, &mut ga, &a.shape, first, second);
        }

        Op::LayerNorm => {
            let width = *a.shape.last().unwrap();
            for (((row, normalized), upstream), gradient) in a
                .data
                .chunks_exact(width)
                .zip(out.data.chunks_exact(width))
                .zip(g.chunks_exact(width))
                .zip(ga.chunks_exact_mut(width))
            {
                let (scale, _, denominator) = layer_norm_stats(row);
                let inverse_std = (1.0 / scale) / denominator;
                let mean_g = upstream.iter().map(|v| v / width as f64).sum::<f64>();
                let mean_gy = upstream
                    .iter()
                    .zip(normalized)
                    .map(|(g, y)| g * y / width as f64)
                    .sum::<f64>();
                for ((dst, &g), &y) in gradient.iter_mut().zip(upstream).zip(normalized) {
                    *dst = inverse_std * (g - mean_g - y * mean_gy);
                }
            }
        }

        Op::BatchMatmul => {
            let b = b.unwrap();
            let db = gb.as_mut().unwrap();
            let rank = a.shape.len();
            let (batch, m, k, n) = (
                a.shape[..rank - 2].iter().product::<usize>(),
                a.shape[rank - 2],
                a.shape[rank - 1],
                *b.shape.last().unwrap(),
            );
            for batch_index in 0..batch {
                let ao = batch_index * m * k;
                let bo = if b.shape.len() == 2 {
                    0
                } else {
                    batch_index * k * n
                };
                let co = batch_index * m * n;
                for i in 0..m {
                    for p in 0..k {
                        for j in 0..n {
                            ga[ao + i * k + p] += g[co + i * n + j] * b.data[bo + p * n + j];
                            db[bo + p * n + j] += a.data[ao + i * k + p] * g[co + i * n + j];
                        }
                    }
                }
            }
        }
        Op::TransposeLast => {
            let rank = a.shape.len();
            let (rows, cols) = (a.shape[rank - 2], a.shape[rank - 1]);
            let width = rows * cols;
            for offset in (0..a.data.len()).step_by(width) {
                for i in 0..rows {
                    for j in 0..cols {
                        ga[offset + i * cols + j] = g[offset + j * rows + i];
                    }
                }
            }
        }
        Op::Rows { start, count } => {
            let width = a.data.len() / a.shape[0];
            ga[start * width..(start + count) * width].copy_from_slice(g);
        }
        Op::Matmul => {
            let b = b.unwrap();
            let db = gb.as_mut().unwrap();
            let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
            for i in 0..m {
                for p in 0..k {
                    for j in 0..n {
                        ga[i * k + p] += g[i * n + j] * b.data[p * n + j];
                        db[p * n + j] += a.data[i * k + p] * g[i * n + j];
                    }
                }
            }
        }
        Op::Transpose => {
            let (rows, cols) = (a.shape[0], a.shape[1]);
            for i in 0..rows {
                for j in 0..cols {
                    ga[i * cols + j] = g[j * rows + i];
                }
            }
        }
        Op::Reshape => ga.copy_from_slice(g),
        Op::Softmax | Op::CausalSoftmax => {
            let classes = *a.shape.last().unwrap();
            for (index, ((prob, upstream), gradient)) in out
                .data
                .chunks_exact(classes)
                .zip(g.chunks_exact(classes))
                .zip(ga.chunks_exact_mut(classes))
                .enumerate()
            {
                let visible = if op == Op::CausalSoftmax {
                    index % classes + 1
                } else {
                    classes
                };
                let (prob, upstream, gradient) = (
                    &prob[..visible],
                    &upstream[..visible],
                    &mut gradient[..visible],
                );
                let dot = prob.iter().zip(upstream).map(|(p, g)| p * g).sum::<f64>();
                for ((dst, &p), &v) in gradient.iter_mut().zip(prob).zip(upstream) {
                    *dst = p * (v - dot);
                }
            }
        }
        Op::CrossEntropy => {
            let b = b.unwrap();
            let db = gb.as_mut().unwrap();
            let classes = a.shape[1];
            let scale = g[0] / a.shape[0] as f64;
            for (r, (row, target)) in a
                .data
                .chunks_exact(classes)
                .zip(b.data.chunks_exact(classes))
                .enumerate()
            {
                let mass = target.iter().sum::<f64>();
                for (c, log_probability) in log_probabilities(row).into_iter().enumerate() {
                    ga[r * classes + c] = scale * (log_probability.exp() * mass - target[c]);
                    db[r * classes + c] = -scale * log_probability;
                }
            }
        }
        Op::BceLogits => {
            let b = b.unwrap();
            let db = gb.as_mut().unwrap();
            let scale = g[0] / a.data.len() as f64;
            for i in 0..ga.len() {
                ga[i] = scale * (sigmoid(a.data[i]) - b.data[i]);
                db[i] = -scale * a.data[i];
            }
        }
        Op::Sum | Op::Mean => {
            let scale = if op == Op::Mean {
                1.0 / a.data.len() as f64
            } else {
                1.0
            };
            ga.fill(g[0] * scale);
        }
        Op::Gelu | Op::Tanh | Op::Sigmoid | Op::Relu | Op::Neg => {
            for i in 0..ga.len() {
                ga[i] = g[i]
                    * match op {
                        Op::Tanh => 1.0 - out.data[i] * out.data[i],
                        Op::Gelu => gelu(a.data[i]).1,
                        Op::Sigmoid => out.data[i] * (1.0 - out.data[i]),
                        Op::Relu => {
                            if a.data[i] > 0.0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => -1.0,
                    };
            }
        }
        _ => {
            let b = b.unwrap();
            let db = gb.as_mut().unwrap();
            for (i, &v) in g.iter().enumerate() {
                let ai = index(i, &out.shape, &a.shape);
                let bi = index(i, &out.shape, &b.shape);
                let x = a.data[ai];
                let y = b.data[bi];
                let (dx, dy) = match op {
                    Op::Add => (1.0, 1.0),
                    Op::Sub => (1.0, -1.0),
                    Op::Mul => (y, x),
                    Op::Div => (1.0 / y, -x / (y * y)),
                    _ => unreachable!(),
                };
                ga[ai] += v * dx;
                db[bi] += v * dy;
            }
        }
    }
    (ga, gb)
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Reshape preserves row-major element order and copies storage in the CPU prototype.
pub fn reshape(a: &Tensor, shape: &[usize]) -> Result<Tensor, String> {
    Tensor::new(shape.to_vec(), a.data.clone())
}

// Subtract the row maximum before exponentiation and retain shifted log values.
// Avoid max + log(sum(exp(...))) - z, which loses log normalization at huge offsets.
fn log_probabilities(row: &[f64]) -> Vec<f64> {
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let normalization = row.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    row.iter().map(|x| (x - max) - normalization).collect()
}

/// Population variance with fixed epsilon, scaled to avoid overflow of squares.
/// Inputs are finite valid tensors. Scale is at least sqrt(epsilon), avoiding
/// division overflow for tiny inputs and preserving constant-row derivatives.
fn layer_norm_stats(row: &[f64]) -> (f64, f64, f64) {
    let sqrt_epsilon = 1e-5_f64.sqrt();
    let scale = row.iter().map(|x| x.abs()).fold(sqrt_epsilon, f64::max);
    let anchor = row[0] / scale;
    let mean = anchor
        + row
            .iter()
            .map(|x| (x / scale - anchor) / row.len() as f64)
            .sum::<f64>();
    let variance = row
        .iter()
        .map(|x| {
            let d = x / scale - mean;
            d * d / row.len() as f64
        })
        .sum::<f64>();
    (scale, mean, variance.sqrt().hypot(sqrt_epsilon / scale))
}

// Swapped axes leave their trailing suffix contiguous. Precompute strides once,
// then copy complete suffix blocks without allocating metadata per element.
fn swap_data(input: &[f64], output: &mut [f64], shape: &[usize], first: usize, second: usize) {
    if first == second {
        output.copy_from_slice(input);
        return;
    }
    let last = first.max(second);
    let block = shape[last + 1..].iter().product::<usize>();
    let mut input_shape = shape[..=last].to_vec();
    input_shape.swap(first, second);
    let mut strides = vec![1; last + 1];
    for axis in (0..last).rev() {
        strides[axis] = strides[axis + 1] * input_shape[axis + 1];
    }
    strides.swap(first, second);
    for (index, dst) in output.chunks_exact_mut(block).enumerate() {
        let mut flat = index;
        let mut offset = 0;
        for axis in (0..=last).rev() {
            offset += (flat % shape[axis]) * strides[axis];
            flat /= shape[axis];
        }
        dst.copy_from_slice(&input[offset * block..(offset + 1) * block]);
    }
}

// Standard tanh approximation, not the exact erf GELU. Saturated tails avoid
// overflowing the cubic for very large finite inputs.
fn gelu(x: f64) -> (f64, f64) {
    if x > 10.0 {
        return (x, 1.0);
    }
    if x < -10.0 {
        return (0.0, 0.0);
    }
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let t = (c * (x + 0.044715 * x * x * x)).tanh();
    (
        0.5 * x * (1.0 + t),
        0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x * x),
    )
}
