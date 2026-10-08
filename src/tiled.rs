//! Safe, single-threaded row-major CPU f64 GEMM, with fixed cache tiles.
//! No BLAS, unsafe code, explicit SIMD, FMA or floating-point reassociation.
use crate::{
    backend::{Backend, CpuBackend},
    tensor::{self, Op, Tensor},
};
pub struct TiledCpuBackend;
const TILE: usize = 32;
const TILE_ROWS: usize = 16;
const TILE_COLUMNS: usize = 128;
const MIN_WORK: usize = 32 * 32 * 32;
fn large(m: usize, k: usize, n: usize) -> bool {
    m.saturating_mul(k).saturating_mul(n) >= MIN_WORK
}
fn gemm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    gemm_into(a, b, m, k, n, &mut c);
    c
}
fn gemm_into(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, c: &mut [f64]) {
    for ii in (0..m).step_by(TILE_ROWS) {
        for pp in (0..k).step_by(TILE) {
            for jj in (0..n).step_by(TILE_COLUMNS) {
                let end_j = (jj + TILE_COLUMNS).min(n);
                for i in ii..(ii + TILE_ROWS).min(m) {
                    let output = &mut c[i * n + jj..i * n + end_j];
                    for p in pp..(pp + TILE).min(k) {
                        let value = a[i * k + p];
                        let row = &b[p * n + jj..p * n + end_j];
                        for (dst, &rhs) in output.iter_mut().zip(row) {
                            *dst += value * rhs;
                        }
                    }
                }
            }
        }
    }
}
fn transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut output = vec![0.0; data.len()];
    for ii in (0..rows).step_by(TILE) {
        for jj in (0..cols).step_by(TILE) {
            for i in ii..(ii + TILE).min(rows) {
                for j in jj..(jj + TILE).min(cols) {
                    output[j * rows + i] = data[i * cols + j];
                }
            }
        }
    }
    output
}
impl Backend for TiledCpuBackend {
    fn forward(
        &self,
        op: Op,
        a: &Tensor,
        b: Option<&Tensor>,
        result_shape: &[usize],
    ) -> Result<Tensor, String> {
        if op == Op::BatchMatmul {
            let shape = tensor::output_shape(op, &a.shape, b.map(|t| t.shape.as_slice()))?;
            let b = b.unwrap();
            let rank = a.shape.len();
            let (batch, m, k, n) = (
                a.shape[..rank - 2].iter().product::<usize>(),
                a.shape[rank - 2],
                a.shape[rank - 1],
                *b.shape.last().unwrap(),
            );
            if !large(m, k, n) {
                return CpuBackend.forward(op, a, Some(b), result_shape);
            }
            let mut data = vec![0.0; tensor::elements(&shape)?];
            for index in 0..batch {
                let left = &a.data[index * m * k..(index + 1) * m * k];
                let offset = if b.shape.len() == 2 { 0 } else { index * k * n };
                gemm_into(
                    left,
                    &b.data[offset..offset + k * n],
                    m,
                    k,
                    n,
                    &mut data[index * m * n..(index + 1) * m * n],
                );
            }
            return Tensor::new(shape, data);
        }
        if op != Op::Matmul {
            return CpuBackend.forward(op, a, b, result_shape);
        }
        let shape = tensor::output_shape(op, &a.shape, b.map(|t| t.shape.as_slice()))?;
        let b = b.unwrap();
        let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
        if !large(m, k, n) {
            return CpuBackend.forward(op, a, Some(b), result_shape);
        }
        Tensor::new(shape, gemm(&a.data, &b.data, m, k, n))
    }
    fn vjp(
        &self,
        op: Op,
        a: &Tensor,
        b: Option<&Tensor>,
        out: &Tensor,
        g: &[f64],
    ) -> (Vec<f64>, Option<Vec<f64>>) {
        if op == Op::BatchMatmul {
            let b = b.unwrap();
            let rank = a.shape.len();
            let (batch, m, k, n) = (
                a.shape[..rank - 2].iter().product::<usize>(),
                a.shape[rank - 2],
                a.shape[rank - 1],
                *b.shape.last().unwrap(),
            );
            if !large(m, k, n) {
                return CpuBackend.vjp(op, a, Some(b), out, g);
            }
            let mut ga = vec![0.0; a.data.len()];
            let mut gb = vec![0.0; b.data.len()];
            for index in 0..batch {
                let left = &a.data[index * m * k..(index + 1) * m * k];
                let offset = if b.shape.len() == 2 { 0 } else { index * k * n };
                let right = &b.data[offset..offset + k * n];
                let upstream = &g[index * m * n..(index + 1) * m * n];
                {
                    let bt = transpose(right, k, n);
                    gemm_into(
                        upstream,
                        &bt,
                        m,
                        n,
                        k,
                        &mut ga[index * m * k..(index + 1) * m * k],
                    );
                }
                {
                    let at = transpose(left, m, k);
                    gemm_into(&at, upstream, k, m, n, &mut gb[offset..offset + k * n]);
                }
            }
            // Shared RHS accumulates directly across batches, preserving the
            // reference reduction order instead of summing rounded batch totals.
            return (ga, Some(gb));
        }
        if op != Op::Matmul {
            return CpuBackend.vjp(op, a, b, out, g);
        }
        let b = b.unwrap();
        let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
        if !large(m, k, n) {
            return CpuBackend.vjp(op, a, Some(b), out, g);
        }
        // Independent scope drops B^T before allocating A^T. These are scratch
        // copies, not retained tensors; process RSS is needed to measure them.
        let ga = {
            let bt = transpose(&b.data, k, n);
            gemm(g, &bt, m, n, k)
        };
        let gb = {
            let at = transpose(&a.data, m, k);
            gemm(&at, g, k, m, n)
        };
        (ga, Some(gb))
    }
}
