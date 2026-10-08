//! Kernel dispatch boundary. The v0.1 ABI uses host-resident dense tensors.
//! GPU-native storage, asynchronous events and memory planning are future work.
use crate::tensor::{self, Op, Tensor};
pub trait Backend {
    fn forward(
        &self,
        op: Op,
        a: &Tensor,
        b: Option<&Tensor>,
        result_shape: &[usize],
    ) -> Result<Tensor, String>;
    fn vjp(
        &self,
        op: Op,
        a: &Tensor,
        b: Option<&Tensor>,
        out: &Tensor,
        g: &[f64],
    ) -> (Vec<f64>, Option<Vec<f64>>);
}
pub struct CpuBackend;
impl Backend for CpuBackend {
    fn forward(
        &self,
        op: Op,
        a: &Tensor,
        b: Option<&Tensor>,
        result_shape: &[usize],
    ) -> Result<Tensor, String> {
        if op == Op::Reshape {
            tensor::reshape(a, result_shape)
        } else {
            tensor::forward(op, a, b)
        }
    }
    fn vjp(
        &self,
        op: Op,
        a: &Tensor,
        b: Option<&Tensor>,
        out: &Tensor,
        g: &[f64],
    ) -> (Vec<f64>, Option<Vec<f64>>) {
        tensor::vjp(op, a, b, out, g)
    }
}
