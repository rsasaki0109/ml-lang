# Prior art and design decisions

Research date: 2026-10-08. This is an initial architecture survey, not an exhaustive
performance evaluation. Direct documentation-site access was restricted in the
cloud environment. Official repository sources were read for JAX, Julia, MLIR,
Mojo and IREE; PyTorch compiler docs and source were inspected in the installed
2.6.0 CPU wheel. The linked upstream main branches can change.

| System | Relevant strengths | Tradeoff for this project | Decision |
| --- | --- | --- | --- |
| Python + PyTorch | Broad operator coverage, eager debugging, autograd, Dynamo/AOTAutograd/Inductor compilation | Python program capture and graph-break handling add a frontend boundary; eager shapes are normally checked during execution | Own source frontend and explicit static shape contracts; retain an eager reference executor for correctness |
| JAX / XLA | Composable transforms, functional array model, JIT fusion | Compiled array shapes generally need to be static; value-dependent output shapes need another strategy; compilation and asynchronous dispatch complicate timing | Explicit shapes first, IR-visible differentiation later; warm up and synchronize benchmarks |
| Mojo | Independent language combining Python-like syntax with systems programming and metaprogramming; CPU/GPU ambitions | Low-level kernel control is useful but does not by itself specify ml-lang training, shape or autodiff semantics | Define language-level tensor and learning semantics; consider kernel/codegen lessons without adopting its frontend |
| Julia | Compiler specialization, expressive numerical code and established libraries | Type stability, globals, allocations and first-call compilation affect performance | Avoid untyped globals; separate cold vs warm measurement and measure memory |
| MLIR Linalg | Structured ops, tiling, fusion, bufferization, LLVM/library lowering | Significant integration/build complexity; transformation legality and ABI must be maintained | Prefer evaluating established codegen over building GPU assemblers; defer integration until SSA/effects exist |
| IREE | MLIR-based end-to-end compiler/runtime and deployment IR | Frontend/export contract, stateful training and target support require explicit investigation | Candidate hardware execution backend, not a substitute for the independent frontend |

These are tradeoffs, not blanket shortcomings or evidence that ml-lang is faster.
PyTorch compilation and JAX already implement sophisticated optimizations. The
first differentiator implemented here is the **language contract**: a malformed
matmul or function tensor shape can be rejected before execution, without
running Python to discover the computation.

## Testable hypotheses

1. An explicit tensor/effect IR may allow optimization of an entire training step
   without Python graph breaks. Compare against compiled PyTorch and JAX at equal
   dtype, device and synchronization, including compile latency.
2. Joint forward/backward liveness and checkpoint scheduling may reduce GPU peak
   allocated memory at a known recomputation cost. Test on real GPU hardware;
   host RSS or a retained-payload counter cannot establish this.
3. Shape/device checking may shorten debugging cycles. Collect diagnostic tests
   and usability evidence; do not confuse static restrictions with generality.
4. An independent syntax may make model definitions concise while retaining
   custom operations. Expand functions, modules and control flow before judging
   language expressiveness.

## Sources consulted

- [PyTorch compiler API source (v2.6.0)](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/compiler/__init__.py)
  and its Dynamo/AOTAutograd/Inductor explanations; inspected from the official CPU wheel.
- [JAX: Thinking in JAX](https://github.com/jax-ml/jax/blob/main/docs/notebooks/thinking_in_jax.md):
  static shapes, JIT, transforms, and `block_until_ready()`.
- [Julia performance tips](https://github.com/JuliaLang/julia/blob/master/doc/src/manual/performance-tips.md):
  type stability, allocation measurement, BLAS threads and first-call latency.
- [Mojo language overview](https://github.com/modular/modular/blob/main/Mojo/README.md)
  and [Modular overview](https://github.com/modular/modular/blob/main/README.md).
  Only the overview was inspected; detailed ownership/AD guarantees remain to be researched.
- [MLIR Linalg design](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Dialects/Linalg/_index.md):
  buffer allocation, tiling, fusion and structured operations.
- [IREE overview](https://github.com/iree-org/iree/blob/main/README.md):
  MLIR-based execution, project scope and licensing.
