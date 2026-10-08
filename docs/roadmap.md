# Roadmap

## M0 — Independent executable CPU language (implemented)

- Rust lexer/parser, shape checks, user functions with local let declarations,
  and scoped fixed-count loops.
- Shape-annotated bytecode VM, dense f64 kernels, reverse-mode gradients, SGD and Adam.
- Differentiable rank-2 transpose, reshape, stable sigmoid and logits BCE.
- Last-axis softmax, stable multiclass cross entropy and a three-cluster classifier.
- Define, train and infer MSE and binary-cross-entropy XOR solely from `.ml` source.
- Gradient finite differences, compiler diagnostics, integration tests and CI.
- Symbolic function dimensions, detach, explicit inference, and model/Adam-state checkpoints.
- Batched products/shared-weight gradients, last-axis transpose and a trainable
  single-head attention reconstruction example, plus causal masking and its VJP.
- Tanh-approximate GELU and a trainable causal pre-norm Transformer block.
- Multiple batch axes, differentiable axis swaps and a two-head causal training example.
- Last-axis layer normalization with stable scaled statistics and affine training.
- Optional tiled CPU matmul/VJP kernels with reference fallback for small work.
- Static row minibatches and stable global gradient clipping with saved-state resume.
- CPU PyTorch/JAX comparison harness with wall time, correctness and peak RSS.

Acceptance: examples execute and assert correct outputs; all primitive gradients
and a composed network gradient match numerical checks. GPU and speed advantages
are outside this milestone.

## M1 — Compiler and language foundations (partially implemented)

- Typed forward-only CPU f64 SSA, region captures and last-use inference execution
  are implemented. Extend to training, generalized dtype/device and optimizer effects.
- Source spans through type checking/runtime errors; structured diagnostics.
- Modules, function control flow/effects, scalar types, conditionals, loop-carried
  state. Symbolically shaped function parameters are implemented through inlining.
- Explicit inference and detach are implemented; differentiate IR functions
  without a dynamic tape.
- Serialized compiler output and CPU native-code/BLAS backend experiments.

Acceptance: bytecode/reference and lowered execution agree numerically; runtime
failure modes and state ownership are specified; meaningful model examples use
modules and dynamic batch sizes without silent recompilation.

## M2 — GPU execution

- Evaluate MLIR/LLVM/IREE integration using an end-to-end matmul/activation/gradient.
- Device-resident buffers, allocation ownership, events, explicit transfers.
- CUDA and ROCm capability checks, driver/runtime compatibility, dtype support.
- f32 and mixed precision with explicit accumulation and loss-scaling policies.

Acceptance: training and inference run on available NVIDIA and AMD hardware;
unsupported targets fail explicitly. Compare correct synchronized computations
against PyTorch eager/compiled and JAX on each tested device.

## M3 — Memory and throughput optimization

- Inference last-use reclamation and contiguous-suffix axis swaps are implemented. Add joint forward/backward
  liveness, safe buffer reuse, fusion and kernel autotuning.
- Checkpoint/recompute planner with explicit compute/memory budgets.
- SGD momentum, dynamic/shuffled data batching and portable inference artifacts. Adam and model
  serialization are implemented in the reference runtime.
- Convolution, arbitrary masked/fused attention, class-index/weighted losses and
  representative models. Single-head and two-head causal attention are implemented.

Acceptance: record peak allocated/reserved device memory, wall time, compilation
cost and correctness against optimized baselines, across several shapes. Attribute
any advantage to ablations; do not extrapolate one tiny workload to production.

## Longer term

Distributed execution, sparse tensors, custom differentiable kernels,
shape constraints, editor/LSP tooling and reproducible package management.
No delivery dates or universal performance guarantees are promised.
