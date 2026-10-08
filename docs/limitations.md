# Known limitations

- CPU f64 only; no GPU implementation or GPU memory measurements.
- Reference and optional tiled CPU dense kernels; no BLAS, fusion, native/JIT codegen, buffer reuse,
  activation checkpoint/recomputation, sparse tensors or mixed precision.
- `@` is rank-2; `bmm` accepts rank >= 3 with shared rank-2 or matching-rank RHS.
  Batch prefixes must match exactly; no partial batch broadcasting or convolution.
  Multihead layout uses copying `reshape`/`swap_axes`, not zero-copy strided views.
  `transpose_last` swaps final axes for rank >= 2; row extraction is contiguous only.
  No general indexing. Attention supports a square inclusive causal mask via `causal_softmax`;
  arbitrary/padding masks, rectangular cached decoding, dropout, KV caches and fusion are absent.
  GELU is the tanh approximation, with saturated finite tails, not exact erf GELU.
  The toy Transformer has no tokenizer, embedding lookup, text dataset or generation.
  Layer normalization uses only the last axis with fixed epsilon `1e-5`;
  arbitrary normalized axes and configurable epsilon are absent.
  Softmax uses the last axis; cross entropy requires rank-2 probability targets,
  with no class-index labels or weighted/ignored classes.
- Compile-time specialized symbolic function shapes, restricted functions, fixed repeat counts; no general-purpose
  scalar/control-flow/module ecosystem yet.
- Dynamic reverse-mode tape; no higher-order differentiation or gradients across
  steps/loop iterations. SGD and Adam only; no dataset I/O, shuffled/dynamic minibatches, weight decay or schedulers.
- Model checkpoints are a private versioned CPU f64 format, not ONNX or a portable
  executable. No runtime dynamic shapes or dimension arithmetic. Training resume
  saves optimizer state, not data-loader positions or arbitrary program state.
- Frontend diagnostic locations exist for syntax errors. Most type/runtime errors
  are messages without source spans.
- `print` is buffered. Programs printing inside large loops retain those outputs.
  Reference-VM straight-line programs can retain a tape until a step or loop boundary.
  Optional inference SSA releases temporary values at last use, but keeps all final
  top-level bindings and buffered outputs. SSA training/autodiff is not implemented.
- Per-tensor size/nesting guards are not a total resource limit or security boundary.
  Very large ASTs, inlining, loops or outputs can exhaust resources.
- The payload counter is not a heap profiler. Actual comparisons use process RSS;
  host RSS does not measure GPU memory or allocator efficiency.
- Validation is Linux x86-64 CPU only. CI is configured locally; a GitHub-hosted
  CI run has not been observed until the repository is pushed.
- Published packages and releases do not yet exist. Benchmarks are small initial
  measurements, not evidence of superiority over production ML systems.
