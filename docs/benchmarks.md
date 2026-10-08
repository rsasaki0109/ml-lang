# CPU benchmark results

Measured on 2026-10-08, Linux x86-64, Intel Xeon Platinum 8573C, Rust 1.85.1,
Python 3.12.14, PyTorch 2.6.0+cpu, JAX 0.4.38. All work uses `f64`, CPU,
one inherited logical-CPU affinity and one framework math thread. These are
shared-cloud measurements, with five samples after one warmup; they are not
statistically controlled claims about hardware or compiler superiority.

Current axis-swap kernel ablation: [axis-swap-cpu.json](../benchmarks/axis-swap-cpu.json),
with [method, results and limits](axis-swap.md). It compares the retained historical
permutation algorithm with contiguous-suffix copies, not ML frameworks.

Unmasked batched attention comparison (before causal softmax): [attention-cpu.json](../benchmarks/attention-cpu.json),
with [attention design and measured limits](attention.md).
Optional CPU kernel comparison before batched attention: [tiled-cpu.json](../benchmarks/tiled-cpu.json),
with [kernel design, results and tradeoffs](tiled-cpu.md).
Reference training comparison before the optional tiled backend:
[ir-baseline-cpu.json](../benchmarks/ir-baseline-cpu.json).
Inference VM/SSA comparison: [inference-cpu.json](../benchmarks/inference-cpu.json).
Earlier minibatch measurements remain in [minibatch-cpu.json](../benchmarks/minibatch-cpu.json).
Full-batch Adam measurements remain in [training-cpu.json](../benchmarks/training-cpu.json).
Function-local measurements remain in [functions-cpu.json](../benchmarks/functions-cpu.json).
The preceding multiclass measurements remain in [multiclass-cpu.json](../benchmarks/multiclass-cpu.json).
The previous binary-classification measurements remain in [classification-cpu.json](../benchmarks/classification-cpu.json).
The original prototype measurements remain in [baseline-cpu.json](../benchmarks/baseline-cpu.json).
The earlier framework reports predate causal softmax; they do not measure causal attention or the current executable. Source/executable hashes identify each measured implementation. The unmasked attention report
also records harness and individual ml-lang workload-source hashes. Training
examples include symbolic function shapes, Adam state and returned model snapshots;
losses, probabilities and final Adam-model weights match the framework baselines.
The report records source and executable SHA-256 hashes, versions, CPU settings,
validated outputs, minimum and median workload times, cold/compile times and RSS.

| Workload | Backend | Median time (ms) | Peak process RSS (MiB) |
| --- | --- | ---: | ---: |
| 64×64 matmul + sum, 10 times | ml-lang | 0.813 | 1.81 |
| 64×64 matmul + sum, 10 times | torch-eager | 0.138 | 168.88 |
| 64×64 matmul + sum, 10 times | jax-jit | 0.823 | 121.80 |
| 128×128 matmul + sum, 10 times | ml-lang | 7.644 | 2.67 |
| 128×128 matmul + sum, 10 times | torch-eager | 0.609 | 169.43 |
| 128×128 matmul + sum, 10 times | jax-jit | 3.874 | 122.88 |
| 256×256 matmul + sum, 10 times | ml-lang | 48.869 | 6.07 |
| 256×256 matmul + sum, 10 times | torch-eager | 5.505 | 171.14 |
| 256×256 matmul + sum, 10 times | jax-jit | 16.212 | 127.00 |
| XOR (MSE), 2,000 steps + inference | ml-lang | 16.295 | 1.68 |
| XOR (MSE), 2,000 steps + inference | torch-eager | 384.253 | 170.56 |
| XOR (MSE), 2,000 steps + inference | jax-jit | 3.249 | 144.98 |
| Binary XOR (BCE), 2,000 steps + inference | ml-lang | 13.025 | 1.70 |
| Binary XOR (BCE), 2,000 steps + inference | torch-eager | 353.778 | 170.63 |
| Binary XOR (BCE), 2,000 steps + inference | jax-jit | 3.678 | 148.74 |
| 3-class CE, 1,000 SGD steps + inference | ml-lang | 8.751 | 1.70 |
| 3-class CE, 1,000 SGD steps + inference | torch-eager | 201.674 | 170.81 |
| 3-class CE, 1,000 SGD steps + inference | jax-jit | 3.643 | 154.86 |
| 3-class CE, 300 Adam steps + inference | ml-lang | 2.858 | 1.71 |
| 3-class CE, 300 Adam steps + inference | torch-eager | 133.444 | 227.84 |
| 3-class CE, 300 Adam steps + inference | jax-jit | 3.000 | 188.16 |
| 3-class CE, 300 clipped minibatch Adam steps + inference | ml-lang | 2.447 | 1.73 |
| 3-class CE, 300 clipped minibatch Adam steps + inference | torch-eager | 157.949 | 227.93 |
| 3-class CE, 300 clipped minibatch Adam steps + inference | jax-jit | 2.930 | 189.54 |

## Interpretation

The reference matrix kernels lag PyTorch on all tested sizes and JAX at the
larger sizes. Tiny training-loop rankings vary by workload; PyTorch uses eager
execution, while JAX compiles the training loop. These differences, bookkeeping
and five samples on a shared machine cannot establish a general speed advantage.
PyTorch compiled training has not been benchmarked. No fusion, BLAS or GPU
backend is implemented.

Process RSS is lower for the standalone executable. Framework RSS includes
Python imports, native libraries and JIT compilation, so it mainly measures
total process footprint for these small tasks. It does not isolate tensor
allocation efficiency, parameter memory or GPU memory. No GPU measurements exist.

## Protocol

- Each backend/workload runs in a fresh process. Matmul allocates two ones matrices
  and executes 10 independent matrix products with scalar sums. All ten sums
  are validated as `n³`. JAX uses a JIT matmul/reduction kernel and synchronizes
  each result. PyTorch runs eager matmul and reduction.
- Both XOR tasks use identical SplitMix64 initialization, input/target arrays, `tanh`
  activations, mean squared loss, 2,000 steps and SGD rate `0.1`. Initial/final
  loss and all four predictions are compared to ml-lang at `rtol=1e-9, atol=1e-10`.
  Binary XOR trains raw logits with mean BCE and applies sigmoid only for inference.
  PyTorch uses `binary_cross_entropy_with_logits`; JAX uses stable `softplus`.
- Multiclass training uses six two-feature samples, three one-hot classes, an
  8-hidden-unit tanh layer and 1,000 SGD steps at rate `0.1`. All 18 probabilities
  and both losses match the reference. PyTorch uses probability-target cross
  entropy; JAX uses `log_softmax`. This fits training samples only and does not
  measure generalization.
- Adam uses the same six samples, model and initialization, 300 steps at rate
  `0.03`, betas `(0.9,0.999)` and epsilon `1e-8`. PyTorch uses its native Adam
  optimizer with foreach/fused implementations disabled; JAX implements the
  bias-corrected update in a compiled loop. Both losses, all 27 training/query
  probabilities and all 51 final weights match at `rtol=1e-9, atol=1e-10`.
  Three fixed toy query points are not a generalization benchmark. Checkpoint
  file I/O is excluded; ml-lang execution returns a cloned model/state snapshot
  and buffered final-weight diagnostics. Frameworks return flattened weights.
- Minibatch training visits rows `[0:2]`, `[2:4]`, `[4:6]` in that order for 100
  epochs, with one update per batch (300 total). The model, Adam settings and
  initialization match the full-batch Adam task. Each backward pass uses global
  L2 clipping at `1.0` before Adam. Baselines use explicit epsilon-free clipping
  expressions to match the language policy; PyTorch's `clip_grad_norm_` default
  adds `1e-6` to the norm and is not used. JAX selects the two-row batch within
  its compiled loop; ml-lang statically emits three updates in each epoch.
  Both losses, all 27 probabilities and all 51 weights match the same tolerance.
  Batch order is fixed; there is no shuffling or external data loading. RSS and
  timings cover each complete process/workload, not isolated batch allocations.
- JAX compiles its training loop using `lax.fori_loop`; PyTorch uses an eager
  Python loop. All measured training runs reset parameters. GPU/asynchronous
  results are not timed without synchronization; JAX calls `block_until_ready()`.
- Median execution times exclude imports and the warmup/JIT pass. They include
  initialization, workload execution and result validation. ml-lang compiles
  source once before warmup and reports that compilation time separately.
  Its matrix initializer constants are allocated during compilation and cloned
  by the VM; frameworks allocate inputs during execution. The language example
  includes data setup and initial/final prediction; Python baselines cache the
  tiny input/target arrays. These small bookkeeping differences are limitations.
- ml-lang buffers printed values internally; timing does not include terminal
  printing. Baselines likewise return result lists instead of printing during timing.
- Peak RSS is each running executable/interpreter's Linux `VmHWM` from
  `/proc/self/status` after all runs. It includes code, libraries, compiler/JIT
  state and heap. A separate `launch_peak_rss_bytes` from `wait4` includes launcher
  inheritance before exec; it is not used in the table. This avoids imposing
  the Python controller's memory floor on a small native executable.
- `peak_retained_tensor_bytes` is an ml-lang instrumentation lower bound for
  retained tensor/gradient payloads and persistent Adam moments; it excludes
  optimizer scratch vectors, returned checkpoint snapshots, copies,
  metadata, capacities and result-map cloning. It is not comparable to RSS.
- The platform is shared and timings varied between runs. No statistical
  significance, throughput scaling, GPU speedup or production advantage is claimed.

## Inference last-use comparison

The separate inference harness compares reference-VM inference with optional SSA
execution on the same CPU kernels. It validates bit-identical printed results,
then uses one warmup and five timed executions per mode in isolated processes
with one inherited CPU affinity. Source, executable, both harnesses and workload
hashes identify this run. Compile/lower times are recorded separately.

| Workload | Mode | Median (ms) | Peak RSS (MiB) | Retained payload lower bound (MiB) |
| --- | --- | ---: | ---: | ---: |
| activation-chain-256 | inference-vm | 7.117 | 7.07 | 4.00 |
| activation-chain-256 | inference-ssa | 5.294 | 4.50 | 1.50 |
| activation-chain-512 | inference-vm | 29.312 | 23.57 | 16.00 |
| activation-chain-512 | inference-ssa | 22.494 | 13.57 | 6.00 |
| activation-chain-1024 | inference-vm | 124.534 | 89.55 | 64.00 |
| activation-chain-1024 | inference-ssa | 100.794 | 49.57 | 24.00 |
| dense-network | inference-vm | 0.293 | 2.11 | 0.32 |
| dense-network | inference-ssa | 0.284 | 1.84 | 0.19 |

The six-activation chains deliberately isolate temporary lifetime reclamation;
they are synthetic workloads, not trained production models. At 1024×1024,
peak process RSS fell from 89.55 to 49.57 MiB in this run, about 45%. The initialized
dense network is a 64-sample, 32-input, 128-hidden, 16-output forward computation;
no trained-model accuracy is measured. These measurements show the effect of
last-use release for these programs, not superiority over PyTorch/JAX or GPUs.
No buffer reuse, fusion or arithmetic reassociation occurs.

RSS includes bytecode/IR constants, lowering, warmup and returned model/result
snapshots, whereas the retained-payload counter excludes those compilation and
return copies and kernel scratch vectors. All top-level bindings remain roots;
intermediates explicitly bound there may prevent the same memory reduction.
Printed outputs remain buffered. See [IR semantics](inference-ir.md).

```sh
cargo build --release --locked
python3 benchmarks/inference.py --samples 5 --output /tmp/ml-lang-inference.json
```

## Reproduce

```sh
cargo build --release --locked
python3 -m venv .venv
.venv/bin/python -m pip install -r benchmarks/requirements.txt
.venv/bin/python benchmarks/compare.py --samples 5 --output /tmp/ml-lang-results.json
```

The core language does not depend on these Python packages. The comparison
script currently requires Linux `wait4`, `/proc` and CPU affinity. For the cloud
environment prepared in this task, the installed optional environment is
`/workspace/.ml-lang-bench-venv`; activate it or call its Python directly.

Next measurements should include optimized PyTorch, training SSA/native ml-lang,
larger nonconstant workloads, representative models, compile-latency breakdowns,
multiple samples/machines and real CUDA/ROCm memory counters.
