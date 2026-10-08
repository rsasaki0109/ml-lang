# Optional tiled CPU backend

The `tiled` backend adds fixed-tile matrix multiplication and its first-order
VJP to the dependency-free CPU f64 runtime. The reference `cpu` backend remains
the default. No BLAS library, unsafe Rust, explicit SIMD, threading or GPU
runtime is introduced. Other tensor primitives use the reference CPU kernels.

```sh
ml run examples/tiled_matmul.ml --backend tiled
ml run examples/minibatch.ml --backend tiled
ml run examples/tiled_matmul.ml --inference --optimize --backend tiled
ml bench examples/tiled_matmul.ml 5 --inference --backend tiled
```

`--backend cpu|tiled` is accepted by `run` and `bench`. It applies to reference-VM
training, reference inference and planned SSA inference. Unsupported names fail
explicitly. Parsing, shape checking and `plan` do not depend on a numeric backend.
The benchmark JSON records the backend as well as the execution mode.

## Kernel and gradient design

Rank-two `[m,k] @ [k,n]` uses tiles of 16 rows, 32 reduction elements and 128 columns, with clipped
ends for partial tiles. Transpose copies use 32×32 tiles. Row slices give the innermost loop contiguous input
and output access. Every output still accumulates contributions in increasing
reduction-index order. There is no arithmetic reassociation or explicit FMA.
The compiler may auto-vectorize the independent column updates.

When `m*k*n < 32,768`, the backend delegates forward and backward matmul to the
reference kernels. This is a fixed conservative fallback, not hardware-specific
autotuning or a claim that one threshold is optimal on every CPU.

The VJP computes `g @ transpose(b)` and `transpose(a) @ g` with the same GEMM
routine. Transposes use tiled copies, and the first scratch transpose is dropped
before allocating the second. These scratch copies can increase memory use.
They are excluded from the runtime's retained-tensor payload counter but included
in process RSS. The runtime retains its finite-gradient and atomic optimizer
update checks; clipping cannot repair an already non-finite backward result.

## Correctness and model state

Tests compare every forward and VJP element against the reference on square,
rectangular, skinny and partial-tile sizes, and independently check derivatives
across tile boundaries with finite differences. Tests also cover large-value
errors, primitive delegation, clipped Adam training, CPU-to-tiled checkpoint
continuation and loaded-model SSA inference. On the tested CPU/toolchain these
programs are exactly equal to reference execution. Arbitrary future backends,
compiler flags, architectures and accelerated libraries do not have a bitwise
compatibility guarantee.

Checkpoint files store weights and optimizer state, not a backend selection.
Choose a backend at run time. Initializers and shape/name validation have the
same behavior in both paths. Large training models still use dynamic-tape AD;
SSA execution remains forward-only, regardless of the selected backend.

## Measure and reproduce

```sh
cargo build --release --locked
/workspace/.ml-lang-bench-venv/bin/python benchmarks/kernels.py --samples 5 --output /tmp/ml-lang-kernels.json
```

Outside this prepared cloud environment, create a virtual environment and install
`benchmarks/requirements.txt`. Python, PyTorch and JAX are optional comparison
tools; they are not language dependencies. The harness currently requires Linux
`wait4`, `/proc` and CPU affinity.

Each CPU mode/framework/workload uses a fresh process and one inherited logical
CPU affinity. Framework math threads are set to one. One warmup precedes five
samples; import/JIT and ml-lang compile/lower costs are recorded separately from
warm execution. Every run initializes inputs and validates numerical results.
The harness measures complete programs rather than only the inner GEMM routine.

Forward workloads multiply square ones matrices ten times; every result element
must equal the dimension. Backward workloads evaluate mean-square matmul loss,
then gradients for both ones inputs; every gradient element must equal two.
Loss and gradient sums are checked as well. Sizes 128, 256 and 512 admit these
exact analytical values. Nonconstant values and irregular dimensions are covered
by the Rust numerical tests, not these timing workloads.

Validation/allocation bookkeeping differs between Rust, eager PyTorch and JAX
JIT. ml-lang initializer constants are compiled once and cloned during execution;
framework inputs are allocated per run. Native assertions create expected tensor
values, while frameworks compare to scalars. Returned variable/checkpoint clones
are included in Rust execution time. RSS includes imports, JIT, bytecode constants,
warmup and result snapshots. Thus this protocol cannot establish pure-kernel
speed, allocation efficiency or a production-framework advantage.

The measured table and exact source/executable/harness hashes are recorded below.
Do not extrapolate these shared-cloud, single-thread measurements to other CPUs,
GPUs, BLAS systems or representative model training. Larger-model integration,
f32 and optimized-library backends remain future work.

Current raw evidence: [tiled-cpu.json](../benchmarks/tiled-cpu.json).

| Workload | Size | Backend | Median (ms) | Peak RSS (MiB) |
| --- | ---: | --- | ---: | ---: |
| matmul + elementwise check, 10 times | 128 | ml-cpu | 5.748 | 2.93 |
| matmul + elementwise check, 10 times | 128 | ml-tiled | 5.564 | 2.93 |
| matmul + elementwise check, 10 times | 128 | torch-eager | 1.379 | 169.73 |
| matmul + elementwise check, 10 times | 128 | jax-jit | 3.029 | 126.32 |
| matmul MSE + both gradients | 128 | ml-cpu | 8.032 | 4.94 |
| matmul MSE + both gradients | 128 | ml-tiled | 2.427 | 4.86 |
| matmul MSE + both gradients | 128 | torch-eager | 0.749 | 170.88 |
| matmul MSE + both gradients | 128 | jax-jit | 0.585 | 127.56 |
| matmul + elementwise check, 10 times | 256 | ml-cpu | 40.229 | 7.00 |
| matmul + elementwise check, 10 times | 256 | ml-tiled | 39.990 | 7.05 |
| matmul + elementwise check, 10 times | 256 | torch-eager | 9.408 | 171.97 |
| matmul + elementwise check, 10 times | 256 | jax-jit | 9.116 | 130.15 |
| matmul MSE + both gradients | 256 | ml-cpu | 64.892 | 15.09 |
| matmul MSE + both gradients | 256 | ml-tiled | 16.333 | 15.09 |
| matmul MSE + both gradients | 256 | torch-eager | 3.267 | 174.25 |
| matmul MSE + both gradients | 256 | jax-jit | 3.137 | 131.95 |
| matmul + elementwise check, 10 times | 512 | ml-cpu | 293.244 | 23.55 |
| matmul + elementwise check, 10 times | 512 | ml-tiled | 329.617 | 23.56 |
| matmul + elementwise check, 10 times | 512 | torch-eager | 73.800 | 180.47 |
| matmul + elementwise check, 10 times | 512 | jax-jit | 67.759 | 134.45 |
| matmul MSE + both gradients | 512 | ml-cpu | 505.542 | 55.59 |
| matmul MSE + both gradients | 512 | ml-tiled | 107.692 | 55.59 |
| matmul MSE + both gradients | 512 | torch-eager | 23.175 | 187.25 |
| matmul MSE + both gradients | 512 | jax-jit | 19.689 | 139.44 |

The measured gain is strongest in backward workloads, where the reference VJP
combines two gradient accumulations in one loop and the tiled backend uses two
GEMMs. Forward-only gains are shape-dependent and may regress; choose based on
measured workloads. The tiled backend remains opt-in. PyTorch and JAX retain a
substantial advantage in these larger dense operations. No claim of parity with
optimized numerical libraries is made, and no GPU memory has been measured.

Batched products are supported as well: the tile/fallback policy applies per
batch, shared RHS gradients accumulate across batches, and other new primitives
use reference kernels. See [attention](attention.md) for shape and gradient rules.
