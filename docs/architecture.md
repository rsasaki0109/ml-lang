# Architecture

## Implemented pipeline

```text
.ml source
  -> lexer (line/column diagnostics)
  -> recursive-descent / Pratt parser
  -> AST (functions, scoped bindings, loops, expressions)
  -> structural symbolic shape checker
  -> concrete shape specialization and function inliner
  -> shape-annotated stack bytecode
  -> reference VM (variable slots, stack, loop scopes, optimizer)
     or inference SSA -> last-use plan -> forward-only interpreter
  -> Backend kernel dispatch
  -> CPU dense f64 tensor kernels + reverse-mode tape
```

| Module | Responsibility |
| --- | --- |
| `syntax.rs` | Tokens, numeric and rectangular tensor literals, precedence, functions, loops |
| `typecheck.rs` | Symbolic dimension unification, shape proofs, call-graph validation |
| `optimizer.rs` | Atomic Adam updates and persistent per-parameter moments |
| `checkpoint.rs` | Versioned, bounded model/state encoding and no-overwrite persistence |
| `compiler.rs` | Lexical scopes, shape checking, function signatures, instruction emission |
| `tensor.rs` | Dense row-major storage, broadcasting, forward kernels and VJPs |
| `backend.rs` | Forward/VJP kernel interface; reference `CpuBackend` |
| `tiled.rs` | Optional 16×32×128 CPU matmul tiles, transpose-based VJPs and small-work fallback |
| `ir.rs` | Typed forward SSA, immutable region captures, validation and last-use planning |
| `inference.rs` | Planned forward execution and checkpoint weights |
| `vm.rs` | Tape identities, gradients, bounded loop lifetime, parameter updates |
| `main.rs` | `check`, `run`, `ir`, `plan`, `bench`, nonzero error exit codes |

Compilation happens before tensor execution. Functions are checked even when
unused. At call sites they are inlined with distinct argument and local slots, without
access to global bindings. Function-local `let` declarations are checked and emitted
in order before the return expression; nested calls restore the caller scope. An `Apply` instruction stores its expected output
shape, which the VM also verifies. Bytecode is an internal Rust representation,
not a serialized artifact or native executable. `ml ir` exposes it for debugging.
The prototype has one tensor type family, with shape as a compile-time parameter;
all scalars are rank-zero tensors. `f64` and `cpu` are the only valid dtype/device
parameters. This is a real but deliberately restricted type system.

## Autodiff and ownership

Each VM tensor has an identity in a topologically ordered tape. Only parameter
leaves and their descendants carry gradient edges. Reverse traversal applies
vector-Jacobian products and sums contributions from multiple consumers,
including repeated operands and broadcasts. Broadcast VJPs reduce contributions
into the original operand dimensions. Matmul has separate derivatives for each
operand. ReLU uses a zero subgradient at zero. Transpose and reshape preserve tape edges;
reshape takes its target shape from the typed instruction through backend dispatch.
Row extraction carries constant start/count in its operation and scatters the
VJP into the original shape. Overlapping slices accumulate on shared inputs.
Detach emits a dedicated bytecode instruction that copies values without tape edges.
Sigmoid and logits BCE use stable numerical formulas, with analytic VJPs tested
at moderate values against finite differences and at extreme logits for finiteness.
Softmax normalizes the last axis, with a rowwise vector-Jacobian product.
Multiclass loss keeps shifted log probabilities to preserve normalization even
with huge common offsets, and averages over batch rows. Probability-target
validation precedes evaluation; target-direction checks respect the simplex.

SGD and Adam validate every proposed update before mutating parameter storage. A step
then compacts the tape, preserving slot aliases and parameter identity but
removing edges and gradients. A loop boundary clears local slots and performs
the same compaction. Parameters remain differentiable leaves; other surviving
values become detached constants. Tests compare 3 vs 3,000 training steps and
2 vs 2,000 inference iterations. Peak retained tensor payload remains constant
for these no-output loops. This is **lifetime reclamation**, not buffer reuse,
operator fusion, activation recomputation, or a GPU memory optimizer. `print` buffers results.

The VM's payload counter excludes temporary kernel/VJP vectors, compaction
copies, optimizer update scratch vectors, capacity, metadata, source/bytecode storage
and returned model/result clones. Persistent Adam moments are counted. Process
RSS measurements are used for actual peak memory comparisons.

## Backend decision

The reference CPU implementation uses dependency-free Rust loops to make
correctness, differentiation and reproducibility easy to inspect. It does not
use BLAS or explicit SIMD kernels. An optional tiled backend improves cache
access for matmul and expresses VJPs as two tiled products with transpose scratch
copies. Benchmark it separately from the reference; optimized BLAS remains an
external-library integration candidate. See [kernel design and measurements](tiled-cpu.md).

Keep frontend semantics independent of hardware. The current `Backend` trait
has host-resident tensor inputs and outputs and an explicit expected result shape; it establishes a dispatch boundary,
but it is insufficient for efficient device-resident execution. Do not implement
GPU support by copying every operation through host vectors.

An initial CPU f64 inference SSA now lowers checked bytecode, with concrete
shapes, region captures and last-use release planning. Training remains in the VM.
The next compiler IR stage should extend this foundation with generalized shape/dtype/device types,
parameter-state effects, scalar loops, reduction structure and buffer lifetimes.
Perform autodiff at this level before lowering. This makes forward and backward
operations visible to fusion, liveness and checkpoint planning together. For
initial acceleration, evaluate MLIR `linalg` + LLVM for CPU, and MLIR/IREE for
CUDA and ROCm targets. Neither dependency is linked in v0.1. Compare licensing,
build size, Rust/C ABI stability, training-state mutation and device availability
before selecting the integration. IREE is an execution/codegen candidate, not
an imported language frontend or a framework dependency.

## Planned memory semantics

Use distinct immutable tensor values and mutable optimizer-owned parameter
buffers. Model device transfers explicitly. Buffer reuse must respect aliasing,
backward saves and asynchronous events. Track shape, dtype, device, layout,
last-use and recomputation cost in IR. Checkpoint planning must state its
compute/memory tradeoff and validate gradients. CUDA/ROCm allocators should expose
peak allocated/reserved bytes separately from host RSS.

Native compilation, GPU support, fusion, buffer reuse and activation
checkpoint/recomputation remain future work. Last-use reclamation is implemented
for optional inference SSA; see [its exact semantics](inference-ir.md).


## Model state and inference

Adam state is keyed by stable parameter slots rather than tape node identities.
Checkpoints map slots to names, keeping weights, moments, hyperparameters and
per-parameter counters. Restoration requires exact names and declaration shapes;
initializers are still evaluated. Saving publishes a complete checksummed file
without replacing an existing destination. This is model persistence, distinct
from future activation checkpoint/recompute planning.

Explicit inference rejects gradient/optimizer effects before executing any code,
including nested loops, and marks all parameter values as untracked. Adam state
is not restored in this mode. Reference-VM inference removes gradient edges;
`--optimize` additionally lowers typed SSA and releases temporaries at last use.
Buffer reuse, BLAS and GPU numerical backends remain future work; optional CPU
matmul tiling is available independently of last-use planning. See [checkpoint semantics](checkpoints.md).


## Gradient control and minibatch updates

Global clipping reads connected parameter-leaf gradients in stable slot order.
It scales the norm by the largest component before summing squares, avoiding
intermediate overflow, and applies clipping with a two-stage normalized transform
to preserve representable results even when a combined scale would underflow.
Both optimizers consume the resulting leaf gradients;
clipping does not mutate moments or advance counters. Intermediate tape gradients
remain unchanged. Inference effect validation rejects clipping.

Row extraction enables fixed compile-time minibatches on any leading axis, with
bounds checked structurally in functions and concretely in source. This is a dense
copy primitive with a scatter VJP, not view/alias semantics or dynamic data loading.
The clipped minibatch example and saved-state continuation tests exercise these
operations together. Long-loop payload tests establish bounded retained storage
for that program; they do not prove a memory advantage over framework batching.

CLI backend selection applies consistently to the reference VM and SSA executor,
including checkpoint loading and training continuation. `vm::execute_with_options`
exposes backend, optional checkpoint and inference effects together. SSA execution
uses `inference::execute_with_backend`; optimizer/tape semantics stay in the VM.


## Batched attention primitives

`BatchMatmul` carries rank >= 3 inputs with either a shared rank-two RHS or an
exactly matching rank/prefix RHS. Its VJP reduces shared RHS contributions across
batches. Tiled execution calls GEMM on disjoint output slices and accumulates
shared gradients directly into one buffer, preserving reference reduction order.
`TransposeLast` permutes final axes while retaining leading batch dimensions.
Both operations have concrete and structural symbolic shape rules and typed SSA
lowering. Ordinary functions compose them with softmax into attention;
there is no attention-specific frontend or framework dependency.

`SwapAxes` stores two constant axes in the typed operation. Shape checking swaps
those dimensions, forward copies row-major values, and reverse applies the inverse
permutation. Multiple batch axes in `BatchMatmul` flatten into a deterministic
batch traversal on both CPU backends. They enable multihead attention without
adding model-specific syntax or implicit batch broadcasting.

Axis swaps precompute input strides once and copy the contiguous suffix after
the largest swapped axis in blocks. The VJP reuses this kernel with the inverse
layout. Equal axes use one contiguous copy. This removes per-element metadata
allocations while preserving owned row-major output buffers; it is not a view
or buffer-reuse optimization.
