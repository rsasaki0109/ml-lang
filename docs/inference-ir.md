# Typed inference SSA and last-use execution

The compiler can lower its checked stack bytecode into a typed, forward-only SSA
representation and execute it with a last-use release plan. This is an optional
CPU f64 interpreter, not native code generation, GPU execution, buffer reuse or
fusion. Training still uses the reference VM and dynamic reverse-mode tape.

```sh
ml plan examples/inference_memory.ml
ml run examples/inference_memory.ml --inference --optimize
ml bench examples/inference_memory.ml 5 --inference
ml bench examples/inference_memory.ml 5 --inference --optimize
```

Saved models work with the same path:

```sh
checkpoint_dir=$(mktemp -d)
ml run examples/minibatch.ml --save "$checkpoint_dir/model.mlckpt"
ml run examples/adam_infer.ml --load "$checkpoint_dir/model.mlckpt" --inference --optimize
```

`--optimize` requires `--inference`, and inference cannot save optimizer state.
`plan` always lowers for inference and rejects training effects. Both execution
paths reject `grad`, `backward`, clipping and optimizer steps, including inside
zero-count loops. `bench` accepts inference flags and reports `execution_mode`
and `lower_seconds`; parsing/type checking time remains `compile_seconds`.
Lowering occurs once before warmup, not once per timed execution. Checkpoint
loading is supported by `run`, not by `bench`.

## Representation and validation

Each computed tensor has a unique `%id` and a concrete shape. All value types
are currently `Tensor<f64,cpu>`. Bytecode loads/stores become value references;
ordinary variable aliases do not produce new definitions. Parameter initialization
is an explicit operation, with a name, initializer and independently restored
value. A parameter initializer is evaluated even when a checkpoint replaces it.

Constants, seeded random initialization and primitive operations produce values.
Printing and assertions are ordered effects. Fixed-count loops contain regions
with explicit immutable captures. Region-local definitions represent a new
instance on each iteration; they are not unrolled, and no loop-carried values or
phi nodes are implemented. Nested regions preserve values they capture until
that region finishes. This matches the language's current no-reassignment,
non-escaping loop-local semantics.

Lowering checks stack balance, initialized/single-assignment slots, parameter
identity, operation result types, reshape size, assertion types and final roots.
IR fields are private to the crate and construction goes through `ir::lower`.
There is no serialized IR format or arbitrary artifact-loading interface.
`InferenceProgram::display()` prints deterministic types, captures and releases.
Public Rust APIs are `ir::lower`, `inference::execute` and
`inference::execute_with_backend`.

## Release plan and observable behavior

For each region, the planner counts operand uses, including repeated operands,
and treats a nested region's captures as uses by that loop. After the final use,
values defined in that region are released. Unused results are released after
evaluation. Captured values belong to their defining outer region, so repeated
body executions cannot release them prematurely. Region-local payload is freed
each iteration.

All top-level bindings and parameters remain roots until returned in the Rust
`Execution` result. Thus a top-level `let hidden = ...` is intentionally retained,
while hidden values inside functions or nested expressions can be reclaimed.
This preserves the reference API's complete final variable map. The planner does
not discard observable bindings simply because the CLI prints fewer values.

Every operation still executes in source order. Unused arithmetic can raise an
error; removing it would change semantics, so this is not dead-code elimination.
Buffered prints are emitted only after successful execution. Printed tensors own
snapshots and remain live; printing large values in a loop still consumes memory.
`detach` can become a value alias in this IR because all values are immutable and
optimizer effects are forbidden; its observable snapshot behavior is preserved.

The runtime drops owned Tensor vectors at last use. It does not reuse allocation
slots for unrelated buffers, fuse kernels, change arithmetic order, or remove
copies inside numerical primitives. Future IR autodiff and mutation will require
new ownership/effect rules before these transformations can be used in training.

## Validation and measurement

Tests compare all primitive operations, composed functions, nested/zero loops,
shared operands, aliases, outputs, complete variable maps and loaded checkpoints
against explicit reference-VM inference. Checkpoint weights match exactly;
optimizer moments are ignored in both inference modes. Tests retain errors from
unused operations, reject training effects and malformed bytecode, and confirm
bounded no-print loop payload and reduced temporary activation payload.

Run the standalone comparison harness:

```sh
cargo build --release --locked
python3 benchmarks/inference.py --samples 5 --output /tmp/ml-lang-inference.json
```

It validates bit-identical printed results for three synthetic six-activation
chains and an initialized dense network. Each mode runs in a fresh process with
one inherited CPU affinity. RSS includes source/bytecode, lowering/IR constants,
warmup, execution and result snapshots. `peak_retained_tensor_bytes` is only a
lower bound for runtime tensor payload; it excludes IR constants, scratch
vectors, metadata, capacity and returned result/checkpoint copies. See
[benchmark results](benchmarks.md). Synthetic activation chains isolate lifetime
reclamation and do not establish an advantage over production frameworks or GPUs.
