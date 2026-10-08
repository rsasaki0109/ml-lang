# Model checkpoints and inference

The CPU runtime can persist model weights and Adam optimizer state without a
Python dependency. This is model serialization, separate from activation
checkpointing/recomputation, which is not implemented.

## Train, infer, resume

```sh
checkpoint_dir=$(mktemp -d)
ml run examples/adam_multiclass.ml --save "$checkpoint_dir/model.mlckpt"
ml run examples/adam_infer.ml --load "$checkpoint_dir/model.mlckpt" --inference
ml run examples/adam_multiclass.ml --load "$checkpoint_dir/model.mlckpt" --save "$checkpoint_dir/resumed.mlckpt"
```

The first run trains a two-layer, eight-hidden-unit classifier for 300 Adam steps.
The second process uses the same function with batch size three, instead of six.
The third continues for another 300 steps. These fixed toy queries check numerical
behavior; they are not an independent dataset or an accuracy estimate.

`--load` requires the exact set of parameter names and matching declaration shapes.
Missing, extra or differently shaped parameters are errors. Initializers are still
compiled and evaluated, then replaced by saved values at parameter declaration.
Programs can use different input batches if their specialized shapes type-check.
Names are the model-state identity; renaming a parameter breaks compatibility.

A training load restores first/second moments, per-parameter counters and Adam
hyperparameters. With identical deterministic computations, 7 saved steps followed
by 13 resumed steps produce bit-identical weights and state to 20 uninterrupted
steps; both the Rust API and separate CLI processes test this guarantee.
Learning rate belongs to program source and may change. Betas/epsilon must match
already initialized Adam state. Parameters disconnected from a loss do not advance
Adam state. SGD loads weights and leaves saved Adam state available for later Adam.

Explicit `--inference` restores weights only and disables parameter gradient
tracking. It rejects `grad`, `backward`, `clip_grad_norm`, `step` and `adam_step` before execution,
even in zero-count loops. CLI `--save` is disallowed in this mode to avoid discarding
optimizer state accidentally. It is not a native executable export or an optimized
inference compiler. The Rust API exposes `Checkpoint` and `vm::execute_inference`.

## Storage and failure semantics

Files start with `MLCKPT01`. Integers and CPU f64 bits are little endian. Records
contain sorted parameter names, shapes and values, optional Adam moments/counters,
and optional shared Adam configuration. An FNV-1a 64-bit checksum detects accidental
corruption; it does not authenticate a model. No interoperability or forward-version
compatibility with external formats such as ONNX is promised.

Decoder and encoder validate names, duplicates, ranks, shape/value counts, finite
numbers, nonnegative second moments, positive counters and hyperparameters. Limits
are 4,096 parameters, 128 ASCII bytes per name, rank eight, 16,777,216 total f64
values including moments, and 129 MiB per file. Truncated files, trailing payload,
unsupported versions, invalid flags and checksummed but malformed metadata fail.
These checks are not a hardened sandbox or a global process resource quota.

Save happens after successful program execution, before buffered output is emitted.
A temporary file is created exclusively in the destination directory, written and
synced, then published with a hard link that fails if the destination exists.
Existing files are never replaced. The parent directory must exist and support
hard links; Linux x86-64 is validated. Temporary files created by a failed save are
removed where possible. Directory durability after sudden power loss is not
promised. Choose a new filename for every resumed checkpoint.

Only parameters and Adam state are serialized. Tape edges, gradients, local/global
`let` values, source/bytecode, prints, dataset positions and general program state
are excluded. Current random initialization is explicit and deterministic; no
stateful dropout/data-loader randomness is implemented. Exact continuation requires
the same operation order and compatible floating-point backend, not arbitrary
changed training code or cross-device bitwise reproducibility.

The minibatch example can use the same save/load workflow and inference source.
Exact resume must start at the same batch-order position: checkpointing after
complete epochs in the example preserves that order. Batch cursor, clipped
gradients and detached values are not serialized; clipping precedes each step.

`--inference --optimize` loads the same weights into typed SSA execution; it
requires the same names/shapes and produces the same forward values. Both modes
ignore Adam state after validating the checkpoint. See [inference IR](inference-ir.md).

Checkpoint storage is independent of `--backend cpu|tiled`. The selected backend
handles both loaded-model inference and training continuation; CPU-to-tiled
continuation is tested for exact weights/moments on the current toolchain.
