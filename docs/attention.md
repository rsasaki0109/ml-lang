# Batched tensors and single-head attention

The CPU language supports differentiable batched matrix multiplication and
last-two-axis transposition. Together with ordinary functions and last-axis
softmax, these operations express unmasked scaled dot-product attention without
an external ML framework or a special model DSL.

```ml
fn attend(x: Tensor[B,T,D], wq: Tensor[D,H], wk: Tensor[D,H],
          wv: Tensor[D,V], scale: Tensor[]) -> Tensor[B,T,V] {
    let query = bmm(x,wq);
    let key = bmm(x,wk);
    let value = bmm(x,wv);
    let weights = softmax(bmm(query,transpose_last(key))*scale);
    return bmm(weights,value);
}
```

`B` counts sequences, `T` tokens, `D` input features, `H` query/key features and
`V` output features. The scale is an explicit scalar; for head width four use
`0.5 = 1/sqrt(4)`. Dimension names cannot appear in ordinary scalar arithmetic.
Projection matrices are shared across every sequence and token.

## Shape and gradient rules

`bmm(a,b)` accepts `[...,M,K]` with either shared `[K,N]` or matching `[...,K,N]`
weights, returning `[...,M,N]`. Rank, inner dimensions and batch equality are
checked statically. The right batch dimension cannot broadcast from one, and
arbitrary NumPy-style batch broadcasting is unsupported. Multiple batch axes
now support rank-four heads (see [multihead attention](multihead-attention.md)). Rank-two
`@` remains unchanged. Symbolic functions must prove matching dimensions; unrelated
batch names are not assumed equal. Concrete specializations obey tensor-size limits.

The VJP multiplies upstream gradients by the other input's transpose. Shared
right-hand weights sum their gradients across all batches, not only the last
sequence. Reference accumulation visits batches then rows in order. The tiled
backend accumulates directly into the shared gradient buffer in that same order,
avoiding grouping rounded per-batch totals. Its small-work fallback is based on
one batch's matrix size; larger batches use independent tiled products.

`transpose_last(x)` swaps only the final two axes of any rank-two-or-higher tensor.
All leading dimensions are preserved. It copies dense row-major storage, and its
VJP swaps the same axes back. `transpose(x)` retains its original rank-two-only
contract. Batched matmul and transpose-last participate in reverse-mode AD, shape
specialization, backend dispatch and optional forward SSA planning.

## Train, save, infer

```sh
checkpoint_dir=$(mktemp -d)
ml run examples/attention.ml --save "$checkpoint_dir/attention.mlckpt"
ml run examples/attention_infer.ml --load "$checkpoint_dir/attention.mlckpt" --inference --optimize
ml run examples/attention.ml --load "$checkpoint_dir/attention.mlckpt" --save "$checkpoint_dir/resumed.mlckpt" --backend tiled
```

The example reconstructs two sequences of three two-feature toy tokens. It learns
three shared projections using 500 Adam updates at rate `0.03`, betas `(0.9,0.999)`,
epsilon `1e-8` and global clipping at `1`. Deterministic CPU loss decreases from
`1.2894703646` to `0.0000097899`. Training and a separate one-sequence query are
checked within absolute tolerance `0.03`. Outputs are reconstructed features,
not probabilities; they can be slightly negative or above one.

This is a functional toy reconstruction task, not language modeling or a dataset
accuracy/generalization result. Attention is single-head, dense and unmasked.
This example omits causal/padding masks, dropout, residual layers, normalization, positional encoding,
KV caches and efficient/fused attention kernels are not implemented. Score storage
still scales quadratically with token count. No GPU memory claim is made.

The same model accepts a different compile-time specialized batch size on load.
Checkpoints save projection weights and Adam moments/counters. A 211-step save
followed by 289 resumed steps matches 500 uninterrupted steps exactly on the
tested backend/toolchain. Data cursor state is not serialized. `attention_infer.ml`
deliberately requires trained weights; use `scripts/smoke.sh` to exercise both
examples in order. Both CPU backends and optional SSA inference are covered.

## Numerical validation and benchmarks

Tests check all shared/per-batch input derivatives by central finite differences,
all transpose-last elements across ranks, and a composed attention gradient for
each projection. Tests also cover symbolic rejection cases, irregular large tiled
products, complete reference/tiled output and gradient equality, checkpoint resume,
SSA agreement and bounded retained training-loop payload.

```sh
cargo build --release --locked
/workspace/.ml-lang-bench-venv/bin/python benchmarks/attention.py --samples 5 --output /tmp/ml-lang-attention.json
```

Use `benchmarks/requirements.txt` in a venv outside this prepared cloud environment.
The comparison matches initial/final loss, all 18 training/query feature outputs
and all 20 final projection weights at `rtol=1e-9, atol=1e-10`. PyTorch uses eager
batched products with native Adam; JAX JIT-compiles its loop with functional Adam.
Both implement the language's epsilon-free clipping policy explicitly. Each mode
runs in a fresh process with one CPU affinity/framework math thread, one warmup
and five measured executions. Parameters reset each run; JAX results synchronize.

Warm times exclude imports, initial JIT and ml-lang compilation, and include
initialization, training, query evaluation, validation and returned weights.
Native timing includes model-state snapshots and extra buffered weight diagnostics;
frameworks return flattened weight lists. Input data allocation/caching and
validation bookkeeping differ. Peak RSS includes imports/JIT/compiler constants,
warmup and result snapshots, not isolated tensor allocator efficiency. Checkpoint
file I/O is excluded. This is not a comparison against compiled PyTorch, production
attention libraries or GPUs. Exact source/binary/harness hashes accompany the report.

Current raw evidence: [attention-cpu.json](../benchmarks/attention-cpu.json).

| Backend | Median complete workload (ms) | Peak process RSS (MiB) |
| --- | ---: | ---: |
| ml-cpu | 5.374 | 1.72 |
| ml-tiled | 5.384 | 1.71 |
| torch-eager | 193.046 | 227.08 |
| jax-jit | 2.173 | 171.49 |

All toy matrix products fall below the tiled threshold, so this example measures
that backend's fallback path, not a batched-tiling speedup. Native/reference and
tiled outputs and weights match exactly; PyTorch/JAX match the stated tolerance.
These tiny eager/compiled timing differences and process footprints cannot
establish superiority over production frameworks or GPU attention.

The retained unmasked attention benchmark predates `causal_softmax`; its hashes
identify the measured implementation. It does not measure causal attention
or the current executable. See [causal attention](causal-attention.md).
