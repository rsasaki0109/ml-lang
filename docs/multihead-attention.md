# Multihead causal attention

The language now supports `bmm` with multiple batch axes and differentiable
`swap_axes`. [The working example](../examples/multihead_attention.ml)
implements two causal heads using ordinary functions and Tensor operations.

## Layout and types

`bmm(a,b)` requires LHS rank >= 3, `[...,M,K]`. RHS is either shared
`[K,N]` or the same rank with exactly matching batch prefix `[...,K,N]`.
Output is `[...,M,N]`. A shared RHS gradient sums every flattened batch
coordinate in row-major order. Partial broadcasting, including batch-one
expansion, is rejected. `@` remains rank-2. Symbolic prefixes must match
structurally even in unused functions. CPU and tiled kernels use the same
flattened batch traversal; each product retains its reduction order.

`swap_axes(x,a,b)` swaps any two zero-based constant axes. Its output is a
row-major copy, not a view. Its VJP applies the inverse permutation, which
is the same swap. Equal axes are valid. Negative, fractional, symbolic or
out-of-range axes are rejected. No axis renumbering happens implicitly.

The example projects `Tensor[B,T,2]` with three separate `Tensor[2,4]`
weights, reshapes each to `[B,T,2,2]`, and swaps axes 1 and 2 to `[B,2,T,2]`.
The two heads have distinct projection columns. Scores are `[B,2,T,T]`,
scaled by `1/sqrt(2)` and normalized with `causal_softmax`. Context is
`[B,2,T,2]`; swapping back and reshaping merges it into `[B,T,4]`. A learned
`Tensor[4,2]` output projection returns `[B,T,2]`.

Dimensions are deliberately explicit: shape arithmetic is not supported,
so this function fixes two heads of width two while specializing symbolic
batch and token dimensions. Other fixed head counts/widths can be written
using the same operations.

## Run and validation

```sh
ml check examples/multihead_attention.ml
ml run examples/multihead_attention.ml
ml run examples/multihead_attention.ml --backend tiled
```

The 500-step deterministic toy reconstruction learns all four projections
with clipped Adam. MSE decreases from approximately 0.420896 to 0.000003714.
Assertions check both training sequences and a one-sequence query. This is
not a language model or a demonstration of generalization.

Six new tests check hand-calculated axis layout and all swap VJPs, all input
finite differences for rank-4 shared and batched products, symbolic/concrete
shape rejection, large tiled forward/VJP equality, exact 217 + 283 step
checkpoint resume and loaded CPU/tiled SSA inference, and each projection's
composed gradient plus future-token independence.

Optional PyTorch/JAX comparison, using the benchmark Python environment:

```sh
python benchmarks/multihead_check.py
```

It compares two batch/token shapes through CPU and tiled backends, checking
all outputs, MSE and all 32 parameter gradients with `rtol=atol=1e-10`.
This is a numerical check, with no timing or memory measurement.

## Limitations

Dense CPU f64 only. The explicit score matrix costs O(B*H*T*T) elements;
axis swaps allocate copies. There is no fused attention, general batched
broadcasting, padding mask, KV cache, dropout, GPU execution or symbolic
dimension arithmetic. Retained benchmark reports predate this extension;
no speed or memory advantage is claimed.

Axis swaps now copy contiguous suffix blocks without per-element coordinate
allocations. See [CPU permutation ablation](axis-swap.md) for measured scope,
including the absence of demonstrated peak-RSS reduction.
