# Causal single-head attention

`causal_softmax(scores)` is a CPU f64 differentiable primitive for square
attention scores `Tensor[...,T,T]`. Query row `i` can attend to keys `j <= i`.
The diagonal is visible, so every row has at least one visible entry. The
output preserves shape; arbitrary leading axes are independent matrices.

Only the visible prefix participates in the stable log-sum-exp computation.
Masked logits never enter normalization; masked probabilities and input
gradients are exactly zero. For visible entries, the VJP is
`p_j * (g_j - sum_visible(p_k * g_k))`. This avoids introducing infinity
constants into the language. Symbolic final axes must be provably equal.

```ml
fn causal(x: Tensor[B,T,D]) -> Tensor[B,T,D] {
    let scores = bmm(x,transpose_last(x)) * 0.5;
    return bmm(causal_softmax(scores),x);
}
let x = [[[1,0],[0,1],[1,0]]];
print(causal(x));
```

Run the complete trainable projection example:

```sh
ml check examples/causal_attention.ml
ml run examples/causal_attention.ml
ml run examples/causal_attention.ml --backend tiled
```

The deterministic 500-step toy reconstruction example uses three learned
projections, clipping and Adam. On the validated CPU implementation, initial
MSE is approximately 1.421738 and final MSE is 0.000021711. Assertions check
reconstruction and a separate single-sequence query. This is a correctness
demonstration, not a language model or evidence of generalization.

`tests/causal_attention.rs` checks stable normalization with extreme finite
masked logits; every input gradient by central differences for ranks 2, 3 and
4; concrete and symbolic shape rejection; prefix output invariance under
changes to future tokens and zero future gradients; and exact 213 + 287 step
checkpoint resume versus uninterrupted training. Loaded reference and SSA
inference agree with CPU and tiled backends.

The score/output buffers are still dense and quadratic in sequence length.
There is no sparse triangular storage, fused attention, padding mask,
arbitrary mask, rectangular KV-cache decoding, multihead layout, dropout or
GPU support. No new performance advantage is claimed; previous unmasked
attention benchmark results do not measure this primitive.
