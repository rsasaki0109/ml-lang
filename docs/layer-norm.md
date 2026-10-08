# Last-axis layer normalization

`layer_norm(x)` maps `Tensor[...,D]` to the same shape, for rank >= 1.
Each last-axis row is independent. It computes
`y = (x - mean(x)) / sqrt(mean((x - mean(x))^2) + 1e-5)`.
Variance is population variance, not an unbiased sample estimate. Epsilon
is fixed at `1e-5` in this initial primitive. A width-one row returns zero
and has zero input derivative; constant wider rows return zero but generally
have nonzero derivatives. Scalars are rejected, including in unused functions.

The CPU f64 implementation scales inputs before computing statistics, anchors
the scaled mean at the first element, and uses `hypot` for the denominator.
This avoids overflowing variance squares for extreme finite values and keeps
constant rows exactly zero. In ordinary f64 arithmetic, nearly equal values
with large offsets can still lose precision. Values and derivatives are
approximate floating-point calculations, not exact real arithmetic.

Given upstream gradient `g`, the VJP is
`inverse_std * (g - mean(g) - y * mean(g*y))`.
Learned scale and bias are ordinary parameters using broadcasting:

```ml
fn normalized(x: Tensor[B,T,D], gain: Tensor[D], bias: Tensor[D]) -> Tensor[B,T,D] {
    return tanh(layer_norm(x) * gain + bias);
}
param gain = ones(3);
param bias = zeros(3);
let x = [[[1,2,4],[4,1,2]]];
let y = normalized(x,gain,bias);
print(y);
backward(sum(y));
print(grad(gain));
print(grad(bias));
```

For residual causal attention, compose
`layer_norm(x + attend(x,wq,wk,wv,scale))` when the output feature width
matches the input. Normalization affects only the current token's features,
so it does not add future-token dependencies. No Transformer architecture
is built into the language.

## Run and validate

```sh
ml check examples/layer_norm.ml
ml run examples/layer_norm.ml
ml run examples/layer_norm.ml --backend tiled
```

The deterministic 300-step example learns scale/bias through `tanh` from a
fixed synthetic teacher. MSE decreases from approximately 0.072055 to below
`1e-12`; assertions also check a separate query with a constant row. This
is a training correctness demonstration, not evidence of model quality.

Six tests cover handcrafted forward values, extreme finite and constant
rows, every input gradient via central differences for ranks 1–3, concrete
and symbolic type checks, exact 117 + 183 step checkpoint resume, CPU/tiled
loaded reference/SSA agreement, and a composed residual causal-attention
gradient. The primitive uses the same CPU implementation through both
backend dispatch paths.

Optional independent numerical comparison (requires the benchmark Python
environment described in [benchmarks](benchmarks.md)):

```sh
python benchmarks/layer_norm_check.py --binary target/release/ml
```

Eight CPU f64 cases (ranks 1–3, varying and constant values) compare forward
outputs and every input VJP against PyTorch's layer norm and an independent
JAX expression, with `rtol=atol=1e-10`. The validated optional environment
uses PyTorch 2.6.0+cpu and JAX 0.4.38. This comparison is a correctness check;
it measures neither timing nor memory.

## Limits

Only dense CPU f64 and last-axis normalization are supported. There are no
configurable epsilon, multiple normalized axes, batch norm, RMS norm, mixed
precision, fusion or GPU kernels. Existing benchmark reports predate this
primitive and do not measure its speed or memory use. No performance
advantage is claimed.
