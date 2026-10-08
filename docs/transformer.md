# Trainable causal Transformer block

[transformer.ml](../examples/transformer.ml) implements a one-block CPU f64
Transformer entirely in ml-lang. It uses two width-two causal attention heads,
learned layer-normalization scale/bias, pre-norm residual connections, a
width-eight GELU feed-forward layer, and a two-class token readout. All
14 parameters (166 scalar values) train with clipped Adam. No external
framework implements its forward or backward pass.

The function specializes symbolic batch/token dimensions. Head count, feature
width and FFN width are explicit constants; dimension arithmetic is absent.
It is an ordinary function composition, not a built-in Transformer or model DSL.

## GELU semantics

`gelu(x)` preserves all shapes, including scalar tensors. The implementation
uses the standard tanh approximation:

`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.

Its VJP is the derivative of that approximation, not exact erf GELU.
For finite `x > 10` it returns `x` with derivative 1; for `x < -10` it
returns zero with derivative zero. These saturated tails avoid overflowing
the cubic, where f64 tanh has already saturated. At zero the derivative
is 0.5. The CPU and tiled backends share this elementwise kernel, and
forward-only SSA supports the operation.

## Train, save and infer

```sh
ml check examples/transformer.ml
ml run examples/transformer.ml
ml run examples/transformer.ml --save transformer.mlckpt
ml run examples/transformer_infer.ml --load transformer.mlckpt --inference
ml run examples/transformer_infer.ml --load transformer.mlckpt --inference --optimize --backend tiled
```

Use a new checkpoint path: saving never overwrites an existing file.
The standalone inference example requires the trained checkpoint; its zero
initializers only declare matching parameter shapes. Inference returns
last-axis softmax probabilities. The optimized path uses existing typed
SSA and last-use reclamation.

The fixed synthetic dataset contains two sequences of three tokens. Inputs
have four manually supplied features, labels have two probability classes,
and the classifier reproduces a category already present in the inputs.
Cross entropy flattens `[2,3,2]` logits/targets to `[6,2]`. The 300-step
example reduces loss from approximately 0.710182 to `1.87e-10`. Assertions
check training probabilities and the first training sequence evaluated with
batch size one. This is deliberately a correctness example: there is no
held-out accuracy, language modeling, text generation or generalization claim.

## Validation

Six new automated tests cover GELU reference values and finite extreme tails,
all GELU input finite differences, symbolic shapes/arity and SSA execution,
exact 137 + 163 step checkpoint resume versus uninterrupted training, loaded
CPU/tiled reference and SSA inference, central differences for **every one of
the 166 model parameter values**, and past-output invariance/zero future
input gradients. The smoke script exercises training, checkpoint saving and
loaded inference through both execution paths.

Optional independent comparison using the benchmark Python environment:

```sh
python benchmarks/transformer_check.py
```

It compares two batch/token shapes and both CPU backends against independently
written PyTorch and JAX blocks. Logits, cross entropy and all 166 parameter
gradients must match at `rtol=atol=1e-10`. PyTorch uses
`gelu(..., approximate="tanh")`; JAX uses approximate GELU. This checks
numerical correctness only and measures neither execution time nor memory.

## Scope

No tokenizer, embedding lookup, learned positional embeddings, attention
padding mask, dropout, KV cache, generation, stacked blocks, mixed precision,
GPU or fused kernels are implemented. Features and labels are source literals.
The block uses copied layouts and dense quadratic attention scores. Earlier
kernel/framework benchmark reports do not measure this model. No performance
or model-quality advantage is claimed.
