# ml-lang

**A standalone, shape-checked programming language for machine learning.**
Write tensor computations, differentiate them, and train small neural networks
using an independent Rust compiler and runtime, including batched single-head
attention defined with ordinary functions. No Python, PyTorch, or JAX is
needed to run an ml-lang program. An optional typed inference IR releases
temporary tensors after their last use. An optional tiled CPU backend provides cache-tiled
matrix products and gradient kernels without a framework dependency.

```ml
param weights: Tensor[2,1] = [[0.1],[-0.2]];
let inputs: Tensor[4,2] = [[0,0],[0,1],[1,0],[1,1]];
let targets: Tensor[4,1] = [[0],[1],[1],[2]];

repeat 500 {
    let loss = mse(inputs @ weights, targets);
    backward(loss);
    step(0.1);
}
print(inputs @ weights);
```

**Status: experimental v0.1 CPU prototype.** Static tensor shapes, reverse-mode
autodiff, symbolically shaped functions, SGD/Adam, gradient clipping, minibatches, and model checkpoints work today. All values use
`f64` on CPU. GPU execution, fusion, mixed precision, general control flow,
and native machine-code generation are not implemented. High performance and
low GPU memory use are goals, not demonstrated advantages.

## Install

Install [Rust](https://rustup.rs/), then build the dependency-free implementation:

```sh
git clone https://github.com/rsasaki0109/ml-lang.git
cd ml-lang
cargo build --release --locked
cargo install --path . --locked
```

`rust-toolchain.toml` pins Rust 1.85.1. Linux x86-64 is validated; other platforms
are unverified. There are no GPU drivers, BLAS libraries, Python packages, or
external services required for the language itself. Release artifacts have not
been published.

## Quick start

```sh
ml check examples/xor.ml
ml run examples/xor.ml
ml run examples/autodiff.ml
ml run examples/minibatch.ml
ml run examples/attention.ml
ml ir examples/matmul.ml
ml run examples/tiled_matmul.ml --backend tiled
ml plan examples/inference_memory.ml
ml run examples/inference_memory.ml --inference --optimize
```

Without installing the CLI, use `cargo run --release -- run examples/xor.ml`.
The XOR example defines a 2-layer, 8-hidden-unit `tanh` network and trains it for
2,000 SGD steps, entirely in ml-lang. Its deterministic initial loss is about
`1.7236`, and its final loss is about `0.000405`. Predictions approach
`[-1, 1, 1, -1]`; the program checks all four outputs with an absolute tolerance
of `0.06`. These are small-scale functional results, not model-quality claims.

Train, save, and run inference in a separate process:

```sh
checkpoint_dir=$(mktemp -d)
ml run examples/adam_multiclass.ml --save "$checkpoint_dir/model.mlckpt"
ml run examples/adam_infer.ml --load "$checkpoint_dir/model.mlckpt" --inference
ml run examples/adam_multiclass.ml --load "$checkpoint_dir/model.mlckpt" --save "$checkpoint_dir/resumed.mlckpt"
```

The Adam example reduces loss from `1.3394` to `0.0005468` in 300 steps.
Its generic model handles six training samples and three separate toy queries.
`--inference` disables gradient tracking and rejects training operations.
Checkpoints retain weights and Adam moments/counters for exact continuation;
existing files are never overwritten. Add `--optimize` to inference runs to
use typed SSA execution with last-use reclamation. See [checkpoint semantics](docs/checkpoints.md).
This is a functional toy demonstration, not a generalization benchmark.

## Language

```ml
fn predict(x: Tensor[B,2], w: Tensor[2,1]) -> Tensor[B,1] {
    let projection = x @ w;
    let activations = tanh(projection);
    return activations;
}
let x: Tensor<f64,cpu>[4,2] = [[0,0],[0,1],[1,0],[1,1]];
param w = rand(2,1,42);
let loss = mean(predict(x,w) * predict(x,w));
backward(loss);
print(grad(w));
```

`Tensor[4,2]` is shorthand for `Tensor<f64,cpu>[4,2]`; `Tensor[]` is a scalar.
Unsupported dtype/device annotations are errors. Shape inference checks
rectangular literals, rank-2 matrix multiplication, broadcasting, function
arguments/returns, and annotations before execution.

| Feature | Syntax |
| --- | --- |
| Immutable binding / trainable binding | `let x = ...;` / `param w = ...;` |
| Elementwise arithmetic / matrix multiplication | `+ - * /` / `@`, `bmm(a,b)` |
| Trailing-axis broadcasting | scalar, row bias, or dimensions of size 1 |
| Initialization | `zeros(dim...)`, `ones(dim...)`, `rand(dim..., seed)` |
| Differentiable operations | `tanh`, `gelu`, `sigmoid`, `relu`, `sum`, `mean`, `mse`, `bce_logits`, `softmax`, `causal_softmax`, `layer_norm`, `cross_entropy` |
| Shape operations | `transpose(matrix)`, `transpose_last(x)`, `reshape(x, dim...)`, `rows(x,start,count)` |
| Gradient control | `detach(x)`, `clip_grad_norm(maximum);` |
| Learning | `backward(loss); step(rate);` or `adam_step(rate);` |
| Inspect gradients / check results | `grad(w)`, `assert_close(a,b,tolerance);` |
| Fixed-count loops | `repeat 2000 { ... }` |

Function dimensions such as `B` are inferred from argument shapes and specialized
at compile time. The same function can process different batch sizes.
Functions have explicitly shaped arguments, sequential local `let` declarations,
and a final `return` expression;
recursion and closures are unsupported. `rand` uses deterministic SplitMix64
uniform values in `[-1,1)` with a positive integer seed. Bindings are lexical;
loop locals do not escape. Parameters must be declared outside loops.

`backward` resets previous gradients. `step` updates connected parameters with
SGD; `adam_step` uses bias-corrected Adam. Both detach the graph. Every repeat-iteration boundary also detaches the
graph and drops local slots. Consequently, backpropagation through loop
iterations is unsupported. Recompute a loss after each step; old losses cannot
be differentiated again. A finite, positive learning rate is required.
`print` results are buffered and returned after execution, so repeated printing
consumes memory. For binary classification, use `bce_logits(logits, targets)` for training and
`sigmoid(logits)` for probabilities. Targets must be in `[0,1]`; the loss averages
over all elements and requires equal shapes. The stable implementation supports
large positive/negative logits without taking the logarithm of rounded probabilities.
For multiple classes, use `cross_entropy(logits, targets)` with equal
`[batch,classes]` shapes and one-hot or soft-label target rows summing to one.
`softmax(logits)` produces probabilities along the last axis. Class-index labels
are not implemented. The toy classifier is a training-fit demonstration, not a
held-out accuracy benchmark.
See the [language specification](docs/language.md).

## Examples

- [Transformer block](examples/transformer.ml): train a causal pre-norm block with GELU, residuals and a token classifier; [guide](docs/transformer.md).
- [Axis swap performance](docs/axis-swap.md): measured CPU permutation improvements with unchanged copying semantics.
- [Multihead attention](examples/multihead_attention.ml): train two causal heads and an output projection; [semantics and validation](docs/multihead-attention.md).
- [Layer normalization](examples/layer_norm.ml): train an affine normalized network; [semantics and validation](docs/layer-norm.md).
- [Causal attention](examples/causal_attention.ml): train a single head with future positions excluded; [semantics and validation](docs/causal-attention.md).
- [Attention](examples/attention.ml): train shared batched projections and load a one-sequence query.
- [Tiled matmul](examples/tiled_matmul.ml): use the optional CPU kernel backend.
- [Inference memory](examples/inference_memory.ml): compare reference and SSA temporary lifetimes.
- [Minibatch training](examples/minibatch.ml): row batches, global gradient clipping and Adam.
- [Stop gradients](examples/detach.ml): freeze a selected gradient path.
- [Adam classifier](examples/adam_multiclass.ml): train, save, resume, and test toy queries.
- [Checkpoint inference](examples/adam_infer.ml): load trained weights in a separate process.
- [XOR network](examples/xor.ml): define, train, and infer a nonlinear network.
- [Multiclass classifier](examples/multiclass.ml): fit three toy clusters with softmax inference.
- [Binary XOR classifier](examples/binary_xor.ml): stable logits loss and probability inference.
- [Autodiff](examples/autodiff.ml): verify `d sum(x*x) / dx = 2*x`.
- [Matrix multiplication](examples/matmul.ml): statically shaped matrix kernels.
- [Linear regression](examples/linear.ml): the README training example with assertions.

## Validation and benchmarks

```sh
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo test --release --locked
ml bench examples/xor.ml 5
```

Tests cover compiler rejection cases, CLI failure status, every differentiable
primitive against central finite differences, a composed network gradient,
SGD/Adam trajectories, symbolic shape rejection, exact training resume, malformed
checkpoint rejection, separate-process inference, row-scatter gradients, stop-gradient
snapshots, stable global clipping, inference IR/VM equivalence, last-use releases,
and bounded retained tensor storage.

The optional [comparison harness](benchmarks/compare.py) measures CPU `f64`
matmul, MSE/BCE XOR, and SGD/Adam and clipped minibatch multiclass training against PyTorch eager and JAX JIT, verifies matching numerical
outputs, and records wall time and per-process peak RSS. See
[measured results and methodology](docs/benchmarks.md). The small eager training
comparison does **not** establish an advantage over compiled frameworks or GPUs.

## Design and development

The frontend and bytecode VM are independent of ML frameworks. A kernel dispatch
interface separates numerical operations from execution. The current host-tensor
ABI will need device buffers and asynchronous execution for a GPU-native backend.

- [Batched attention, shapes and gradients](docs/attention.md)
- [Tiled CPU kernels and numerical validation](docs/tiled-cpu.md)
- [Inference IR and last-use semantics](docs/inference-ir.md)
- [Architecture and backend decisions](docs/architecture.md)
- [Research and tradeoffs](docs/research.md)
- [Roadmap and acceptance criteria](docs/roadmap.md)
- [Known limitations](docs/limitations.md)
- [Contributing](CONTRIBUTING.md)

CI builds, lints, tests, and executes examples. MIT licensed; see [LICENSE](LICENSE).
