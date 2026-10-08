# v0.1 language reference

A program contains function declarations and statements. Functions can be
forward-referenced. Statement order is significant. Keywords and identifiers
are ASCII. `//` starts a line comment. Statements end in `;` except repeat blocks.

```ebnf
program   = { function | statement } ;
function  = "fn", identifier, "(", [ parameters ], ")", "->", type,
            "{", { local }, "return", expression, ";", "}" ;
local     = "let", identifier, [ ":", type ], "=", expression, ";" ;
parameter = identifier, ":", type ;
type      = "Tensor", [ "<f64,cpu>" ], "[", [ dimensions ], "]" ;
statement = ("let" | "param"), identifier, [ ":", type ], "=", expression, ";"
          | "repeat", integer, "{", { statement }, "}"
          | identifier, "(", [ expressions ], ")", ";" ;
expression = number | identifier | array | call | "(", expression, ")"
           | "-", expression | expression, operator, expression ;
operator  = "+" | "-" | "*" | "/" | "@" ;
```

Grammar notation omits list punctuation: all dimension, argument and element
lists use commas, with no trailing commas. Function bodies contain zero or more local `let` declarations followed by
exactly one final `return` expression. There are no imports, assignments, conditionals, strings,
booleans, classes or recursion. Parameters and `let` variables have identical
value types; `param` additionally registers an optimizer-owned differentiable leaf.
Declarations cannot shadow a live lexical binding. Function parameters are local.
Repeat locals do not escape. Function invocations are inlined.

`@`, `*`, `/` share precedence above `+`, `-`, and associate left-to-right.
Unary minus has higher precedence. Numbers allow decimal/exponent notation;
non-finite literals are rejected. Negative array literal entries are supported.
Array literals must be nonempty, rectangular numeric constants. Scalars have
shape `[]`; reduction operations always return scalars. Dimensions are positive
integers or function-bound symbolic names (see below). The total element limit is 16,777,216 per tensor, not a global
memory quota. Type annotations allow up to rank 8. Initializer dimensions and
random seeds are positive integers up to 1,000,000; repeat counts may be zero
and have the same upper bound. Nesting limits guard expressions and blocks;
this interpreter is not a hardened sandbox for hostile input.

Matrix multiplication is rank-2 `[m,k] @ [k,n] -> [m,n]`. Elementwise operators
align shapes from the right; dimensions must be equal or one. `mse(a,b)` requires
exactly equal shapes and means `(a-b)*(a-b)` over all elements. Division by zero,
non-finite kernel output or gradients, and non-finite optimizer updates are
runtime errors. `tanh`, `sigmoid`, `relu`, `sum`, `mean` each take one argument. ReLU's
subgradient at zero is zero.

`backward(loss)` requires a scalar connected to a parameter and resets earlier
gradients. `grad(value)` reads a gradient produced by the latest backward pass;
unconnected values are errors. Higher-order AD is unsupported: returned gradients
are detached. `step(rate)` requires a backward pass since the most recent step
or loop boundary; disconnected parameters are unchanged. Graphs are truncated
at steps and iteration boundaries. Saved intermediate losses are detached there
and cannot be reused for gradient propagation. No backpropagation through time.
`let` aliases to parameter leaves share their current parameter value when an
optimizer updates it; the language does not promise snapshot semantics for those
aliases in v0.1.

`assert_close(a,b,tolerance)` requires equal shapes and a finite, nonnegative
scalar absolute tolerance. `print(value)` records a tensor for CLI output after
execution. Runtime errors abort the program; buffered prints are not emitted on
failure. `check` never executes initializers or training; it cannot establish
numerical convergence or catch division by zero.

All values are CPU `f64`, including scalar constants. `f32`, GPU devices and
mixed-device/mixed-dtype conversion have no fallback: unsupported annotations
are rejected. Symbolic function shapes are specialized at compile time;
data-dependent shapes remain unsupported.

## Shape transformations and binary classification

`transpose(x)` swaps the two axes of a rank-2 tensor; other ranks are rejected
at compile time. `reshape(x, dim...)` preserves row-major element order and must
preserve element count. Target dimensions are positive integers or bound symbolic names, with
at most rank 8. `reshape(x)` produces a scalar only for a one-element input.
Neither operation is a view in the current runtime: both copy storage. Their
VJPs reverse the permutation or restore original shape respectively; gradients
are accumulated when transformed and original values share an input.

`sigmoid(x)` is elementwise and preserves shape. Positive and negative branches
avoid exponential overflow. `bce_logits(logits, targets)` returns scalar mean
binary cross entropy and requires exactly equal shapes, without broadcasting.
Targets are checked at execution time to be in `[0,1]`, including soft labels.
Use raw logits, not sigmoid probabilities, as the first argument. Per-element
loss is `max(z,0) - z*y + log1p(exp(-abs(z)))`, with derivative `sigmoid(z)-y`,
scaled by element count. Targets also have derivative `-z / element_count` if
registered as parameters. This is first-order differentiation only. Logits of
`-1000` and `1000` are covered by numerical stability tests. Reductions can still
overflow for values near the f64 limit and then return a runtime error.

## Multiclass classification

`softmax(x)` accepts rank >= 1 and normalizes each row along the last axis.
Its result has the same shape. No axis argument is accepted in v0.1. The VJP
is `p * (g - sum(p*g))` independently for each last-axis row.

`cross_entropy(logits, targets)` requires equal rank-2 `[batch,classes]` tensors
and produces a scalar **mean over the batch**, summing classes within each row.
Targets can be one-hot or soft labels. Every value must be in `[0,1]`, and each
row must sum to one within absolute tolerance `1e-9`; this is checked at runtime.
Targets are not automatically renormalized. Class-index targets, class weights,
ignore-index and automatic label smoothing are not implemented. Pass raw logits,
not softmax probabilities.

The implementation subtracts the row maximum, computes shifted log probabilities,
and directly forms `-sum(target * log_probability) / batch`. It avoids forming
`log(softmax(z))` and avoids adding the huge original row offset back into
log-sum-exp. Thus uniform logits near `1e16` still produce loss `log(classes)`.
Zero target terms are skipped when log probabilities have an unrepresentable
negative gap. A positively weighted gap or loss beyond f64 range returns an
error rather than an infinite result.

For targets with row mass `s`, the logit gradient is `(s*p-target) / batch`,
including the upstream scalar multiplier; this handles the small normalization
tolerance consistently. If targets are registered as parameters, their gradient
is `-log_probability / batch`; constrained target directions can be checked by
moving probability between classes. Optimizer updates are not projected onto a
probability simplex, so learned label values may be rejected on later execution.
All gradient and shape rules remain first-order CPU f64 operations.

## Function-local computations

A function can name intermediate values, with inferred or annotated tensor shapes:

```ml
fn predict(x: Tensor[4,2], w: Tensor[2,3], b: Tensor[1,3]) -> Tensor[4,3] {
    let projection: Tensor[4,3] = x @ w;
    let logits = projection + b;
    let probabilities = softmax(logits);
    return probabilities;
}
```

Local declarations are checked sequentially; use before declaration, duplicate
names, parameter shadowing and annotation mismatches are errors. Callees can
have the same local names as their callers without sharing bindings. Function
locals cannot read global variables or escape their scope. Unused functions
are still fully type checked, including their locals.

Only `let` declarations and a final return are supported inside functions.
`param`, `repeat`, printing, optimizer updates and backward statements remain
at program/block level. There is no early return or reassignment. Calls still
inline into bytecode with distinct argument/local slots. Local values preserve
tape edges and shared-value gradient accumulation. Inlined locals inside repeat
blocks are released at iteration boundaries; no new call-frame VM is introduced.


## Symbolic function dimensions

```ml
fn classify(x: Tensor[B,2], w: Tensor[2,C], b: Tensor[1,C]) -> Tensor[B,C] {
    let z: Tensor[B,C] = x @ w + b;
    return softmax(z);
}
```

Names in parameter annotations implicitly bind dimensions. Every occurrence of
one name must unify to the same dimension. Each function has its own dimension
scope; callers and callees can reuse names. Locals, return annotations, `zeros`,
`ones`, `rand` dimensions and `reshape` may reference bound names. Dimension names
are distinct from tensor values and cannot be used in ordinary arithmetic.
Top-level annotations must be concrete. Random seeds remain literal integers.

All functions are checked structurally, including unused functions. Broadcasting
requires dimensions provably equal or one; unknown names are not assumed equal.
Reshape requires identical constant products and symbolic-factor multisets.
There are no dimension arithmetic expressions, constraints, inferred `-1`, or
runtime dimension values. At each call, the compiler substitutes concrete shapes
and emits specialized bytecode; repeated calls are inlined, not cached native code.
Recursion is rejected and call depth is limited to 32.

## Adam optimizer

`adam_step(rate)` defaults to `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`.
`adam_step(rate,beta1,beta2,epsilon)` specifies all options. Rate and epsilon must
be finite and positive, betas finite in `[0,1)`. A backward pass is required.
For each connected parameter, the counter advances and the update is:

```text
m = beta1*m + (1-beta1)*g
v = beta2*v + (1-beta2)*g*g
p = p - rate * (m/(1-beta1^t)) / (sqrt(v/(1-beta2^t)) + epsilon)
```

Moments and counters belong to parameter slots, survive tape compaction, and
are advanced only for connected parameters. Disconnected parameters are unchanged.
All candidate updates are validated before any weights or optimizer state mutate.
Betas/epsilon cannot change after Adam initialization; learning rate can change.
SGD does not erase existing Adam moments. Weight decay, AMSGrad, clipping,
schedulers and momentum SGD are unsupported.

## Saved models and inference

`ml run source.ml --save model.mlckpt` writes weights and Adam state after successful
execution. `--load` requires exactly matching parameter names and shapes, replaces
initial values, and restores optimizer state. `--inference` loads weights only,
disables gradient tracking, and rejects `grad`, `backward`, `step`, `adam_step`,
including inside zero-count loops. It cannot be combined with `--save` in the CLI.
See [checkpoints](checkpoints.md) for format, failure and continuation guarantees.

## Row extraction, minibatches and stopped gradients

`rows(x,start,count)` copies a contiguous range on the leading axis of any
rank-one-or-higher tensor. Start is a literal nonnegative integer; count is a
literal positive integer. Both are limited to 1,000,000. The compiler verifies
`start + count <= x.shape[0]`, and the result replaces that first dimension with
`count`. Empty ranges, negative indexing, strides, runtime indices and shuffling
are unsupported. A function using `rows` must have a concrete leading dimension
so bounds can be proven even without calls; trailing dimensions may be symbolic.
The VJP scatters into a zero-filled original shape; overlapping uses accumulate.
This currently copies storage, rather than creating an alias or view.

```ml
let x = [[1,2],[3,4],[5,6],[7,8]];
let batch = rows(x,1,2);
assert_close(batch,[[3,4],[5,6]],0);
```

`detach(x)` creates an independent value snapshot with the same shape, dtype and
device, without any gradient edge. Unlike a direct parameter alias, it retains
its original values after optimizer updates. A loss consisting entirely of
detached values cannot be differentiated; `grad` on a detached value is an error.
`detach` can be used inside functions or explicit inference mode. No higher-order
AD is introduced; this is a selected-path stop-gradient operation.

## Gradient clipping

`clip_grad_norm(maximum)` requires a fresh `backward` and a finite positive scalar
maximum. It computes the global L2 norm across **connected parameter gradients**
and rescales them together only when that norm exceeds the maximum. It does not
clip each parameter independently, and does not include nonparameter tape nodes.
Zero gradients are unchanged. Parameter slots are visited in declaration order
for repeatable reductions. Largest-component scaling avoids overflow from directly
squaring huge gradients; applying the transform as a normalized value times the
target magnitude also avoids underflow of a tiny overall scale factor.
Non-finite gradients remain errors and cannot be repaired
by clipping. There is no additive epsilon in the norm or scale definition.

```ml
param x = [3,4];
backward(sum(x*x)/2);
clip_grad_norm(2);
assert_close(grad(x),[1.2,1.6],1e-12);
step(0.1);
```

Clipping changes the stored leaf gradients used by both SGD and Adam; it leaves
existing detached `grad` snapshots and intermediate tape gradients unchanged.
It neither advances optimizer counters nor detaches the graph. A new backward
resets clipped gradients. Repeated clipping is allowed. Steps and repeat boundaries
invalidate the backward state, so clip before the optimizer update. Explicit
inference rejects clipping even inside zero-count loops.

`examples/minibatch.ml` visits three fixed contiguous batches of two samples per
epoch, performs clipping followed by Adam, and evaluates the whole dataset and
three toy queries. It demonstrates minibatch updates, not a data-loader ecosystem;
there are still no external datasets, runtime batch indices or epoch shuffling.


## Inference execution choices

`ml run source.ml --inference` uses forward-only reference-VM execution.
Adding `--optimize` lowers typed SSA once and releases temporary values at their
last use. Numerical operations and observable bindings remain unchanged.
`ml plan source.ml` displays that representation and release schedule;
`ml bench source.ml 5 --inference --optimize` measures repeated planned execution.
Training operations are rejected in either inference mode, including in zero
loops. See [inference IR](inference-ir.md) for captures, root lifetimes and limits.


## CPU backend selection

`run` and `bench` accept `--backend cpu` (default) or `--backend tiled`.
Both use CPU f64 and the same shape rules. Selection works with training,
checkpoint restoration and either inference mode; it does not enable GPU or
SSA autodiff. `plan` is backend-independent. See [tiled CPU](tiled-cpu.md) for
small-work fallback, scratch memory and numerical validation.


## Batched products and attention

`bmm(a,b)` maps `[...,M,K]` and shared `[K,N]` or matching `[...,K,N]` to
`[...,M,N]` (LHS rank >= 3). Shared weights accumulate gradients over batches. Per-batch dimensions
must match exactly; singleton-batch broadcasting is unsupported; multiple batch axes are supported. Rank-two `@` is unchanged. Symbolic function signatures must prove
these equalities even without calls.

`transpose_last(x)` copies and swaps only the final two axes for rank >= 2,
keeping all leading axes; its gradient reverses the swap. Both operations work
with CPU/tiled backends and forward SSA. `softmax` already normalizes the last
axis, so functions can compose these operations into attention. See
[attention semantics, toy training and validation](attention.md).

## Causal softmax

`causal_softmax(x)` preserves `Tensor[...,T,T]` shape (rank >= 2).
Row `i` normalizes columns `0..i`, including the diagonal; columns `j > i`
are exactly zero and have zero VJP. Each leading-axis matrix has its own
mask. Symbolic final dimensions must be identical, including in unused
functions. See [causal attention](causal-attention.md).

## Layer normalization

`layer_norm(x)` accepts rank >= 1, preserves shape, and normalizes each
last-axis row using population variance with fixed epsilon `1e-5`.
Use `layer_norm(x) * gain + bias` for learned affine normalization;
`gain` and `bias` can be `Tensor[D]` parameters. See
[layer normalization semantics and validation](layer-norm.md).

## Axis swapping and multiple batch axes

`swap_axes(x,first,second)` swaps two zero-based constant nonnegative axes,
preserving element count. It copies into row-major output storage and has
an inverse-permutation VJP. Equal axes are an identity copy; out-of-range
axes are compile errors. Symbolic dimensions follow the swapped positions.

`bmm` supports one or more batch axes. A batched RHS must have exactly the
same batch prefix, including symbolic names; partial batch broadcasting is
rejected. A rank-2 RHS is shared across every batch coordinate and its VJP
sums all contributions. See [multihead attention](multihead-attention.md).

## GELU

`gelu(x)` preserves any Tensor shape, including scalar shape. It uses the
standard tanh approximation
`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`, not exact erf GELU.
Its VJP differentiates this approximation. Saturated tails outside `[-10,10]`
return `x`/derivative 1 for positive input and zero/derivative 0 for negative
input, avoiding cubic overflow for finite values. See the
[Transformer guide](transformer.md) for a working composition.
