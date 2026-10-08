# CPU axis-swap kernel ablation

`swap_axes` previously allocated a coordinate vector for every element in
forward and VJP. It now computes a stride mapping once, then copies each
contiguous suffix block. Equal axes use one contiguous copy. The VJP uses
the same kernel for the inverse swap. Output storage is still owned dense
row-major memory; this does not introduce views, fusion or buffer reuse.

## Method

```sh
python3 benchmarks/axis_swap.py --samples 5 --output /tmp/axis-swap.json
```

The stdlib-only Python controller compiles a standalone Rust harness with
Rust 1.85.1 at opt-level 3. The harness imports the current Tensor kernel
and retains the historical coordinate-allocation permutation algorithm as
`legacy`, including zeroed result allocation and forward finite-value
validation. This is an isolated kernel ablation, not two whole CLI/compiler
versions or a comparison with PyTorch/JAX. Shape-dispatch bookkeeping is
not identical between the paths.

Three shapes/layouts contain 131,072 f64 elements each: head split/merge
`[8,64,8,32]`, axes 1 and 2; last-axis swap `[8,8,64,32]`, axes 2 and 3;
and identity `[8,8,64,32]`, axis 1 with itself. The first case copies
32-element blocks; the last-axis case copies one element at a time.

Each sample is a fresh process pinned to one inherited CPU, performs one
warmup and times 20 iterations with `black_box`. Five samples produce a
median. Compilation is separate. All output or gradient elements are
verified after timing. Raw records include every sample, host/toolchain
metadata, shapes, axes and source/harness/controller/executable SHA-256.

Peak memory is Linux process `VmHWM`, including input, upstream gradients,
a zero-filled output-shape fixture, result storage, validation and allocator
state. Zero pages can remain uncommitted. The metric is host RSS, not GPU
memory or exact live Tensor bytes. The fixture/setup are shared between
paths. Timing and RSS reflect this harness, not end-to-end model training.


## Measured results

Raw report: [axis-swap-cpu.json](../benchmarks/axis-swap-cpu.json).

| Case | Legacy ms | Blocks ms | Speed ratio | Legacy / blocks RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| heads forward | 3.001 | 0.101 | 29.8x | 4.25 / 4.25 |
| heads vjp | 2.759 | 0.054 | 51.4x | 4.19 / 4.26 |
| last forward | 3.013 | 1.006 | 3.0x | 4.27 / 4.25 |
| last vjp | 2.994 | 1.029 | 2.9x | 4.25 / 4.18 |
| identity forward | 2.987 | 0.101 | 29.5x | 4.25 / 4.25 |
| identity vjp | 2.842 | 0.046 | 61.7x | 4.27 / 4.27 |

Peak RSS differences are small and mixed; no memory reduction is established.

## Correctness and limits

Independent input-coordinate scatter tests check all axis pairs for ranks
1 through 9, including singleton axes, identity swaps, inverse swaps and
every VJP element. Rank 9 exercises the public Tensor API; source-language
reshape still has its rank limit. Existing finite differences, checkpoint
resume, CPU/tiled loaded SSA and PyTorch/JAX multihead comparisons pass.
A training test also checks retained Tensor payload for 3 versus 3,000
multihead optimizer steps without prints.

These measurements support a local CPU permutation speed improvement. They
do not establish a faster Transformer, framework superiority, GPU speed,
or a peak-memory reduction. Last-axis swaps retain elementwise indexing,
and every non-view output still needs its full buffer. Shared cloud noise
limits precision and generalization across shapes and machines.

The retained permutation report captures the implementation before GELU and
the Transformer example were added. Its hashes identify that measured source;
it does not measure the current full language executable or Transformer training.
