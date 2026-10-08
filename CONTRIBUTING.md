# Contributing

Use the pinned Rust toolchain. The compiler/runtime have no third-party crate
dependencies. Keep numerical primitives and their VJPs documented together.

```sh
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo test --release --locked
cargo build --release --locked
scripts/smoke.sh
```

For language changes, update the specification and add accepted/rejected source
tests. For kernel changes, add finite-difference and reference-value checks,
including broadcasts and repeated consumers. Do not disable checks to make
benchmarks pass. See `docs/benchmarks.md` for baseline reproducibility.

Propose performance changes with the workload, dtype/device, tool versions,
thread configuration, compile/warmup policy, numerical checks and measured
peak memory. Keep implemented behavior distinct from roadmap proposals.

Optimizer changes must preserve finite updates, disconnected-parameter semantics
and exact saved-state continuation. Persistence changes require malformed-file,
no-overwrite and separate-process train/load tests.

For IR changes, compare outputs and complete final bindings with reference VM
inference, including captures, nested/zero loops, aliases and runtime errors from
unused operations. Memory optimizations must preserve checkpoint loading and
buffered output semantics. See docs/inference-ir.md for the release-plan contract.

Alternative kernels must match reference forward and VJP values for full and
partial tiles, include independent finite differences, preserve state restoration
and document scratch allocation. Report per-backend results instead of attributing
all numerical execution to the default kernel.

Batched kernels must test shared-gradient reduction, per-batch weights, symbolic
batch mismatch and irregular tile boundaries. Attention examples must state masks,
head counts and scale, and distinguish toy reconstruction from dataset accuracy.
