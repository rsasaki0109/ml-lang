#!/bin/sh
# Core development environment; benchmarks are optional, see docs/benchmarks.md.
set -eu
cd /workspace/ml-lang
export CARGO_HOME=/workspace/.ml-lang-tools/cargo
export RUSTUP_HOME=/workspace/.ml-lang-tools/rustup
if [ "$(uname -m)" != x86_64 ]; then
    echo 'This cloud setup script is validated for Linux x86-64 only.' >&2
    exit 1
fi
if [ ! -x "$CARGO_HOME/bin/rustup" ]; then
    installer=$(mktemp /tmp/ml-lang-rustup.XXXXXX)
    trap 'rm -f "$installer"' EXIT HUP INT TERM
    curl --fail --location --proto '=https' --tlsv1.2 https://sh.rustup.rs -o "$installer"
    sh "$installer" -y --profile minimal --default-toolchain 1.85.1 --no-modify-path
fi
if [ ! -x "$RUSTUP_HOME/toolchains/1.85.1-x86_64-unknown-linux-gnu/bin/rustfmt" ] || [ ! -x "$RUSTUP_HOME/toolchains/1.85.1-x86_64-unknown-linux-gnu/bin/clippy-driver" ]; then
    "$CARGO_HOME/bin/rustup" toolchain install 1.85.1 --profile minimal --component rustfmt --component clippy
fi
export PATH="$RUSTUP_HOME/toolchains/1.85.1-x86_64-unknown-linux-gnu/bin:$CARGO_HOME/bin:$PATH"
cargo build --release --locked
cargo test --locked
scripts/smoke.sh
