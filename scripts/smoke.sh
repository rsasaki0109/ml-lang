#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
for file in examples/*.ml; do
    target/release/ml check "$file"
    case "$file" in
        examples/adam_infer.ml|examples/adam_multiclass.ml|examples/minibatch.ml|examples/attention.ml|examples/attention_infer.ml|examples/transformer.ml|examples/transformer_infer.ml) continue ;;
    esac
    target/release/ml run "$file"
done
checkpoint_dir=$(mktemp -d /tmp/ml-lang-smoke.XXXXXX)
trap 'rm -rf "$checkpoint_dir"' EXIT HUP INT TERM
target/release/ml run examples/adam_multiclass.ml --save "$checkpoint_dir/model.mlckpt"
target/release/ml run examples/adam_infer.ml --load "$checkpoint_dir/model.mlckpt" --inference
target/release/ml run examples/minibatch.ml --save "$checkpoint_dir/minibatch.mlckpt"
target/release/ml run examples/adam_infer.ml --load "$checkpoint_dir/minibatch.mlckpt" --inference
target/release/ml run examples/adam_infer.ml --load "$checkpoint_dir/minibatch.mlckpt" --inference --optimize
target/release/ml run examples/inference_memory.ml --inference --optimize
target/release/ml plan examples/inference_memory.ml > "$checkpoint_dir/inference-plan.txt"
target/release/ml run examples/tiled_matmul.ml --backend tiled
target/release/ml run examples/tiled_matmul.ml --inference --optimize --backend tiled
target/release/ml run examples/adam_infer.ml --load "$checkpoint_dir/minibatch.mlckpt" --inference --optimize --backend tiled
target/release/ml run examples/attention.ml --save "$checkpoint_dir/attention.mlckpt"
target/release/ml run examples/attention_infer.ml --load "$checkpoint_dir/attention.mlckpt" --inference
target/release/ml run examples/attention_infer.ml --load "$checkpoint_dir/attention.mlckpt" --inference --optimize --backend tiled
target/release/ml run examples/transformer.ml --save "$checkpoint_dir/transformer.mlckpt"
target/release/ml run examples/transformer_infer.ml --load "$checkpoint_dir/transformer.mlckpt" --inference
target/release/ml run examples/transformer_infer.ml --load "$checkpoint_dir/transformer.mlckpt" --inference --optimize --backend tiled
