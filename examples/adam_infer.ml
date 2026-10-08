// Run with: ml run examples/adam_infer.ml --load /tmp/classifier.mlckpt --inference
fn logits(x: Tensor[B,2], w1: Tensor[2,8], b1: Tensor[1,8],
          w2: Tensor[8,3], b2: Tensor[1,3]) -> Tensor[B,3] {
    let hidden = tanh(x @ w1 + b1);
    return hidden @ w2 + b2;
}
// The CLI restores these named parameters before their first use.
param w1 = zeros(2,8);
param b1 = zeros(1,8);
param w2 = zeros(8,3);
param b2 = zeros(1,3);
let query = [[-1.1,-0.9],[0.9,-1.1],[0.1,0.9]];
let probabilities = softmax(logits(query,w1,b1,w2,b2));
print(probabilities);
// This example deliberately fails without the trained checkpoint.
assert_close(probabilities,[[1,0,0],[0,1,0],[0,0,1]],0.03);
