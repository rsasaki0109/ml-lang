// Train with raw logits and numerically stable binary cross entropy.
fn logits(x: Tensor[4,2], w1: Tensor[2,8], b1: Tensor[1,8],
          w2: Tensor[8,1], b2: Tensor[1,1]) -> Tensor[4,1] {
    let hidden: Tensor[4,8] = tanh(x @ w1 + b1);
    let output = hidden @ w2 + b2;
    return output;
}
let x = [[-1,-1],[-1,1],[1,-1],[1,1]];
let y = [[0],[1],[1],[0]];
param w1 = rand(2,8,42);
param b1 = zeros(1,8);
param w2 = rand(8,1,123);
param b2 = zeros(1,1);
print(bce_logits(logits(x,w1,b1,w2,b2),y));
repeat 2000 {
    let loss = bce_logits(logits(x,w1,b1,w2,b2),y);
    backward(loss);
    step(0.1);
}
let z = logits(x,w1,b1,w2,b2);
print(bce_logits(z,y));
let probabilities = sigmoid(z);
print(probabilities);
assert_close(probabilities,y,0.02);
