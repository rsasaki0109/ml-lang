// One shape-polymorphic model used for training (B=6) and inference (B=3).
fn logits(x: Tensor[B,2], w1: Tensor[2,8], b1: Tensor[1,8],
          w2: Tensor[8,3], b2: Tensor[1,3]) -> Tensor[B,3] {
    let hidden: Tensor[B,8] = tanh(x @ w1 + b1);
    let output: Tensor[B,3] = hidden @ w2 + b2;
    return output;
}
let x = [[-1,-1],[-0.8,-1.2],[1,-1],[1.2,-0.8],[0,1],[0.2,1.2]];
let y = [[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]];
param w1 = rand(2,8,42);
param b1 = zeros(1,8);
param w2 = rand(8,3,123);
param b2 = zeros(1,3);
print(cross_entropy(logits(x,w1,b1,w2,b2),y));
repeat 300 {
    let loss = cross_entropy(logits(x,w1,b1,w2,b2),y);
    backward(loss);
    adam_step(0.03);
}
let z = logits(x,w1,b1,w2,b2);
print(cross_entropy(z,y));
print(softmax(z));
assert_close(softmax(z),y,0.02);
// Separate unseen toy points, with a different statically specialized batch size.
let query = [[-1.1,-0.9],[0.9,-1.1],[0.1,0.9]];
let prediction = softmax(logits(query,w1,b1,w2,b2));
print(prediction);
assert_close(prediction,[[1,0,0],[0,1,0],[0,0,1]],0.03);
