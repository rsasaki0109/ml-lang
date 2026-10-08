// One model: B=2 minibatches, B=6 full evaluation, B=3 toy queries.
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
// Each epoch visits three contiguous batches of two samples.
repeat 100 {
    let xb0 = rows(x,0,2);
    let yb0 = rows(y,0,2);
    let loss0 = cross_entropy(logits(xb0,w1,b1,w2,b2),yb0);
    backward(loss0);
    clip_grad_norm(1);
    adam_step(0.03);
    let xb2 = rows(x,2,2);
    let yb2 = rows(y,2,2);
    let loss2 = cross_entropy(logits(xb2,w1,b1,w2,b2),yb2);
    backward(loss2);
    clip_grad_norm(1);
    adam_step(0.03);
    let xb4 = rows(x,4,2);
    let yb4 = rows(y,4,2);
    let loss4 = cross_entropy(logits(xb4,w1,b1,w2,b2),yb4);
    backward(loss4);
    clip_grad_norm(1);
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
