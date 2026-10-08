// Three clusters, two samples per class, softmax probability inference.
fn logits(x: Tensor[6,2], w1: Tensor[2,8], b1: Tensor[1,8],
          w2: Tensor[8,3], b2: Tensor[1,3]) -> Tensor[6,3] {
    let hidden: Tensor[6,8] = tanh(x @ w1 + b1);
    let output = hidden @ w2 + b2;
    return output;
}
let x = [[-1,-1],[-0.8,-1.2],[1,-1],[1.2,-0.8],[0,1],[0.2,1.2]];
let y = [[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]];
param w1 = rand(2,8,42);
param b1 = zeros(1,8);
param w2 = rand(8,3,123);
param b2 = zeros(1,3);
print(cross_entropy(logits(x,w1,b1,w2,b2),y));
repeat 1000 {
    let loss = cross_entropy(logits(x,w1,b1,w2,b2),y);
    backward(loss);
    step(0.1);
}
let z = logits(x,w1,b1,w2,b2);
print(cross_entropy(z,y));
let probabilities = softmax(z);
print(probabilities);
assert_close(probabilities,y,0.02);
