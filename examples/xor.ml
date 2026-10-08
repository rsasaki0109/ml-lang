// Every tensor is statically shaped, f64, and on CPU in v0.1.
fn predict(x: Tensor[4,2], w1: Tensor[2,8], b1: Tensor[1,8],
           w2: Tensor[8,1], b2: Tensor[1,1]) -> Tensor[4,1] {
    let hidden = tanh(x @ w1 + b1);
    let output = hidden @ w2 + b2;
    return tanh(output);
}
let x: Tensor[4,2] = [[-1,-1],[-1,1],[1,-1],[1,1]];
let y: Tensor[4,1] = [[-1],[1],[1],[-1]];
param w1: Tensor[2,8] = rand(2,8,42);
param b1: Tensor[1,8] = zeros(1,8);
param w2: Tensor[8,1] = rand(8,1,123);
param b2: Tensor[1,1] = zeros(1,1);
print(mse(predict(x,w1,b1,w2,b2),y));
repeat 2000 {
    let loss = mse(predict(x,w1,b1,w2,b2),y);
    backward(loss);
    step(0.1);
}
let prediction = predict(x,w1,b1,w2,b2);
print(mse(prediction,y));
print(prediction);
assert_close(prediction,y,0.06);
