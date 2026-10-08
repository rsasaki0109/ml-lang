param weights: Tensor[2,1] = [[0.1],[-0.2]];
let inputs: Tensor[4,2] = [[0,0],[0,1],[1,0],[1,1]];
let targets: Tensor[4,1] = [[0],[1],[1],[2]];
repeat 500 {
    let loss = mse(inputs @ weights, targets);
    backward(loss);
    step(0.1);
}
let prediction = inputs @ weights;
print(prediction);
assert_close(prediction,targets,1e-5);
assert_close(weights,[[1],[1]],1e-5);
