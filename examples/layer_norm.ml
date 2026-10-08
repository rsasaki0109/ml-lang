// Layer normalization followed by a learned affine transform and activation.
fn model(x: Tensor[B,T,D], gain: Tensor[D], bias: Tensor[D]) -> Tensor[B,T,D] {
    return tanh(layer_norm(x) * gain + bias);
}
let x = [[[1,2,4],[4,1,2]],[[2,5,1],[3,2,6]]];
let target = model(x,[0.7,1.2,0.4],[0.2,-0.3,0.1]);
param gain = ones(3);
param bias = zeros(3);
print(mse(model(x,gain,bias),target));
repeat 300 {
    let loss = mse(model(x,gain,bias),target);
    backward(loss);
    clip_grad_norm(1);
    adam_step(0.03);
}
let prediction = model(x,gain,bias);
print(mse(prediction,target));
assert_close(prediction,target,0.001);
let query = [[[8,1,3],[1,1,1]]];
let result = model(query,gain,bias);
print(result);
assert_close(result,model(query,[0.7,1.2,0.4],[0.2,-0.3,0.1]),0.001);
