// Two heads, each width 2; explicit dimensions avoid implicit shape arithmetic.
fn attend(x: Tensor[B,T,2], wq: Tensor[2,4], wk: Tensor[2,4],
          wv: Tensor[2,4], wo: Tensor[4,2]) -> Tensor[B,T,2] {
    // Separate projection columns for each head, shared across batches.
    let query = swap_axes(reshape(bmm(x,wq),B,T,2,2),1,2);
    let key = swap_axes(reshape(bmm(x,wk),B,T,2,2),1,2);
    let value = swap_axes(reshape(bmm(x,wv),B,T,2,2),1,2);
    let weights = causal_softmax(bmm(query,transpose_last(key)) * 0.7071067811865476);
    let context = bmm(weights,value);
    let merged = reshape(swap_axes(context,1,2),B,T,4);
    return bmm(merged,wo);
}
let x = [[[1,0],[0,1],[1,0]],[[0,1],[1,0],[0,1]]];
param wq = rand(2,4,42);
param wk = rand(2,4,123);
param wv = rand(2,4,99);
param wo = rand(4,2,17);
print(mse(attend(x,wq,wk,wv,wo),x));
repeat 500 {
    let loss = mse(attend(x,wq,wk,wv,wo),x);
    backward(loss);
    clip_grad_norm(1);
    adam_step(0.03);
}
let prediction = attend(x,wq,wk,wv,wo);
print(mse(prediction,x));
assert_close(prediction,x,0.03);
let query = [[[1,0],[0,1],[0,1]]];
let result = attend(query,wq,wk,wv,wo);
print(result);
assert_close(result,query,0.03);
