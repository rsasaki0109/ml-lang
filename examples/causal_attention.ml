// Causal single-head scaled dot-product attention. Shared weights across sequences.
fn attend(x: Tensor[B,T,D], wq: Tensor[D,H], wk: Tensor[D,H],
          wv: Tensor[D,V], scale: Tensor[]) -> Tensor[B,T,V] {
    let query = bmm(x,wq);
    let key = bmm(x,wk);
    let value = bmm(x,wv);
    let scores = bmm(query,transpose_last(key)) * scale;
    let weights = causal_softmax(scores);
    return bmm(weights,value);
}
let x = [[[1,0],[0,1],[1,0]],[[0,1],[1,0],[0,1]]];
param wq = rand(2,4,42);
param wk = rand(2,4,123);
param wv = rand(2,2,99);
// Head width H=4: use 1/sqrt(H)=0.5. Dimensions are not scalar expressions.
print(mse(attend(x,wq,wk,wv,0.5),x));
repeat 500 {
    let loss = mse(attend(x,wq,wk,wv,0.5),x);
    backward(loss);
    clip_grad_norm(1);
    adam_step(0.03);
}
let prediction = attend(x,wq,wk,wv,0.5);
print(mse(prediction,x));
print(prediction);
assert_close(prediction,x,0.03);
let query = [[[1,0],[0,1],[0,1]]];
let result = attend(query,wq,wk,wv,0.5);
print(result);
assert_close(result,query,0.03);
