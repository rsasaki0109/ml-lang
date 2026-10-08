// Requires a checkpoint from attention.ml.
fn attend(x: Tensor[B,T,D], wq: Tensor[D,H], wk: Tensor[D,H],
          wv: Tensor[D,V], scale: Tensor[]) -> Tensor[B,T,V] {
    let q = bmm(x,wq);
    let k = bmm(x,wk);
    let v = bmm(x,wv);
    return bmm(softmax(bmm(q,transpose_last(k))*scale),v);
}
param wq = zeros(2,4);
param wk = zeros(2,4);
param wv = zeros(2,2);
let query = [[[1,0],[0,1],[0,1]]];
let result = attend(query,wq,wk,wv,0.5);
print(result);
assert_close(result,query,0.03);
