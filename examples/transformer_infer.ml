// One pre-norm causal Transformer block, two width-2 heads, width-8 FFN.
fn model(x: Tensor[B,T,4], wq: Tensor[4,4], wk: Tensor[4,4],
         wv: Tensor[4,4], wo: Tensor[4,4],
         gain1: Tensor[4], bias1: Tensor[4], gain2: Tensor[4], bias2: Tensor[4],
         w1: Tensor[4,8], b1: Tensor[8], w2: Tensor[8,4], b2: Tensor[4],
         readout: Tensor[4,2], readbias: Tensor[2]) -> Tensor[B,T,2] {
    let normalized = layer_norm(x) * gain1 + bias1;
    let q = swap_axes(reshape(bmm(normalized,wq),B,T,2,2),1,2);
    let k = swap_axes(reshape(bmm(normalized,wk),B,T,2,2),1,2);
    let v = swap_axes(reshape(bmm(normalized,wv),B,T,2,2),1,2);
    let scores = bmm(q,transpose_last(k)) * 0.7071067811865476;
    let context = bmm(causal_softmax(scores),v);
    let attention = bmm(reshape(swap_axes(context,1,2),B,T,4),wo);
    let residual = x + attention;
    let feed_input = layer_norm(residual) * gain2 + bias2;
    let hidden = gelu(bmm(feed_input,w1) + b1);
    let block = residual + bmm(hidden,w2) + b2;
    return bmm(block,readout) + readbias;
}
// Load a checkpoint produced by transformer.ml; initializers preserve shapes.
param wq = zeros(4,4);
param wk = zeros(4,4);
param wv = zeros(4,4);
param wo = zeros(4,4);
param gain1 = ones(4);
param bias1 = zeros(4);
param gain2 = ones(4);
param bias2 = zeros(4);
param w1 = zeros(4,8);
param b1 = zeros(8);
param w2 = zeros(8,4);
param b2 = zeros(4);
param readout = zeros(4,2);
param readbias = zeros(2);
let query = [[[1,0,1,0],[0,1,0,1],[1,0,1,1]]];
let result = softmax(model(query,wq,wk,wv,wo,gain1,bias1,gain2,bias2,w1,b1,w2,b2,readout,readbias));
print(result);
assert_close(result,[[[1,0],[0,1],[1,0]]],0.03);
