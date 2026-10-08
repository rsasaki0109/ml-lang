// Six temporary activation buffers; SSA last-use execution releases them early.
// An intentionally simple synthetic memory workload, not a production model.
param x = ones(256,256);
let result = mean(tanh(tanh(tanh(tanh(tanh(tanh(x)))))));
print(result);
assert_close(result,0.4397976612426585,1e-10);
