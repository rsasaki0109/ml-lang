// Stop gradients on a selected path while preserving its current values.
param x = [2,3];
let fixed = detach(x);
let loss = sum(x * fixed);
backward(loss);
assert_close(grad(x),[2,3],0);
step(0.1);
assert_close(fixed,[2,3],0);
assert_close(x,[1.8,2.7],1e-12);
print(x);
