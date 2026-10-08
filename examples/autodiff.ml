param x: Tensor[3] = [1,2,3];
let loss = sum(x*x);
backward(loss);
print(grad(x));
assert_close(grad(x),[2,4,6],1e-12);
