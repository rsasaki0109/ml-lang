let a: Tensor[2,3] = [[1,2,3],[4,5,6]];
let b: Tensor[3,2] = [[1,2],[3,4],[5,6]];
let result: Tensor[2,2] = a @ b;
print(result);
assert_close(result,[[22,28],[49,64]],1e-12);
