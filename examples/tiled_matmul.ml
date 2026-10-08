// A rectangular product crossing 32-element tile boundaries.
// Run: ml run examples/tiled_matmul.ml --backend tiled
let a = ones(33,35);
let b = ones(35,37);
let c = a @ b;
assert_close(c,ones(33,37)*35,0);
print(sum(c));
