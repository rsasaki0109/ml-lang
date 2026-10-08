use ml_lang::{
    run,
    tensor::{forward, vjp, Op, Tensor},
};
// Independent input-coordinate scatter oracle (the implementation gathers blocks).
fn destination(mut flat: usize, shape: &[usize], first: usize, second: usize) -> usize {
    let mut coordinates = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coordinates[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    coordinates.swap(first, second);
    let mut out = shape.to_vec();
    out.swap(first, second);
    coordinates
        .iter()
        .zip(out)
        .fold(0, |index, (coordinate, dim)| index * dim + coordinate)
}
#[test]
fn all_axis_pairs_ranks_one_through_nine_match_independent_scatter() {
    for rank in 1..=9 {
        let shape: Vec<_> = (0..rank)
            .map(|i| {
                if i % 3 == 1 {
                    1
                } else if i % 3 == 2 {
                    3
                } else {
                    2
                }
            })
            .collect();
        let x = Tensor::new(
            shape.clone(),
            (0..shape.iter().product())
                .map(|i| i as f64 - 13.)
                .collect(),
        )
        .unwrap();
        for first in 0..rank {
            for second in 0..rank {
                let op = Op::SwapAxes { first, second };
                let y = forward(op, &x, None).unwrap();
                let g: Vec<_> = (0..x.data.len()).map(|i| i as f64 / 7. - 0.2).collect();
                let (grad, _) = vjp(op, &x, None, &y, &g);
                for (i, &value) in x.data.iter().enumerate() {
                    let j = destination(i, &shape, first, second);
                    assert_eq!(y.data[j], value);
                    assert_eq!(grad[i], g[j]);
                }
                assert_eq!(forward(op, &y, None).unwrap(), x);
            }
        }
    }
}
#[test]
fn multihead_training_payload_does_not_grow_with_step_count() {
    let s = include_str!("../examples/multihead_attention.ml");
    let f = s.split("let x =").next().unwrap();
    let source = |steps| {
        format!("{f}let x=[[[1,0],[0,1]],[[0,1],[1,0]]];param wq=rand(2,4,42);param wk=rand(2,4,123);param wv=rand(2,4,99);param wo=rand(4,2,17);repeat {steps}{{let loss=mse(attend(x,wq,wk,wv,wo),x);backward(loss);clip_grad_norm(1);adam_step(0.03);}}")
    };
    assert_eq!(
        run(&source(3)).unwrap().peak_tensor_bytes,
        run(&source(3000)).unwrap().peak_tensor_bytes
    );
}
