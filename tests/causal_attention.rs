use ml_lang::{
    backend::CpuBackend,
    compile, run,
    tensor::{forward, vjp, Op, Tensor},
    tiled::TiledCpuBackend,
};
#[test]
fn causal_probabilities_are_stable_normalized_and_exactly_masked() {
    let x = Tensor::new(
        vec![2, 3, 3],
        vec![
            1000., -1000., 1e300, 1000., 1000., 1e300, 0., 0., 0., -1000., 1e300, 1e300, 0., 0.,
            1e300, 1000., 999., 998.,
        ],
    )
    .unwrap();
    let y = forward(Op::CausalSoftmax, &x, None).unwrap();
    for (i, row) in y.data.chunks_exact(3).enumerate() {
        assert!((row.iter().sum::<f64>() - 1.).abs() < 1e-12);
        assert!(row[i % 3 + 1..].iter().all(|x| *x == 0.));
    }
    assert_eq!(y.data[0], 1.);
    assert_eq!(&y.data[3..6], &[0.5, 0.5, 0.]);
}
#[test]
fn every_causal_gradient_matches_finite_difference() {
    for shape in [vec![3, 3], vec![2, 3, 3], vec![2, 2, 3, 3]] {
        let x = Tensor::new(
            shape.clone(),
            (0..shape.iter().product())
                .map(|i| i as f64 / 13. - 1.)
                .collect(),
        )
        .unwrap();
        let y = forward(Op::CausalSoftmax, &x, None).unwrap();
        let g: Vec<_> = (0..x.data.len())
            .map(|i| (i % 7) as f64 / 5. - 0.3)
            .collect();
        let (gradient, _) = vjp(Op::CausalSoftmax, &x, None, &y, &g);
        for (i, &analytic) in gradient.iter().enumerate() {
            let loss = |d| {
                let mut v = x.clone();
                v.data[i] += d;
                forward(Op::CausalSoftmax, &v, None)
                    .unwrap()
                    .data
                    .iter()
                    .zip(&g)
                    .map(|(p, g)| p * g)
                    .sum::<f64>()
            };
            assert!((analytic - (loss(1e-6) - loss(-1e-6)) / 2e-6).abs() < 2e-9);
            if i % 3 > i / 3 % 3 {
                assert_eq!(analytic, 0.);
            }
        }
    }
}
#[test]
fn invalid_and_symbolically_unproven_shapes_are_rejected() {
    for source in [
        "let y=causal_softmax(1);",
        "let y=causal_softmax([1,2]);",
        "let y=causal_softmax([[1,2]]);",
        "let y=causal_softmax([[1]],[[1]]);",
        "fn f(x:Tensor[B,T,S])->Tensor[B,T,S]{return causal_softmax(x);}",
    ] {
        assert!(compile(source).is_err(), "{source}");
    }
    run("fn f(x:Tensor[B,T,T])->Tensor[B,T,T]{return causal_softmax(x);}let y=f([[[1,2],[3,4]]]);")
        .unwrap();
}
#[test]
fn future_tokens_cannot_change_past_outputs_or_receive_past_gradients() {
    let f = include_str!("../examples/causal_attention.ml")
        .split("let x =")
        .next()
        .unwrap();
    let source = |future| {
        format!("{f}param x=[[[1,0],[0,1],[{future},-7]]];let w=[[1,0],[0,1]];let y=attend(x,w,w,w,0.5);let prefix=rows(reshape(y,3,2),0,2);backward(sum(prefix));let g=grad(x);")
    };
    let a = run(&source(2)).unwrap();
    let b = run(&source(900)).unwrap();
    assert_eq!(a.variables["prefix"], b.variables["prefix"]);
    assert_eq!(&a.variables["g"].data[4..], &[0., 0.]);
    assert_eq!(&b.variables["g"].data[4..], &[0., 0.]);
}
#[test]
fn causal_training_resume_and_both_inference_backends_agree() {
    let source = include_str!("../examples/causal_attention.ml");
    let full = run(source).unwrap();
    assert!(full.output[1].data[0] < 0.001);
    let first = run(&source.replace("repeat 500", "repeat 213")).unwrap();
    let saved =
        ml_lang::checkpoint::Checkpoint::decode(&first.checkpoint.encode().unwrap()).unwrap();
    let resumed = ml_lang::vm::execute_with_options(
        &compile(&source.replace("repeat 500", "repeat 287")).unwrap(),
        &TiledCpuBackend,
        Some(&saved),
        false,
    )
    .unwrap();
    assert_eq!(full.checkpoint, resumed.checkpoint);
    let f = source.split("let x =").next().unwrap();
    let p=compile(&format!("{f}param wq=zeros(2,4);param wk=zeros(2,4);param wv=zeros(2,2);let x=[[[1,0],[0,1],[0,1]]];print(attend(x,wq,wk,wv,0.5));")).unwrap();
    let reference =
        ml_lang::vm::execute_with_options(&p, &CpuBackend, Some(&full.checkpoint), true).unwrap();
    let ir = ml_lang::ir::lower(&p).unwrap();
    for backend in [
        &CpuBackend as &dyn ml_lang::backend::Backend,
        &TiledCpuBackend,
    ] {
        let actual =
            ml_lang::inference::execute_with_backend(&ir, Some(&full.checkpoint), backend).unwrap();
        assert_eq!(reference.output, actual.output);
        assert_eq!(actual.output[0], full.output[3]);
    }
}
