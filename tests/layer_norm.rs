use ml_lang::{
    backend::{Backend, CpuBackend},
    compile, run,
    tensor::{forward, vjp, Op, Tensor},
    tiled::TiledCpuBackend,
};
fn close(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= eps, "{a} != {b}");
    }
}
#[test]
fn values_use_last_axis_population_variance_and_fixed_epsilon() {
    let x = Tensor::new(
        vec![2, 2, 3],
        vec![1., 2., 3., 4., 4., 4., -2., 0., 2., 0., 0., 0.],
    )
    .unwrap();
    let y = forward(Op::LayerNorm, &x, None).unwrap();
    let d = (2. / 3. + 1e-5_f64).sqrt();
    let e = (8. / 3. + 1e-5_f64).sqrt();
    close(
        &y.data,
        &[
            -1. / d,
            0.,
            1. / d,
            0.,
            0.,
            0.,
            -2. / e,
            0.,
            2. / e,
            0.,
            0.,
            0.,
        ],
        1e-14,
    );
    assert_eq!(y.shape, x.shape);
}
#[test]
fn extreme_finite_and_constant_rows_do_not_overflow() {
    for width in [1, 3, 6, 7] {
        for value in [0., 1e-300, 1e308, -1e308] {
            let x = Tensor::new(vec![width], vec![value; width]).unwrap();
            let y = forward(Op::LayerNorm, &x, None).unwrap();
            assert_eq!(y.data, vec![0.; width]);
            let g: Vec<_> = (0..width).map(|i| i as f64).collect();
            let (actual, _) = vjp(Op::LayerNorm, &x, None, &y, &g);
            let mean = (width - 1) as f64 / 2.;
            let expected: Vec<_> = g.iter().map(|g| (g - mean) / 1e-5_f64.sqrt()).collect();
            close(&actual, &expected, 1e-10);
        }
    }
    let x = Tensor::new(vec![3], vec![1e308, -1e308, 0.]).unwrap();
    let y = forward(Op::LayerNorm, &x, None).unwrap();
    close(&y.data, &[(1.5_f64).sqrt(), -(1.5_f64).sqrt(), 0.], 1e-14);
}
#[test]
fn every_input_vjp_matches_central_differences() {
    for shape in [vec![1], vec![3], vec![2, 3], vec![2, 2, 3]] {
        for constant in [false, true] {
            let x = Tensor::new(
                shape.clone(),
                (0..shape.iter().product())
                    .map(|i| {
                        if constant {
                            0.7
                        } else {
                            (i % 5) as f64 / 7. - 0.3
                        }
                    })
                    .collect(),
            )
            .unwrap();
            let y = forward(Op::LayerNorm, &x, None).unwrap();
            let g: Vec<_> = (0..x.data.len())
                .map(|i| (i % 7) as f64 / 11. - 0.2)
                .collect();
            let (grad, _) = vjp(Op::LayerNorm, &x, None, &y, &g);
            for (i, &actual) in grad.iter().enumerate() {
                let loss = |d| {
                    let mut a = x.clone();
                    a.data[i] += d;
                    forward(Op::LayerNorm, &a, None)
                        .unwrap()
                        .data
                        .iter()
                        .zip(&g)
                        .map(|(y, g)| y * g)
                        .sum::<f64>()
                };
                assert!(
                    (actual - (loss(1e-7) - loss(-1e-7)) / 2e-7).abs() < 2e-7,
                    "{shape:?} {constant} {i}"
                );
            }
        }
    }
}
#[test]
fn shape_and_arity_checks_include_unused_symbolic_functions() {
    for s in [
        "let y=layer_norm(1);",
        "let y=layer_norm();",
        "let y=layer_norm([1],[2]);",
        "fn f(x:Tensor[])->Tensor[]{return layer_norm(x);}",
        "fn f(x:Tensor[B,D])->Tensor[D,B]{return layer_norm(x);}",
    ] {
        assert!(compile(s).is_err(), "accepted {s}");
    }
    run("fn f(x:Tensor[B,T,D])->Tensor[B,T,D]{return layer_norm(x);}let y=f([[[1,2,3]]]);")
        .unwrap();
}
#[test]
fn affine_training_resume_and_loaded_ssa_match() {
    let s = include_str!("../examples/layer_norm.ml");
    let full = run(s).unwrap();
    assert!(full.output[1].data[0] < 1e-12);
    let first = run(&s.replace("repeat 300", "repeat 117")).unwrap();
    let saved =
        ml_lang::checkpoint::Checkpoint::decode(&first.checkpoint.encode().unwrap()).unwrap();
    let resumed = ml_lang::vm::execute_with_options(
        &compile(&s.replace("repeat 300", "repeat 183")).unwrap(),
        &TiledCpuBackend,
        Some(&saved),
        false,
    )
    .unwrap();
    assert_eq!(full.checkpoint, resumed.checkpoint);
    let f = s.split("let x =").next().unwrap();
    let p=compile(&format!("{f}param gain=ones(3);param bias=zeros(3);let query=[[[8,1,3],[1,1,1]]];print(model(query,gain,bias));")).unwrap();
    let ir = ml_lang::ir::lower(&p).unwrap();
    let reference =
        ml_lang::vm::execute_with_options(&p, &CpuBackend, Some(&full.checkpoint), true).unwrap();
    for backend in [&CpuBackend as &dyn Backend, &TiledCpuBackend] {
        let actual =
            ml_lang::inference::execute_with_backend(&ir, Some(&full.checkpoint), backend).unwrap();
        assert_eq!(actual.output, reference.output);
        assert_eq!(actual.output[0], full.output[2]);
    }
}
#[test]
fn residual_causal_attention_composed_gradient_is_correct() {
    let f = include_str!("../examples/causal_attention.ml")
        .split("let x =")
        .next()
        .unwrap();
    let source = |d| {
        format!("{f}let x=[[[1,0],[0,1]]];param w=[[{},0.2],[-0.3,0.4]];let y=layer_norm(x+attend(x,w,w,w,0.5));let loss=sum(y*[[[0.1,0.7],[0.4,-0.2]]]);backward(loss);let g=grad(w);",0.1+d)
    };
    let r = run(&source(0.)).unwrap();
    let analytic = r.variables["g"].data[0];
    let plus = run(&source(1e-5)).unwrap().variables["loss"].data[0];
    let minus = run(&source(-1e-5)).unwrap().variables["loss"].data[0];
    assert!((analytic - (plus - minus) / 2e-5).abs() < 1e-8);
}
