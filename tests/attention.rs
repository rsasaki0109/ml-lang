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
fn batched_values_and_shared_weight_gradient_are_correct() {
    let r=run("param a=[[[1,2],[3,4]],[[5,6],[7,8]]];param b=[[1,2,3],[4,5,6]];let y=bmm(a,b);backward(sum(y));let ga=grad(a);let gb=grad(b);let z=bmm(a,[[[1],[2]],[[3],[4]]]);").unwrap();
    assert_eq!(r.variables["y"].shape, vec![2, 2, 3]);
    close(
        &r.variables["y"].data,
        &[9., 12., 15., 19., 26., 33., 29., 40., 51., 39., 54., 69.],
        0.,
    );
    close(
        &r.variables["ga"].data,
        &[6., 15., 6., 15., 6., 15., 6., 15.],
        0.,
    );
    close(&r.variables["gb"].data, &[16., 16., 16., 20., 20., 20.], 0.);
    close(&r.variables["z"].data, &[5., 11., 39., 53.], 0.);
}
#[test]
fn batched_vjps_match_every_input_finite_difference() {
    for shared in [true, false] {
        let a = Tensor::new(
            vec![2, 2, 3],
            (0..12).map(|i| i as f64 / 10. - 0.5).collect(),
        )
        .unwrap();
        let b = Tensor::new(
            if shared { vec![3, 2] } else { vec![2, 3, 2] },
            (0..if shared { 6 } else { 12 })
                .map(|i| i as f64 / 13. - 0.2)
                .collect(),
        )
        .unwrap();
        let out = forward(Op::BatchMatmul, &a, Some(&b)).unwrap();
        let g = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let (ga, gb) = vjp(Op::BatchMatmul, &a, Some(&b), &out, &g);
        let gb = gb.unwrap();
        for left in [true, false] {
            let analytic = if left { &ga } else { &gb };
            for (index, value) in analytic.iter().enumerate() {
                let loss = |offset| {
                    let mut a = a.clone();
                    let mut b = b.clone();
                    if left {
                        a.data[index] += offset;
                    } else {
                        b.data[index] += offset;
                    }
                    forward(Op::BatchMatmul, &a, Some(&b))
                        .unwrap()
                        .data
                        .iter()
                        .zip(&g)
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                };
                close(&[*value], &[(loss(1e-6) - loss(-1e-6)) / 2e-6], 2e-9);
            }
        }
    }
}
#[test]
fn transpose_last_swaps_only_last_axes_and_has_inverse_vjp() {
    for shape in [vec![2, 3], vec![2, 2, 3], vec![2, 2, 2, 3]] {
        let a = Tensor::new(
            shape.clone(),
            (0..shape.iter().product()).map(|i| i as f64 - 4.).collect(),
        )
        .unwrap();
        let out = forward(Op::TransposeLast, &a, None).unwrap();
        assert_eq!(forward(Op::TransposeLast, &out, None).unwrap(), a);
        let g: Vec<_> = out.data.iter().map(|x| x / 11.).collect();
        let (ga, _) = vjp(Op::TransposeLast, &a, None, &out, &g);
        let upstream = g.clone();
        let expected = forward(
            Op::TransposeLast,
            &Tensor::new(out.shape.clone(), g).unwrap(),
            None,
        )
        .unwrap();
        close(&ga, &expected.data, 0.);
        for (index, &analytic) in ga.iter().enumerate() {
            let loss = |delta| {
                let mut input = a.clone();
                input.data[index] += delta;
                forward(Op::TransposeLast, &input, None)
                    .unwrap()
                    .data
                    .iter()
                    .zip(&upstream)
                    .map(|(x, g)| x * g)
                    .sum::<f64>()
            };
            close(&[analytic], &[(loss(1e-5) - loss(-1e-5)) / 2e-5], 1e-7);
        }
    }
    let r = run("let x=[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]];let y=transpose_last(x);").unwrap();
    close(
        &r.variables["y"].data,
        &[1., 4., 2., 5., 3., 6., 7., 10., 8., 11., 9., 12.],
        0.,
    );
}
#[test]
fn symbolic_batched_shapes_and_invalid_programs_are_checked() {
    run("fn f(a:Tensor[B,M,K],b:Tensor[K,N])->Tensor[B,M,N]{return bmm(a,b);}let y=f([[[1,2]]],[[3],[4]]);").unwrap();
    for s in [
        "let y=bmm([[1,2]],[[3],[4]]);",
        "let y=bmm([[[1,2]]],[[[3],[4]],[[5],[6]]]);",
        "let y=bmm([[[1,2]]],[[3]]);",
        "let y=transpose_last([1,2]);",
        "let y=bmm([[[1]]]);",
        "fn f(a:Tensor[B,M,K],b:Tensor[C,K,N])->Tensor[B,M,N]{return bmm(a,b);}",
        "fn f(a:Tensor[B,M,K],b:Tensor[K,N])->Tensor[B,N,M]{return bmm(a,b);}",
        "fn f(a:Tensor[B,M,K])->Tensor[B,M,K]{return transpose_last(a);}",
    ] {
        assert!(compile(s).is_err(), "accepted {s}");
    }
    assert!(forward(
        Op::BatchMatmul,
        &Tensor::new(vec![1, 1, 1], vec![1.]).unwrap(),
        None
    )
    .is_err());
}
#[test]
fn tiled_batched_forward_and_shared_gradient_preserve_reduction_order() {
    for shared in [true, false] {
        let a = Tensor::new(
            vec![2, 17, 35],
            (0..2 * 17 * 35)
                .map(|i| (i % 37) as f64 / 23. - 0.5)
                .collect(),
        )
        .unwrap();
        let shape = if shared {
            vec![35, 129]
        } else {
            vec![2, 35, 129]
        };
        let b = Tensor::new(
            shape.clone(),
            (0..shape.iter().product())
                .map(|i| (i % 31) as f64 / 29. - 0.5)
                .collect(),
        )
        .unwrap();
        let shape = vec![2, 17, 129];
        let reference = CpuBackend
            .forward(Op::BatchMatmul, &a, Some(&b), &shape)
            .unwrap();
        let actual = TiledCpuBackend
            .forward(Op::BatchMatmul, &a, Some(&b), &shape)
            .unwrap();
        assert_eq!(reference, actual);
        let g: Vec<_> = actual.data.iter().map(|x| x / 13.).collect();
        assert_eq!(
            CpuBackend.vjp(Op::BatchMatmul, &a, Some(&b), &reference, &g),
            TiledCpuBackend.vjp(Op::BatchMatmul, &a, Some(&b), &actual, &g)
        );
    }
}
#[test]
fn composed_attention_gradients_match_finite_difference() {
    let f = include_str!("../examples/attention.ml")
        .split("let x =")
        .next()
        .unwrap();
    for weight in ["wq", "wk", "wv"] {
        let source = |offset| {
            let matrix = |name| {
                if name == weight {
                    format!("[[{},0.2],[-0.3,0.4]]", 0.1 + offset)
                } else {
                    "[[0.1,0.2],[-0.3,0.4]]".into()
                }
            };
            format!("{f}let x=[[[1,0],[0,1]],[[0.5,0.3],[0.2,0.8]]];param wq={};param wk={};param wv={};let loss=mse(attend(x,wq,wk,wv,0.7),x);backward(loss);let g=grad({weight});",matrix("wq"),matrix("wk"),matrix("wv"))
        };
        let analytic = run(&source(0.)).unwrap().variables["g"].data[0];
        let plus = run(&source(1e-6)).unwrap().variables["loss"].data[0];
        let minus = run(&source(-1e-6)).unwrap().variables["loss"].data[0];
        close(&[analytic], &[(plus - minus) / 2e-6], 1e-8);
    }
}
#[test]
fn trained_attention_checkpoint_resume_and_ssa_inference_match() {
    let source = include_str!("../examples/attention.ml");
    let full = run(source).unwrap();
    assert!(full.output[1].data[0] < 0.001);
    let first = run(&source.replace("repeat 500", "repeat 211")).unwrap();
    let saved =
        ml_lang::checkpoint::Checkpoint::decode(&first.checkpoint.encode().unwrap()).unwrap();
    let resumed = ml_lang::vm::execute_with_options(
        &compile(&source.replace("repeat 500", "repeat 289")).unwrap(),
        &TiledCpuBackend,
        Some(&saved),
        false,
    )
    .unwrap();
    assert_eq!(full.checkpoint, resumed.checkpoint);
    let inference = compile(include_str!("../examples/attention_infer.ml")).unwrap();
    let plan = ml_lang::ir::lower(&inference).unwrap();
    let actual =
        ml_lang::inference::execute_with_backend(&plan, Some(&full.checkpoint), &TiledCpuBackend)
            .unwrap();
    assert_eq!(actual.output[0], full.output[3]);
}
#[test]
fn large_batched_ssa_and_reference_inference_match() {
    let p=compile("fn f(a:Tensor[B,M,K],b:Tensor[K,N])->Tensor[B,M,N]{return bmm(a,b);}param a=rand(2,17,35,42);param b=rand(35,129,123);let y=transpose_last(f(a,b));print(sum(y));").unwrap();
    let reference = ml_lang::vm::execute_with_options(&p, &CpuBackend, None, true).unwrap();
    let actual = ml_lang::inference::execute_with_backend(
        &ml_lang::ir::lower(&p).unwrap(),
        None,
        &TiledCpuBackend,
    )
    .unwrap();
    assert_eq!(reference.variables, actual.variables);
    assert_eq!(reference.output, actual.output);
}

#[test]
fn attention_training_retained_payload_is_bounded_over_steps() {
    let f = include_str!("../examples/attention.ml")
        .split("let x =")
        .next()
        .unwrap();
    let source = |steps| {
        format!("{f}let x=[[[1,0],[0,1]],[[0,1],[1,0]]];param wq=rand(2,4,42);param wk=rand(2,4,123);param wv=rand(2,2,99);repeat {steps}{{let loss=mse(attend(x,wq,wk,wv,0.5),x);backward(loss);clip_grad_norm(1);adam_step(0.03);}}")
    };
    assert_eq!(
        run(&source(3)).unwrap().peak_tensor_bytes,
        run(&source(3000)).unwrap().peak_tensor_bytes
    );
}
