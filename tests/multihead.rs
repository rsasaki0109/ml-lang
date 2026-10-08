use ml_lang::{
    backend::{Backend, CpuBackend},
    compile, run,
    tensor::{forward, vjp, Op, Tensor},
    tiled::TiledCpuBackend,
};
#[test]
fn axis_swap_values_inverse_and_every_vjp_are_correct() {
    let x = Tensor::new(vec![2, 3, 2], (0..12).map(|i| i as f64).collect()).unwrap();
    let op = Op::SwapAxes {
        first: 0,
        second: 1,
    };
    let y = forward(op, &x, None).unwrap();
    assert_eq!(y.shape, vec![3, 2, 2]);
    assert_eq!(
        y.data,
        vec![0., 1., 6., 7., 2., 3., 8., 9., 4., 5., 10., 11.]
    );
    for first in 0..3 {
        for second in 0..3 {
            let op = Op::SwapAxes { first, second };
            let y = forward(op, &x, None).unwrap();
            assert_eq!(forward(op, &y, None).unwrap(), x);
            let g: Vec<_> = (0..12).map(|i| i as f64 / 7. - 0.2).collect();
            let (grad, _) = vjp(op, &x, None, &y, &g);
            for (i, &analytic) in grad.iter().enumerate() {
                let loss = |d| {
                    let mut x = x.clone();
                    x.data[i] += d;
                    forward(op, &x, None)
                        .unwrap()
                        .data
                        .iter()
                        .zip(&g)
                        .map(|(x, g)| x * g)
                        .sum::<f64>()
                };
                assert!((analytic - (loss(1e-5) - loss(-1e-5)) / 2e-5).abs() < 1e-8);
            }
        }
    }
}
#[test]
fn multiple_batch_axes_vjps_match_all_input_finite_differences() {
    for shared in [true, false] {
        let a = Tensor::new(
            vec![2, 2, 2, 3],
            (0..24).map(|i| i as f64 / 17. - 0.4).collect(),
        )
        .unwrap();
        let shape = if shared { vec![3, 2] } else { vec![2, 2, 3, 2] };
        let b = Tensor::new(
            shape.clone(),
            (0..shape.iter().product())
                .map(|i| i as f64 / 19. - 0.2)
                .collect(),
        )
        .unwrap();
        let y = forward(Op::BatchMatmul, &a, Some(&b)).unwrap();
        assert_eq!(y.shape, vec![2, 2, 2, 2]);
        let g: Vec<_> = (0..16).map(|i| (i % 5) as f64 / 11. - 0.3).collect();
        let (ga, gb) = vjp(Op::BatchMatmul, &a, Some(&b), &y, &g);
        for left in [true, false] {
            let grad = if left { &ga } else { gb.as_ref().unwrap() };
            for (i, &analytic) in grad.iter().enumerate() {
                let loss = |d| {
                    let mut a = a.clone();
                    let mut b = b.clone();
                    if left {
                        a.data[i] += d
                    } else {
                        b.data[i] += d
                    };
                    forward(Op::BatchMatmul, &a, Some(&b))
                        .unwrap()
                        .data
                        .iter()
                        .zip(&g)
                        .map(|(x, g)| x * g)
                        .sum::<f64>()
                };
                assert!((analytic - (loss(1e-6) - loss(-1e-6)) / 2e-6).abs() < 2e-9);
            }
        }
    }
}
#[test]
fn concrete_and_symbolic_batch_prefixes_and_axes_are_checked() {
    run("fn f(a:Tensor[B,H,M,K],b:Tensor[B,H,K,N])->Tensor[B,H,M,N]{return bmm(a,b);}let y=f(ones(2,3,4,5),ones(2,3,5,6));").unwrap();
    run(
        "fn f(x:Tensor[B,T,H,D])->Tensor[B,H,T,D]{return swap_axes(x,1,2);}let y=f(ones(2,3,4,5));",
    )
    .unwrap();
    for s in [
        "let y=bmm(ones(2,3,4,5),ones(2,1,5,6));",
        "let y=bmm(ones(2,3,4,5),ones(3,5,6));",
        "fn f(a:Tensor[B,H,M,K],b:Tensor[B,J,K,N])->Tensor[B,H,M,N]{return bmm(a,b);}",
        "let y=swap_axes([1,2],0,1);",
        "let y=swap_axes([1,2],0);",
        "let y=swap_axes([1,2],-1,0);",
        "let y=swap_axes([[1]],0.5,1);",
        "fn f(x:Tensor[B,T])->Tensor[B,T]{return swap_axes(x,0,1);}",
    ] {
        assert!(compile(s).is_err(), "accepted {s}");
    }
}
#[test]
fn multiple_large_batch_axes_match_tiled_forward_and_vjp() {
    for shared in [true, false] {
        let a = Tensor::new(
            vec![2, 2, 17, 35],
            (0..4 * 17 * 35)
                .map(|i| (i % 37) as f64 / 23. - 0.5)
                .collect(),
        )
        .unwrap();
        let shape = if shared {
            vec![35, 129]
        } else {
            vec![2, 2, 35, 129]
        };
        let b = Tensor::new(
            shape.clone(),
            (0..shape.iter().product())
                .map(|i| (i % 31) as f64 / 29. - 0.5)
                .collect(),
        )
        .unwrap();
        let shape = vec![2, 2, 17, 129];
        let y = CpuBackend
            .forward(Op::BatchMatmul, &a, Some(&b), &shape)
            .unwrap();
        assert_eq!(
            y,
            TiledCpuBackend
                .forward(Op::BatchMatmul, &a, Some(&b), &shape)
                .unwrap()
        );
        let g: Vec<_> = y.data.iter().map(|x| x / 13.).collect();
        assert_eq!(
            CpuBackend.vjp(Op::BatchMatmul, &a, Some(&b), &y, &g),
            TiledCpuBackend.vjp(Op::BatchMatmul, &a, Some(&b), &y, &g)
        );
    }
}
#[test]
fn multihead_checkpoint_resume_and_loaded_ssa_agree() {
    let s = include_str!("../examples/multihead_attention.ml");
    let full = run(s).unwrap();
    assert!(full.output[1].data[0] < 0.001);
    let first = run(&s.replace("repeat 500", "repeat 217")).unwrap();
    let saved =
        ml_lang::checkpoint::Checkpoint::decode(&first.checkpoint.encode().unwrap()).unwrap();
    let resumed = ml_lang::vm::execute_with_options(
        &compile(&s.replace("repeat 500", "repeat 283")).unwrap(),
        &TiledCpuBackend,
        Some(&saved),
        false,
    )
    .unwrap();
    assert_eq!(full.checkpoint, resumed.checkpoint);
    let f = s.split("let x =").next().unwrap();
    let p=compile(&format!("{f}param wq=zeros(2,4);param wk=zeros(2,4);param wv=zeros(2,4);param wo=zeros(4,2);let x=[[[1,0],[0,1],[0,1]]];print(attend(x,wq,wk,wv,wo));")).unwrap();
    let ir = ml_lang::ir::lower(&p).unwrap();
    let reference =
        ml_lang::vm::execute_with_options(&p, &CpuBackend, Some(&full.checkpoint), true).unwrap();
    for backend in [&CpuBackend as &dyn Backend, &TiledCpuBackend] {
        let actual =
            ml_lang::inference::execute_with_backend(&ir, Some(&full.checkpoint), backend).unwrap();
        assert_eq!(reference.output, actual.output);
        assert_eq!(full.output[2], actual.output[0]);
    }
}
#[test]
fn every_projection_and_future_dependency_are_validated() {
    let f = include_str!("../examples/multihead_attention.ml")
        .split("let x =")
        .next()
        .unwrap();
    for weight in ["wq", "wk", "wv", "wo"] {
        let source = |delta| {
            let matrix = |name| {
                if name == weight {
                    if name == "wo" {
                        format!("[[{},0.2],[-0.3,0.4],[0.5,-0.1],[0.2,0.3]]", 0.1 + delta)
                    } else {
                        format!("[[{},0.2,-0.3,0.4],[0.5,-0.1,0.2,0.3]]", 0.1 + delta)
                    }
                } else if name == "wo" {
                    "[[0.1,0.2],[-0.3,0.4],[0.5,-0.1],[0.2,0.3]]".into()
                } else {
                    "[[0.1,0.2,-0.3,0.4],[0.5,-0.1,0.2,0.3]]".into()
                }
            };
            format!("{f}let x=[[[1,0],[0,1],[0.2,0.8]]];param wq={};param wk={};param wv={};param wo={};let loss=mse(attend(x,wq,wk,wv,wo),x);backward(loss);let g=grad({weight});",matrix("wq"),matrix("wk"),matrix("wv"),matrix("wo"))
        };
        let actual = run(&source(0.)).unwrap().variables["g"].data[0];
        let plus = run(&source(1e-6)).unwrap().variables["loss"].data[0];
        let minus = run(&source(-1e-6)).unwrap().variables["loss"].data[0];
        assert!((actual - (plus - minus) / 2e-6).abs() < 1e-9);
    }
    let source = |future| {
        format!("{f}param x=[[[1,0],[0,1],[{future},-7]]];let w=ones(2,4);let y=attend(x,w,w,w,ones(4,2));let prefix=rows(reshape(y,3,2),0,2);backward(sum(prefix));let g=grad(x);")
    };
    let a = run(&source(2)).unwrap();
    let b = run(&source(900)).unwrap();
    assert_eq!(a.variables["prefix"], b.variables["prefix"]);
    assert_eq!(&a.variables["g"].data[4..], &[0., 0.]);
    assert_eq!(&b.variables["g"].data[4..], &[0., 0.]);
}
