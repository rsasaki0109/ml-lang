use ml_lang::{
    backend::{Backend, CpuBackend},
    compile, run,
    tensor::{forward, vjp, Op, Tensor},
    tiled::TiledCpuBackend,
};
#[test]
fn gelu_values_and_extreme_finite_tails_are_correct() {
    let x = Tensor::new(vec![7], vec![-f64::MAX, -10., -1., 0., 1., 10., f64::MAX]).unwrap();
    let y = forward(Op::Gelu, &x, None).unwrap();
    assert_eq!(y.data[0], 0.);
    assert_eq!(y.data[1], 0.);
    assert!((y.data[2] + 0.15880800939172324).abs() < 1e-14);
    assert_eq!(y.data[3], 0.);
    assert!((y.data[4] - 0.8411919906082768).abs() < 1e-14);
    assert_eq!(y.data[5], 10.);
    assert_eq!(y.data[6], f64::MAX);
    let (g, _) = vjp(Op::Gelu, &x, None, &y, &[1.; 7]);
    assert_eq!(g[0], 0.);
    assert_eq!(g[1], 0.);
    assert_eq!(g[3], 0.5);
    assert_eq!(g[5], 1.);
    assert_eq!(g[6], 1.);
}
#[test]
fn all_gelu_gradients_match_finite_differences() {
    for shape in [vec![], vec![9], vec![3, 3], vec![1, 3, 3]] {
        let x = Tensor::new(
            shape.clone(),
            (0..shape.iter().product::<usize>())
                .map(|i| i as f64 - 4.)
                .collect(),
        )
        .unwrap();
        let y = forward(Op::Gelu, &x, None).unwrap();
        let g: Vec<_> = (0..x.data.len()).map(|i| i as f64 / 7. + 0.2).collect();
        let (grad, _) = vjp(Op::Gelu, &x, None, &y, &g);
        for (i, &analytic) in grad.iter().enumerate() {
            let loss = |d| {
                let mut x = x.clone();
                x.data[i] += d;
                forward(Op::Gelu, &x, None)
                    .unwrap()
                    .data
                    .iter()
                    .zip(&g)
                    .map(|(v, g)| v * g)
                    .sum::<f64>()
            };
            assert!((analytic - (loss(1e-5) - loss(-1e-5)) / 2e-5).abs() < 2e-9);
        }
    }
}
#[test]
fn gelu_symbolic_shape_arity_and_ssa_are_checked() {
    run("fn f(x:Tensor[B,T,D])->Tensor[B,T,D]{return gelu(x);}let y=f(ones(2,3,4));let scalar=gelu(0);").unwrap();
    for s in [
        "let x=gelu();",
        "let x=gelu([1],[2]);",
        "fn gelu(x:Tensor[D])->Tensor[D]{return x;}",
        "fn f(x:Tensor[B,T])->Tensor[T,B]{return gelu(x);}",
    ] {
        assert!(compile(s).is_err());
    }
    let p = compile("let x=gelu([[-2,-1,0],[1,2,3]]);print(x);").unwrap();
    let reference = ml_lang::vm::execute_with_options(&p, &CpuBackend, None, true).unwrap();
    let actual = ml_lang::inference::execute_with_backend(
        &ml_lang::ir::lower(&p).unwrap(),
        None,
        &TiledCpuBackend,
    )
    .unwrap();
    assert_eq!(reference.output, actual.output);
}
#[test]
fn transformer_train_resume_and_loaded_ssa_are_exact() {
    let s = include_str!("../examples/transformer.ml");
    let full = run(s).unwrap();
    assert!(full.output[1].data[0] < 1e-7);
    assert_eq!(full.checkpoint.parameters.len(), 14);
    assert_eq!(
        full.checkpoint
            .parameters
            .values()
            .map(|v| v.data.len())
            .sum::<usize>(),
        166
    );
    let first = run(&s.replace("repeat 300", "repeat 137")).unwrap();
    let saved =
        ml_lang::checkpoint::Checkpoint::decode(&first.checkpoint.encode().unwrap()).unwrap();
    let resumed = ml_lang::vm::execute_with_options(
        &compile(&s.replace("repeat 300", "repeat 163")).unwrap(),
        &TiledCpuBackend,
        Some(&saved),
        false,
    )
    .unwrap();
    assert_eq!(full.checkpoint, resumed.checkpoint);
    let p = compile(include_str!("../examples/transformer_infer.ml")).unwrap();
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
fn every_transformer_parameter_gradient_matches_finite_differences() {
    let s = include_str!("../examples/transformer.ml");
    let prefix = s.split("let initial =").next().unwrap();
    let tail="let logits=model(x,wq,wk,wv,wo,gain1,bias1,gain2,bias2,w1,b1,w2,b2,readout,readbias);let loss=cross_entropy(reshape(logits,6,2),reshape(target,6,2));";
    let p = compile(&format!("{prefix}{tail}")).unwrap();
    let initial = ml_lang::vm::execute(&p).unwrap();
    let mut analytic_source = format!("{prefix}{tail}backward(loss);");
    for name in initial.checkpoint.parameters.keys() {
        analytic_source += &format!("let g_{name}=grad({name});");
    }
    let analytic = run(&analytic_source).unwrap();
    let mut count = 0;
    for (name, value) in &initial.checkpoint.parameters {
        for i in 0..value.data.len() {
            let loss = |delta| {
                let mut checkpoint = initial.checkpoint.clone();
                checkpoint.parameters.get_mut(name).unwrap().data[i] += delta;
                ml_lang::vm::execute_with_options(&p, &CpuBackend, Some(&checkpoint), false)
                    .unwrap()
                    .variables["loss"]
                    .data[0]
            };
            let numeric = (loss(1e-5) - loss(-1e-5)) / 2e-5;
            let actual = analytic.variables[&format!("g_{name}")].data[i];
            assert!(
                (actual - numeric).abs() < 2e-8,
                "{name}[{i}]: {actual} vs {numeric}"
            );
            count += 1;
        }
    }
    assert_eq!(count, 166);
}
#[test]
fn transformer_prefix_is_independent_of_future_features() {
    let s = include_str!("../examples/transformer.ml");
    let prefix = s.split("let initial =").next().unwrap();
    let tail="param input=[[[1,0,1,0],[0,1,0,1],[FUTURE,-7,9,5]]];let y=model(input,wq,wk,wv,wo,gain1,bias1,gain2,bias2,w1,b1,w2,b2,readout,readbias);let prefix=rows(reshape(y,3,2),0,2);backward(sum(prefix));let g=grad(input);";
    let a = run(&format!("{prefix}{}", tail.replace("FUTURE", "2"))).unwrap();
    let b = run(&format!("{prefix}{}", tail.replace("FUTURE", "900"))).unwrap();
    assert_eq!(a.variables["prefix"], b.variables["prefix"]);
    assert_eq!(&a.variables["g"].data[8..], &[0.; 4]);
    assert_eq!(&b.variables["g"].data[8..], &[0.; 4]);
}
