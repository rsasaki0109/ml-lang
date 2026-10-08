use ml_lang::{
    compile, run,
    tensor::{forward, vjp, Op, Tensor},
};
fn close(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= eps, "{x} != {y}");
    }
}
#[test]
fn rows_values_ranks_and_symbolic_trailing_dimensions() {
    let r=run("fn batch(x:Tensor[4,F])->Tensor[2,F]{return rows(x,1,2);}let y=batch([[1,2],[3,4],[5,6],[7,8]]);let vector=rows([1,2,3],0,2);let cube=rows([[[1,2]],[[3,4]],[[5,6]]],1,1);").unwrap();
    assert_eq!(r.variables["y"].shape, vec![2, 2]);
    close(&r.variables["y"].data, &[3., 4., 5., 6.], 0.);
    close(&r.variables["vector"].data, &[1., 2.], 0.);
    assert_eq!(r.variables["cube"].shape, vec![1, 1, 2]);
    close(&r.variables["cube"].data, &[3., 4.], 0.);
}
#[test]
fn rows_scatter_gradient_matches_every_finite_difference() {
    let a = Tensor::new(vec![3, 2], vec![0.1, 0.2, 0.3, -0.4, 0.5, -0.6]).unwrap();
    let op = Op::Rows { start: 1, count: 2 };
    let out = forward(op, &a, None).unwrap();
    let g = vec![1., -2., 3., -4.];
    let (actual, none) = vjp(op, &a, None, &out, &g);
    assert!(none.is_none());
    for (i, &analytic) in actual.iter().enumerate() {
        let loss = |offset| {
            let mut x = a.clone();
            x.data[i] += offset;
            forward(op, &x, None)
                .unwrap()
                .data
                .iter()
                .zip(&g)
                .map(|(x, g)| x * g)
                .sum::<f64>()
        };
        close(&[analytic], &[(loss(1e-6) - loss(-1e-6)) / 2e-6], 1e-9);
    }
    let r=run("param x=[[1,2],[3,4],[5,6]];let a=rows(x,0,2);let b=rows(x,1,2);backward(sum(a*a)+sum(b));let g=grad(x);").unwrap();
    close(&r.variables["g"].data, &[2., 4., 7., 9., 1., 1.], 0.);
}
#[test]
fn rows_reject_invalid_ranges_and_unprovable_generic_bounds() {
    for source in [
        "let x=rows(1,0,1);",
        "let x=rows([1,2],0,0);",
        "let x=rows([1,2],-1,1);",
        "let x=rows([1,2],1,2);",
        "let x=rows([1,2],0.5,1);",
        "let n=1;let x=rows([1,2],n,1);",
        "let x=rows([1,2],0);",
        "fn f(x:Tensor[B,2])->Tensor[1,2]{return rows(x,0,1);}",
        "fn f(x:Tensor[2,F])->Tensor[1,F]{return rows(x,2,1);}",
    ] {
        assert!(compile(source).is_err(), "accepted {source}");
    }
    let a = Tensor::new(vec![2], vec![1., 2.]).unwrap();
    assert!(forward(
        Op::Rows {
            start: usize::MAX,
            count: 2
        },
        &a,
        None
    )
    .is_err());
}
#[test]
fn detach_cuts_only_selected_edges_and_takes_a_snapshot() {
    let r=run("param x=[2,3];let frozen=detach(x);let loss=sum(x*detach(x));backward(loss);let g=grad(x);step(0.1);assert_close(frozen,[2,3],0);").unwrap();
    close(&r.variables["g"].data, &[2., 3.], 0.);
    close(&r.variables["x"].data, &[1.8, 2.7], 1e-14);
    assert!(run("param x=2;backward(detach(x*x));").is_err());
    assert!(run("param x=2;let d=detach(x);backward(x*x);let g=grad(d);").is_err());
    for source in [
        "let x=detach();",
        "let x=detach(1,2);",
        "fn f(x:Tensor[B])->Tensor[B]{return detach(x);}let y=f([1,2]);",
    ] {
        if source.starts_with("fn") {
            run(source).unwrap();
        } else {
            assert!(compile(source).is_err());
        }
    }
}
#[test]
fn clipping_is_global_handles_aliases_and_preserves_unconnected_parameters() {
    let r=run("param x=[3,4];param y=12;param unused=9;let alias=x;backward(sum(x)+y);clip_grad_norm(1);let gx=grad(x);let gy=grad(y);step(0.1);").unwrap();
    let scale = 1. / 3f64.sqrt();
    close(&r.variables["gx"].data, &[scale, scale], 1e-15);
    close(&r.variables["gy"].data, &[scale], 1e-15);
    close(
        &r.variables["alias"].data,
        &[3. - 0.1 * scale, 4. - 0.1 * scale],
        1e-15,
    );
    close(&r.variables["unused"].data, &[9.], 0.);
    let r=run("param x=[3,4];backward(sum(x*x)/2);let before=grad(x);clip_grad_norm(2);let after=grad(x);clip_grad_norm(2);let again=grad(x);").unwrap();
    close(&r.variables["before"].data, &[3., 4.], 0.);
    close(&r.variables["after"].data, &[1.2, 1.6], 1e-15);
    close(&r.variables["again"].data, &[1.2, 1.6], 1e-15);
}
#[test]
fn clipping_stays_finite_at_extreme_and_zero_norms() {
    use ml_lang::optimizer::clip_transform;
    let gradient = vec![1e308, -1e308];
    let (divisor, magnitude) = clip_transform(std::iter::once(gradient.as_slice()), 1.)
        .unwrap()
        .unwrap();
    close(
        &[
            (gradient[0] / divisor) * magnitude,
            (gradient[1] / divisor) * magnitude,
        ],
        &[1. / 2f64.sqrt(), -1. / 2f64.sqrt()],
        1e-15,
    );
    assert_eq!(
        clip_transform(std::iter::once(&[0., 0.][..]), 1.).unwrap(),
        None
    );
    assert_eq!(
        clip_transform(std::iter::once(&[1e-300, -1e-300][..]), 1.).unwrap(),
        None
    );
    assert!(clip_transform(std::iter::once(&[f64::INFINITY][..]), 1.).is_err());
    let tiny =
        run("param x=[1,1];backward(sum(x)*1e300);clip_grad_norm(1e-100);let g=grad(x);").unwrap();
    for &value in &tiny.variables["g"].data {
        assert!((value / (1e-100 / 2f64.sqrt()) - 1.).abs() < 1e-15);
    }
    let r = run("param x=1;backward(x*1e200);clip_grad_norm(1);adam_step(0.1);").unwrap();
    close(&r.variables["x"].data, &[1. - 0.1 / (1. + 1e-8)], 1e-15);
    assert!(run("param x=1;backward(x*1e200);adam_step(0.1);").is_err());
    let r = run("param x=1;backward(x*0);clip_grad_norm(1);step(0.1);").unwrap();
    close(&r.variables["x"].data, &[1.], 0.);
}
#[test]
fn clipping_requires_fresh_backward_and_a_positive_scalar_limit() {
    for source in [
        "param x=1;clip_grad_norm(1);",
        "param x=1;backward(x);clip_grad_norm(0);",
        "param x=1;backward(x);clip_grad_norm(-1);",
        "param x=1;backward(x);step(0.1);clip_grad_norm(1);",
        "param x=1;backward(x);repeat 0{}clip_grad_norm(1);",
        "param x=1;backward(x);clip_grad_norm([1]);",
    ] {
        // A zero-count loop has no boundary to detach, so this particular program is valid.
        if source.contains("repeat 0") {
            run(source).unwrap();
        } else {
            assert!(run(source).is_err(), "accepted {source}");
        }
    }
    assert!(run("param x=1;backward(x);repeat 1{}clip_grad_norm(1);").is_err());
    let program = compile("param x=1;repeat 0{clip_grad_norm(1);}").unwrap();
    assert!(ml_lang::vm::execute_inference(&program, None).is_err());
}
#[test]
fn minibatch_clipped_training_and_resume() {
    let source = include_str!("../examples/minibatch.ml");
    let full = run(source).unwrap();
    assert!(full.output[1].data[0] < 0.02);
    let first = run(&source.replace("repeat 100", "repeat 37")).unwrap();
    let saved =
        ml_lang::checkpoint::Checkpoint::decode(&first.checkpoint.encode().unwrap()).unwrap();
    let resumed = ml_lang::vm::execute_with_checkpoint(
        &compile(&source.replace("repeat 100", "repeat 63")).unwrap(),
        &saved,
    )
    .unwrap();
    assert_eq!(full.checkpoint, resumed.checkpoint);
}
#[test]
fn minibatch_retained_payload_is_bounded_over_epochs() {
    let source = |epochs| {
        format!("param x=[[1,2],[3,4]];repeat {epochs} {{let b=rows(x,0,1);backward(sum(b*b));clip_grad_norm(1);adam_step(0.01);let c=rows(x,1,1);backward(sum(c*c));clip_grad_norm(1);adam_step(0.01);}}")
    };
    assert_eq!(
        run(&source(3)).unwrap().peak_tensor_bytes,
        run(&source(3000)).unwrap().peak_tensor_bytes
    );
}
