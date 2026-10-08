use ml_lang::{
    compile, run,
    tensor::{forward, vjp, Op, Tensor},
};
fn value(source: &str, name: &str) -> Tensor {
    run(source).unwrap().variables[name].clone()
}
fn close(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= eps, "{x} != {y}");
    }
}
#[test]
fn literals_comments_precedence() {
    let t = value("// hi\nlet x = 2+3*4; let y = -1.5e2;", "x");
    close(&t.data, &[14.], 0.);
}
#[test]
fn matrix_example() {
    run(include_str!("../examples/matmul.ml")).unwrap();
}
#[test]
fn autodiff_example() {
    run(include_str!("../examples/autodiff.ml")).unwrap();
}
#[test]
fn xor_training_and_inference() {
    let r = run(include_str!("../examples/xor.ml")).unwrap();
    assert_eq!(r.output.len(), 3);
    assert!(r.output[0].data[0] > 1.0);
    assert!(r.output[1].data[0] < 0.001);
    close(&r.output[2].data, &[-1., 1., 1., -1.], 0.06);
}
#[test]
fn scalar_and_row_broadcast() {
    close(
        &value("let a = [[1,2],[3,4]] + [10,20] + 2;", "a").data,
        &[13., 24., 15., 26.],
        0.,
    );
}
#[test]
fn multidimensional_broadcast_gradient() {
    let r=run("param a = [[[1],[2]],[[3],[4]]]; param b = [2,3,4]; let loss = sum(a*b); backward(loss); let ga=grad(a); let gb=grad(b);").unwrap();
    close(&r.variables["ga"].data, &[9.; 4], 0.);
    close(&r.variables["gb"].data, &[10.; 3], 0.);
}
#[test]
fn shared_node_gradient_accumulates() {
    let t = value(
        "param x=2; let loss=x*x+x; backward(loss); let g=grad(x);",
        "g",
    );
    close(&t.data, &[5.], 0.);
}
#[test]
fn duplicate_backward_resets_gradients() {
    let t = value(
        "param x=2; let loss=x*x; backward(loss); backward(loss); let g=grad(x);",
        "g",
    );
    close(&t.data, &[4.], 0.);
}
#[test]
fn sgd_updates_parameter() {
    close(
        &value("param x=[1,2]; backward(sum(x*x)); step(0.1);", "x").data,
        &[0.8, 1.6],
        1e-14,
    );
}
#[test]
fn nested_function_and_scope() {
    let r=run("fn square(x:Tensor[2])->Tensor[2] { return x*x; } fn f(a:Tensor[2])->Tensor[] { return sum(square(a)); } let x=[2,3]; let y=f(x);").unwrap();
    close(&r.variables["y"].data, &[13.], 0.);
}
#[test]
fn loop_scope_and_nested_loop() {
    let r = run("param x=1; repeat 5 { repeat 2 { let loss=x*x; backward(loss); step(0.1); } }")
        .unwrap();
    close(&r.variables["x"].data, &[0.8f64.powi(10)], 1e-14);
    assert!(!r.variables.contains_key("loss"));
}
#[test]
fn zero_repeat_does_not_execute() {
    run("repeat 0 { assert_close(1,2,0); }").unwrap();
}
#[test]
fn loop_memory_is_bounded() {
    let small =
        run("param x=[1,2]; repeat 3 {let loss=sum(x*x); backward(loss); step(0.01);}").unwrap();
    let large =
        run("param x=[1,2]; repeat 3000 {let loss=sum(x*x); backward(loss); step(0.01);}").unwrap();
    assert_eq!(small.peak_tensor_bytes, large.peak_tensor_bytes);
}
#[test]
fn inference_loop_memory_is_bounded() {
    let small = run("let x=[[1,2],[3,4]]; repeat 2 {let y=tanh(x@x);}").unwrap();
    let large = run("let x=[[1,2],[3,4]]; repeat 2000 {let y=tanh(x@x);}").unwrap();
    assert_eq!(small.peak_tensor_bytes, large.peak_tensor_bytes);
}
#[test]
fn random_is_repeatable_and_seeded() {
    let a = value("let x=rand(2,3,42);", "x");
    let b = value("let x=rand(2,3,42);", "x");
    let c = value("let x=rand(2,3,43);", "x");
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert!(a.data.iter().all(|x| (-1.0..1.0).contains(x)));
}
#[test]
fn reject_invalid_programs() {
    for source in [
        "let x=[[1],[2,3]];",
        "let x:Tensor[2]=[1,2,3];",
        "let x=[[1,2]] @ [[1,2]];",
        "let x=[1,2]+[1,2,3];",
        "let x=missing;",
        "let x=1; let x=2;",
        "print(nope(1));",
        "backward([1,2]);",
        "step([1,2]);",
        "let x:Tensor[0]=[];",
        "let x=zeros(0);",
        "let x=zeros(1000000,1000000);",
        "let x=[1,2",
        "let x=1e999;",
        "let x=1 #;",
        "repeat 1 {param x=1;}",
        "let x=mse([1,2],1);",
        "assert_close([1,2],1,0);",
        "let x=tanh(1,2);",
        "let x=rand(2);",
        "fn f(x:Tensor[2])->Tensor[1]{return x;}",
        "fn f(x:Tensor[])->Tensor[]{return f(x);}",
        "fn f(x:Tensor[],x:Tensor[])->Tensor[]{return x;}",
        "fn tanh(x:Tensor[])->Tensor[]{return x;}",
        "fn f(x:Tensor[2])->Tensor[2]{return x;} let x=f([1]);",
        "fn f(x:Tensor[])->Tensor[]{return y;} let y=1;",
        "repeat 1 {let x=1;} print(x);",
        "fn f(x:Tensor[])->Tensor[]{return x;} let a=f();",
    ] {
        assert!(compile(source).is_err(), "accepted: {source}");
    }
}
#[test]
fn reject_runtime_errors() {
    for source in [
        "let x=1/0;",
        "param x=1; step(0.1);",
        "param x=1; let g=grad(x);",
        "backward(1);",
        "assert_close(1,2,0.1);",
        "param x=1; backward(x*x); step(-1);",
        "assert_close(1,1,-1);",
        "param x=1; backward(x*x); step(0.1); step(0.1);",
        "param x=1; repeat 1 {backward(x*x);} step(0.1);",
    ] {
        assert!(run(source).is_err(), "accepted: {source}");
    }
}
#[test]
fn parse_error_location() {
    let e = compile("// comment\nlet x = ?;").unwrap_err();
    assert!(e.contains("2:9"), "{e}");
}
#[test]
fn expression_nesting_limit() {
    let source = format!("let x={}1{};", "(".repeat(150), ")".repeat(150));
    assert!(compile(&source).unwrap_err().contains("nesting"));
}
#[test]
fn empty_program_is_valid() {
    assert!(run("").unwrap().output.is_empty());
}
fn finite_difference(op: Op, a: Tensor, b: Option<Tensor>) {
    let output = forward(op, &a, b.as_ref()).unwrap();
    let g: Vec<_> = (0..output.data.len())
        .map(|i| 0.3 + i as f64 * 0.1)
        .collect();
    let (ga, gb) = vjp(op, &a, b.as_ref(), &output, &g);
    let objective = |a: &Tensor, b: Option<&Tensor>| {
        forward(op, a, b)
            .unwrap()
            .data
            .iter()
            .zip(&g)
            .map(|(x, y)| x * y)
            .sum::<f64>()
    };
    let eps = 1e-6;
    for (i, &analytic) in ga.iter().enumerate() {
        let mut plus = a.clone();
        let mut minus = a.clone();
        plus.data[i] += eps;
        minus.data[i] -= eps;
        let numeric = (objective(&plus, b.as_ref()) - objective(&minus, b.as_ref())) / (2. * eps);
        close(&[analytic], &[numeric], 1e-6);
    }
    if let (Some(b), Some(gb)) = (b, gb) {
        for (i, &analytic) in gb.iter().enumerate() {
            let mut plus = b.clone();
            let mut minus = b.clone();
            plus.data[i] += eps;
            minus.data[i] -= eps;
            let numeric = (objective(&a, Some(&plus)) - objective(&a, Some(&minus))) / (2. * eps);
            close(&[analytic], &[numeric], 1e-6);
        }
    }
}
#[test]
fn finite_difference_all_elementwise_binary_ops() {
    for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
        finite_difference(
            op,
            Tensor::new(vec![2, 2], vec![0.2, -0.7, 1.2, 2.0]).unwrap(),
            Some(Tensor::new(vec![1, 2], vec![0.8, 1.3]).unwrap()),
        );
    }
}
#[test]
fn finite_difference_unary_and_reductions() {
    for op in [Op::Tanh, Op::Relu, Op::Neg, Op::Sum, Op::Mean] {
        finite_difference(
            op,
            Tensor::new(vec![2, 2], vec![0.2, -0.7, 1.2, 2.0]).unwrap(),
            None,
        );
    }
}
#[test]
fn finite_difference_matmul() {
    finite_difference(
        Op::Matmul,
        Tensor::new(vec![2, 3], vec![0.2, -0.7, 1.2, 2.0, 0.8, -0.3]).unwrap(),
        Some(Tensor::new(vec![3, 2], vec![0.8, 1.3, 0.7, -0.2, 0.3, 0.9]).unwrap()),
    );
}
#[test]
fn network_gradient_matches_finite_difference() {
    let source = |p: f64| {
        format!("let x=[[0.2,-0.4],[1,0.5]]; param w=[[{p},0.3],[-0.2,0.7]]; param b=[[0.1,-0.1]]; let loss=mse(tanh(x@w+b),[[0.5,0.6],[-0.3,0.1]]); backward(loss); let g=grad(w);")
    };
    let r = run(&source(0.4)).unwrap();
    let plus = run(&source(0.400001)).unwrap().variables["loss"].data[0];
    let minus = run(&source(0.399999)).unwrap().variables["loss"].data[0];
    close(&r.variables["g"].data[..1], &[(plus - minus) / 2e-6], 1e-7);
}
#[test]
fn relu_zero_subgradient() {
    close(
        &value(
            "param x=[-1,0,1]; backward(sum(relu(x))); let g=grad(x);",
            "g",
        )
        .data,
        &[0., 0., 1.],
        0.,
    );
}
#[test]
fn tensor_constructor_validates_size_and_finiteness() {
    assert!(Tensor::new(vec![2], vec![1.]).is_err());
    assert!(Tensor::new(vec![1], vec![f64::NAN]).is_err());
    assert!(Tensor::new(vec![usize::MAX, 2], vec![]).is_err());
}
#[test]
fn explicit_dtype_and_device() {
    run("let x:Tensor<f64,cpu>[2]=[1,2];").unwrap();
    assert!(compile("let x:Tensor<f32,cpu>[2]=[1,2];").is_err());
    assert!(compile("let x:Tensor<f64,cuda>[2]=[1,2];").is_err());
}
#[test]
fn optimizer_detaches_old_losses() {
    assert!(run("param x=1; let loss=x*x; backward(loss); step(0.1); backward(loss);").is_err());
}
#[test]
fn loop_boundary_detaches_derived_globals() {
    assert!(run("param x=1; let old=x*x; repeat 1 {let unused=1;} backward(old);").is_err());
}

#[test]
fn linear_training_example() {
    run(include_str!("../examples/linear.ml")).unwrap();
}

#[test]
fn transpose_values_and_inverse_gradient() {
    let r=run("param x=[[1,2,3],[4,5,6]]; let t=transpose(x); backward(sum(t*[[1,2],[3,4],[5,6]])); let g=grad(x);").unwrap();
    assert_eq!(r.variables["t"].shape, vec![3, 2]);
    close(&r.variables["t"].data, &[1., 4., 2., 5., 3., 6.], 0.);
    close(&r.variables["g"].data, &[1., 3., 5., 2., 4., 6.], 0.);
}
#[test]
fn reshape_values_shapes_and_gradient() {
    let r=run("param x=[[1,2,3],[4,5,6]]; let flat=reshape(x,6); let t=reshape(flat,3,2); backward(sum(t*t)); let g=grad(x); let scalar=reshape([7]);").unwrap();
    assert_eq!(r.variables["flat"].shape, vec![6]);
    assert_eq!(r.variables["t"].shape, vec![3, 2]);
    assert!(r.variables["scalar"].shape.is_empty());
    close(&r.variables["t"].data, &[1., 2., 3., 4., 5., 6.], 0.);
    close(&r.variables["g"].data, &[2., 4., 6., 8., 10., 12.], 0.);
}
#[test]
fn finite_difference_sigmoid_and_transpose() {
    for op in [Op::Sigmoid, Op::Transpose] {
        finite_difference(
            op,
            Tensor::new(vec![2, 3], vec![-2., 0.3, 1., -0.4, 0.8, 3.]).unwrap(),
            None,
        );
    }
}
#[test]
fn finite_difference_binary_cross_entropy() {
    finite_difference(
        Op::BceLogits,
        Tensor::new(vec![2, 2], vec![-2., 0.3, 1., 3.]).unwrap(),
        Some(Tensor::new(vec![2, 2], vec![0.2, 0.7, 0.3, 0.8]).unwrap()),
    );
}
#[test]
fn sigmoid_and_cross_entropy_extreme_logits() {
    let r=run("param x=[-1000,0,1000]; let p=sigmoid(x); let loss=bce_logits(x,[1,0.5,0]); backward(loss); let g=grad(x);").unwrap();
    close(&r.variables["p"].data, &[0., 0.5, 1.], 0.);
    close(
        &r.variables["loss"].data,
        &[(2000. + 2.0f64.ln()) / 3.],
        1e-12,
    );
    close(&r.variables["g"].data, &[-1. / 3., 0., 1. / 3.], 1e-12);
    let saturated = run("param x=[-1000,1000]; backward(sum(sigmoid(x))); let g=grad(x);").unwrap();
    close(&saturated.variables["g"].data, &[0., 0.], 0.);
}
#[test]
fn new_operations_reject_invalid_shapes_and_arguments() {
    for source in [
        "let x=transpose([1,2]);",
        "let x=transpose(1);",
        "let x=reshape([1,2],3);",
        "let x=reshape([1,2],0);",
        "let x=reshape([1,2],-1);",
        "let x=reshape();",
        "let x=reshape([1,2],[2]);",
        "let x=bce_logits([1,2],1);",
        "let x=bce_logits(1);",
        "let x=sigmoid(1,2);",
        "fn reshape(x:Tensor[])->Tensor[]{return x;}",
    ] {
        assert!(compile(source).is_err(), "accepted {source}");
    }
    for source in [
        "let loss=bce_logits(1,2);",
        "let loss=bce_logits([0,1],[-0.1,1]);",
    ] {
        assert!(run(source).is_err());
    }
}
#[test]
fn binary_xor_training_and_inference() {
    let r = run(include_str!("../examples/binary_xor.ml")).unwrap();
    assert!(r.output[1].data[0] < 0.02);
    close(&r.output[2].data, &[0., 1., 1., 0.], 0.02);
}
#[test]
fn shape_ops_and_matmul_gradient_matches_finite_difference() {
    let source = |p: f64| {
        format!("param x=[{p},0.3,-0.2,0.7,0.8,-0.5]; let matrix=reshape(x,3,2); let loss=mean(sigmoid(transpose(matrix)@matrix)); backward(loss); let g=grad(x);")
    };
    let analytic = run(&source(0.4)).unwrap().variables["g"].data[0];
    let plus = run(&source(0.400001)).unwrap().variables["loss"].data[0];
    let minus = run(&source(0.399999)).unwrap().variables["loss"].data[0];
    close(&[analytic], &[(plus - minus) / 2e-6], 1e-8);
}

#[test]
fn softmax_last_axis_values_normalization_and_rank() {
    let r = run("let p=softmax([0,0,0]); let q=softmax([[[0,0],[0,0]],[[0,0],[0,0]]]);").unwrap();
    close(&r.variables["p"].data, &[1. / 3.; 3], 1e-15);
    assert_eq!(r.variables["q"].shape, vec![2, 2, 2]);
    close(&r.variables["q"].data, &[0.5; 8], 1e-15);
}
#[test]
fn finite_difference_softmax() {
    finite_difference(
        Op::Softmax,
        Tensor::new(vec![2, 3], vec![-2., 0.3, 1., -0.4, 0.8, 3.]).unwrap(),
        None,
    );
}
#[test]
fn cross_entropy_logit_gradient_matches_finite_difference() {
    let a = Tensor::new(vec![2, 3], vec![-2., 0.3, 1., -0.4, 0.8, 3.]).unwrap();
    let targets = Tensor::new(vec![2, 3], vec![0.2, 0.7, 0.1, 0., 0., 1.]).unwrap();
    let out = forward(Op::CrossEntropy, &a, Some(&targets)).unwrap();
    let (gradient, _) = vjp(Op::CrossEntropy, &a, Some(&targets), &out, &[0.7]);
    for (i, &analytic) in gradient.iter().enumerate() {
        let mut plus = a.clone();
        let mut minus = a.clone();
        plus.data[i] += 1e-6;
        minus.data[i] -= 1e-6;
        let lp = forward(Op::CrossEntropy, &plus, Some(&targets))
            .unwrap()
            .data[0];
        let lm = forward(Op::CrossEntropy, &minus, Some(&targets))
            .unwrap()
            .data[0];
        close(&[analytic], &[0.7 * (lp - lm) / 2e-6], 1e-7);
    }
}
#[test]
fn cross_entropy_target_tangent_gradient() {
    // Labels live on a probability simplex: perturb two entries in opposite directions.
    let r=run("let z=[[0.2,-0.7,1.3]]; param y=[[0.2,0.5,0.3]]; let loss=cross_entropy(z,y); backward(loss); let g=grad(y);").unwrap();
    let objective = |e: f64| {
        run(&format!(
            "let loss=cross_entropy([[0.2,-0.7,1.3]],[[{}, {}, 0.3]]);",
            0.2 + e,
            0.5 - e
        ))
        .unwrap()
        .variables["loss"]
            .data[0]
    };
    let g = &r.variables["g"].data;
    close(
        &[g[0] - g[1]],
        &[(objective(1e-6) - objective(-1e-6)) / 2e-6],
        1e-8,
    );
}
#[test]
fn softmax_and_cross_entropy_stay_stable_at_extreme_offsets() {
    let r=run("param x=[[-1000,1000]]; let p=softmax(x); let loss=cross_entropy(x,[[1,0]]); backward(loss); let g=grad(x); let uniform=softmax([[1e16,1e16,1e16]]); let huge_loss=cross_entropy([[1e16,1e16,1e16]],[[1,0,0]]);").unwrap();
    close(&r.variables["p"].data, &[0., 1.], 0.);
    close(&r.variables["loss"].data, &[2000.], 1e-12);
    close(&r.variables["g"].data, &[-1., 1.], 1e-12);
    close(&r.variables["uniform"].data, &[1. / 3.; 3], 1e-15);
    close(&r.variables["huge_loss"].data, &[3.0f64.ln()], 1e-15);
    run("assert_close(softmax([[1e308,-1e308]]),[[1,0]],0); assert_close(cross_entropy([[1e308,-1e308]],[[1,0]]),0,0);").unwrap();
    assert!(run("let loss=cross_entropy([[1e308,-1e308]],[[0,1]]);").is_err());
}
#[test]
fn cross_entropy_mean_batch_and_shift_invariance() {
    let r=run("let z=[[0.2,-0.7,1.3],[-1,0.5,2]]; let y=[[0,0,1],[1,0,0]]; let a=cross_entropy(z,y); let b=cross_entropy(z+10000,y); assert_close(a,b,1e-11); assert_close(softmax(z),softmax(z+10000),1e-11); let uniform=cross_entropy([[0,0],[0,0]],[[1,0],[0,1]]);").unwrap();
    close(&r.variables["uniform"].data, &[2.0f64.ln()], 1e-15);
}
#[test]
fn multiclass_operations_reject_bad_shapes_and_labels() {
    for source in [
        "let p=softmax(1);",
        "let p=softmax([1,2],1);",
        "let loss=cross_entropy([1,2],[1,0]);",
        "let loss=cross_entropy([[1,2]],[[1],[0]]);",
        "let loss=cross_entropy([[1,2]]);",
        "fn softmax(x:Tensor[])->Tensor[]{return x;}",
    ] {
        assert!(compile(source).is_err(), "accepted {source}");
    }
    for source in [
        "let loss=cross_entropy([[0,0]],[[0,0]]);",
        "let loss=cross_entropy([[0,0]],[[1,1]]);",
        "let loss=cross_entropy([[0,0]],[[-0.1,1.1]]);",
    ] {
        assert!(run(source).is_err(), "accepted {source}");
    }
}
#[test]
fn multiclass_training_and_inference() {
    let r = run(include_str!("../examples/multiclass.ml")).unwrap();
    assert!(r.output[1].data[0] < 0.02);
    let expected = [
        1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
    ];
    close(&r.output[2].data, &expected, 0.02);
}
#[test]
fn multiclass_network_gradient_matches_finite_difference() {
    let source = |p: f64| {
        format!("let x=[[0.2,-0.4],[1,0.5]]; param w=[[{p},0.3,-0.5],[-0.2,0.7,0.1]]; param b=[[0.1,-0.1,0.2]]; let loss=cross_entropy(x@w+b,[[0,1,0],[0.2,0.3,0.5]]); backward(loss); let g=grad(w);")
    };
    let analytic = run(&source(0.4)).unwrap().variables["g"].data[0];
    let plus = run(&source(0.400001)).unwrap().variables["loss"].data[0];
    let minus = run(&source(0.399999)).unwrap().variables["loss"].data[0];
    close(&[analytic], &[(plus - minus) / 2e-6], 1e-8);
}

#[test]
fn function_local_bindings_and_nested_calls() {
    let r=run("fn square(x:Tensor[2])->Tensor[2] {let product=x*x;return product;} fn f(x:Tensor[2])->Tensor[] {let y=square(x);let total:Tensor[]=sum(y);return total;} let y=99; let z=f([2,3])+f([4,5]);").unwrap();
    close(&r.variables["z"].data, &[54.], 0.);
    close(&r.variables["y"].data, &[99.], 0.);
    assert!(!r.variables.contains_key("product"));
    assert!(!r.variables.contains_key("total"));
}
#[test]
fn function_locals_preserve_gradients_and_multiple_consumers() {
    let r=run("fn f(x:Tensor[2])->Tensor[]{let squared=x*x;let total=sum(squared+x);return total;} param x=[2,3];backward(f(x)+f(x));let g=grad(x);").unwrap();
    close(&r.variables["g"].data, &[10., 14.], 0.);
}
#[test]
fn unused_function_local_errors_are_checked() {
    for source in [
        "fn f(x:Tensor[2])->Tensor[2]{let y:Tensor[3]=x;return x;}",
        "fn f(x:Tensor[2])->Tensor[2]{let y=missing;return x;}",
        "fn f(x:Tensor[2])->Tensor[2]{let x=x;return x;}",
        "fn f(x:Tensor[2])->Tensor[2]{let y=x;let y=x;return y;}",
        "fn f(x:Tensor[2])->Tensor[2]{let y=z;let z=x;return y;}",
        "fn f(x:Tensor[2])->Tensor[2]{let y=f(x);return y;}",
        "fn f(x:Tensor[2])->Tensor[2]{let y=x;return sum(y);}",
        "let global=1;fn f(x:Tensor[])->Tensor[]{let y=global;return y;}",
        "fn f(x:Tensor[])->Tensor[]{let y=x;return y;} print(y);",
    ] {
        assert!(compile(source).is_err(), "accepted {source}");
    }
}
#[test]
fn function_requires_final_return_and_pure_local_statements() {
    for source in [
        "fn f(x:Tensor[])->Tensor[]{let y=x;}",
        "fn f(x:Tensor[])->Tensor[]{return x;let y=x;}",
        "fn f(x:Tensor[])->Tensor[]{param y=x;return y;}",
        "fn f(x:Tensor[])->Tensor[]{print(x);return x;}",
        "fn f(x:Tensor[])->Tensor[]{backward(x);return x;}",
        "fn f(x:Tensor[])->Tensor[]{repeat 2 {let y=x;}return x;}",
    ] {
        assert!(compile(source).is_err(), "accepted {source}");
    }
}
#[test]
fn function_local_storage_is_bounded_in_training_loop() {
    let source = |steps: usize| {
        format!("fn loss(x:Tensor[2])->Tensor[]{{let squared=x*x;let total=sum(squared);return total;}} param x=[1,2];repeat {steps} {{let y=loss(x);backward(y);step(0.01);}}")
    };
    let small = run(&source(3)).unwrap();
    let large = run(&source(3000)).unwrap();
    assert_eq!(small.peak_tensor_bytes, large.peak_tensor_bytes);
}
#[test]
fn function_local_network_gradient_matches_finite_difference() {
    let source = |p: f64| {
        format!("fn loss(x:Tensor[2,2],w:Tensor[2,3])->Tensor[]{{let hidden=tanh(x@w);let result=cross_entropy(hidden,[[0,1,0],[0.2,0.3,0.5]]);return result;}} let x=[[0.2,-0.4],[1,0.5]];param w=[[{p},0.3,-0.5],[-0.2,0.7,0.1]];let y=loss(x,w);backward(y);let g=grad(w);")
    };
    let analytic = run(&source(0.4)).unwrap().variables["g"].data[0];
    let plus = run(&source(0.400001)).unwrap().variables["y"].data[0];
    let minus = run(&source(0.399999)).unwrap().variables["y"].data[0];
    close(&[analytic], &[(plus - minus) / 2e-6], 1e-8);
}

#[test]
fn generic_function_multiple_batches_and_nested_calls() {
    let source="fn dense(x:Tensor[B,I],w:Tensor[I,O],b:Tensor[1,O])->Tensor[B,O]{let output:Tensor[B,O]=x@w+b;return output;} fn predict(x:Tensor[N,2],w:Tensor[2,3],b:Tensor[1,3])->Tensor[N,3]{let z=dense(x,w,b);return softmax(z);} let w=[[1,0,-1],[0,1,-1]];let b=zeros(1,3);let one=predict([[1,2]],w,b);let many=predict([[1,2],[3,4]],w,b);";
    let r = run(source).unwrap();
    assert_eq!(r.variables["one"].shape, vec![1, 3]);
    assert_eq!(r.variables["many"].shape, vec![2, 3]);
    close(&r.variables["one"].data, &r.variables["many"].data[..3], 0.);
}
#[test]
fn symbolic_initializers_and_reshape() {
    let r=run("fn f(x:Tensor[B,2])->Tensor[B,2]{let zero:Tensor[B,2]=zeros(B,2);let expanded=reshape(x,B,2,1);return reshape(expanded,B,2)+zero;} let a=f([[1,2],[3,4]]);let b=f([[5,6]]);").unwrap();
    close(&r.variables["a"].data, &[1., 2., 3., 4.], 0.);
    close(&r.variables["b"].data, &[5., 6.], 0.);
}
#[test]
fn symbolic_type_errors_are_checked_without_calls() {
    for s in ["fn f(x:Tensor[B,2])->Tensor[B,3]{return x;}","fn f(x:Tensor[B,2])->Tensor[B,3]{return x@zeros(B,3);}","fn f(x:Tensor[B,2],y:Tensor[C,2])->Tensor[B,2]{return x+y;}","fn f(x:Tensor[B,2])->Tensor[C,2]{return x;}","fn f(x:Tensor[B,2])->Tensor[B,2]{let y:Tensor[C,2]=x;return y;}","fn f(x:Tensor[B,2])->Tensor[B,2]{return reshape(x,B,3);}","fn f(x:Tensor[B,2])->Tensor[B,2]{return zeros(C,2);}","let x:Tensor[B,2]=[[1,2]];","fn f(x:Tensor[B,2])->Tensor[B,2]{return x;}let x=f([1,2]);","fn f(x:Tensor[B,2],y:Tensor[B,2])->Tensor[B,2]{return x+y;}let x=f([[1,2]],[[1,2],[3,4]]);"] {assert!(compile(s).is_err(),"accepted {s}");}
}
#[test]
fn generic_gradients_accumulate_over_different_specializations() {
    let r=run("fn loss(x:Tensor[B,2],w:Tensor[2,1])->Tensor[]{let y=x@w;return sum(y*y);}param w=[[1],[2]];let loss=loss([[1,0]],w)+loss([[0,1],[1,1]],w);backward(loss);let g=grad(w);").unwrap();
    close(&r.variables["g"].data, &[8., 10.], 0.);
}
