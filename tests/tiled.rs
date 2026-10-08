use ml_lang::{
    backend::{Backend, CpuBackend},
    compile,
    tensor::{Op, Tensor},
    tiled::TiledCpuBackend,
    vm,
};
fn matrix(m: usize, n: usize, seed: usize) -> Tensor {
    Tensor::new(
        vec![m, n],
        (0..m * n)
            .map(|i| (((i * 17 + seed) % 41) as f64 - 20.) / 23.)
            .collect(),
    )
    .unwrap()
}
#[test]
fn tiled_forward_and_both_gradients_match_all_reference_elements() {
    for (m, k, n) in [
        (1, 1, 1),
        (2, 3, 4),
        (33, 35, 37),
        (17, 35, 129),
        (64, 65, 31),
        (97, 17, 35),
        (16, 128, 32),
    ] {
        let a = matrix(m, k, 7);
        let b = matrix(k, n, 3);
        let shape = vec![m, n];
        let reference = CpuBackend
            .forward(Op::Matmul, &a, Some(&b), &shape)
            .unwrap();
        let actual = TiledCpuBackend
            .forward(Op::Matmul, &a, Some(&b), &shape)
            .unwrap();
        assert_eq!(actual, reference, "{m},{k},{n}");
        let g = matrix(m, n, 11).data;
        assert_eq!(
            TiledCpuBackend.vjp(Op::Matmul, &a, Some(&b), &actual, &g),
            CpuBackend.vjp(Op::Matmul, &a, Some(&b), &reference, &g),
            "VJP {m},{k},{n}"
        );
    }
}
#[test]
fn tiled_gradients_match_independent_finite_differences_across_tile_boundaries() {
    let a = matrix(33, 35, 7);
    let b = matrix(35, 37, 3);
    let shape = vec![33, 37];
    let g = matrix(33, 37, 11).data;
    let out = TiledCpuBackend
        .forward(Op::Matmul, &a, Some(&b), &shape)
        .unwrap();
    let (ga, gb) = TiledCpuBackend.vjp(Op::Matmul, &a, Some(&b), &out, &g);
    let gb = gb.unwrap();
    for left in [true, false] {
        let length = if left { a.data.len() } else { b.data.len() };
        for index in [0, 31, 32, length / 2, length - 1] {
            let loss = |delta| {
                let mut a = a.clone();
                let mut b = b.clone();
                if left {
                    a.data[index] += delta;
                } else {
                    b.data[index] += delta;
                }
                TiledCpuBackend
                    .forward(Op::Matmul, &a, Some(&b), &shape)
                    .unwrap()
                    .data
                    .iter()
                    .zip(&g)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            };
            let numeric = (loss(1e-5) - loss(-1e-5)) / 2e-5;
            let analytic = if left { ga[index] } else { gb[index] };
            assert!(
                (numeric - analytic).abs() < 1e-7,
                "{left},{index}: {numeric} != {analytic}"
            );
        }
    }
}
#[test]
fn tiled_backend_delegates_shape_ops_and_checks_matmul_errors() {
    let a = matrix(2, 3, 1);
    for op in [
        Op::Tanh,
        Op::Softmax,
        Op::Transpose,
        Op::Rows { start: 1, count: 1 },
    ] {
        let shape = ml_lang::tensor::output_shape(op, &a.shape, None).unwrap();
        assert_eq!(
            TiledCpuBackend.forward(op, &a, None, &shape).unwrap(),
            CpuBackend.forward(op, &a, None, &shape).unwrap()
        );
    }
    assert!(TiledCpuBackend
        .forward(Op::Matmul, &a, None, &[2, 3])
        .is_err());
    assert!(TiledCpuBackend
        .forward(Op::Matmul, &a, Some(&a), &[2, 3])
        .is_err());
    let huge = Tensor::new(vec![33, 33], vec![1e200; 33 * 33]).unwrap();
    assert!(TiledCpuBackend
        .forward(Op::Matmul, &huge, Some(&huge), &[33, 33])
        .is_err());
}
#[test]
fn tiled_training_and_cross_backend_checkpoint_resume_match_exactly() {
    let source = |steps| {
        format!("param a=rand(33,35,42);param b=rand(35,37,123);repeat {steps}{{let z=a@b;backward(mean(z*z));clip_grad_norm(1);adam_step(0.01);}}print(sum(a));print(sum(b));")
    };
    let full = vm::execute_with_backend(&compile(&source(3)).unwrap(), &CpuBackend).unwrap();
    let tiled = vm::execute_with_backend(&compile(&source(3)).unwrap(), &TiledCpuBackend).unwrap();
    assert_eq!(full.output, tiled.output);
    assert_eq!(full.checkpoint, tiled.checkpoint);
    let first = vm::execute_with_backend(&compile(&source(1)).unwrap(), &CpuBackend).unwrap();
    let resumed = vm::execute_with_options(
        &compile(&source(2)).unwrap(),
        &TiledCpuBackend,
        Some(&first.checkpoint),
        false,
    )
    .unwrap();
    assert_eq!(full.checkpoint, resumed.checkpoint);
}
#[test]
fn tiled_ssa_inference_matches_tiled_vm_with_loaded_large_parameters() {
    let source="fn f(a:Tensor[B,35],b:Tensor[35,37])->Tensor[B,37]{return tanh(a@b);}param a=rand(33,35,42);param b=rand(35,37,123);let z=f(a,b);print(sum(z));";
    let p = compile(source).unwrap();
    let saved = vm::execute(&p).unwrap().checkpoint;
    let reference = vm::execute_with_options(&p, &CpuBackend, Some(&saved), true).unwrap();
    let actual = ml_lang::inference::execute_with_backend(
        &ml_lang::ir::lower(&p).unwrap(),
        Some(&saved),
        &TiledCpuBackend,
    )
    .unwrap();
    assert_eq!(actual.variables, reference.variables);
    assert_eq!(actual.output, reference.output);
    assert_eq!(actual.checkpoint, reference.checkpoint);
}
