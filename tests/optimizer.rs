use ml_lang::{
    optimizer::{Adam, AdamConfig},
    run,
    tensor::Tensor,
};
#[test]
fn adam_first_step_and_constant_gradient_trajectory() {
    let mut adam = Adam::default();
    let mut x = Tensor::new(vec![2], vec![1., -2.]).unwrap();
    for _ in 0..10 {
        x = adam
            .update(&[(0, x, vec![2., -4.])], 0.1, AdamConfig::default())
            .unwrap()
            .remove(0)
            .1;
    }
    let eps = 1e-8;
    assert!((x.data[0] - (1. - 10. * 0.1 * 2. / (2. + eps))).abs() < 1e-13);
    assert!((x.data[1] - (-2. + 10. * 0.1 * 4. / (4. + eps))).abs() < 1e-13);
    assert_eq!(adam.states[&0].step, 10);
}
#[test]
fn adam_tracks_time_per_connected_parameter() {
    let mut adam = Adam::default();
    let x = Tensor::scalar(1.);
    adam.update(&[(0, x.clone(), vec![1.])], 0.1, AdamConfig::default())
        .unwrap();
    adam.update(&[(1, x, vec![1.])], 0.1, AdamConfig::default())
        .unwrap();
    assert_eq!(adam.states[&0].step, 1);
    assert_eq!(adam.states[&1].step, 1);
}
#[test]
fn adam_updates_are_atomic_on_numerical_failure() {
    let mut adam = Adam::default();
    assert!(adam
        .update(
            &[
                (0, Tensor::scalar(1.), vec![1.]),
                (1, Tensor::scalar(1.), vec![1e200])
            ],
            0.1,
            AdamConfig::default()
        )
        .is_err());
    assert!(adam.states.is_empty());
    assert!(adam.config.is_none());
}
#[test]
fn adam_hyperparameters_and_gradient_sizes_are_validated() {
    for config in [
        AdamConfig {
            beta1: 1.,
            ..Default::default()
        },
        AdamConfig {
            beta2: -0.1,
            ..Default::default()
        },
        AdamConfig {
            epsilon: 0.,
            ..Default::default()
        },
    ] {
        assert!(Adam::default().update(&[], 0.1, config).is_err());
    }
    assert!(Adam::default()
        .update(
            &[(0, Tensor::scalar(1.), vec![])],
            0.1,
            AdamConfig::default()
        )
        .is_err());
    assert!(Adam::default()
        .update(&[], 0., AdamConfig::default())
        .is_err());
    let mut adam = Adam::default();
    adam.update(
        &[(0, Tensor::scalar(1.), vec![1.])],
        0.1,
        AdamConfig::default(),
    )
    .unwrap();
    let states = adam.states.clone();
    assert!(adam
        .update(
            &[(0, Tensor::scalar(1.), vec![1.])],
            0.1,
            AdamConfig {
                beta1: 0.8,
                ..Default::default()
            }
        )
        .is_err());
    assert_eq!(adam.states, states);
}
#[test]
fn adam_language_training_and_memory_are_bounded() {
    let source = |n: usize| {
        format!("param x=[1,2];repeat {n} {{let loss=sum(x*x);backward(loss);adam_step(0.01);}}")
    };
    let short = run(&source(3)).unwrap();
    let long = run(&source(3000)).unwrap();
    assert_eq!(short.peak_tensor_bytes, long.peak_tensor_bytes);
    assert!(long.variables["x"].data.iter().all(|x| x.abs() < 1e-5));
    assert_eq!(long.checkpoint.adam["x"].step, 3000);
}
#[test]
fn adam_statement_rejects_missing_backward_and_invalid_options() {
    for s in [
        "param x=1;adam_step(0.1);",
        "param x=1;backward(x*x);adam_step(-0.1);",
        "param x=1;backward(x*x);adam_step(0.1,1,0.999,1e-8);",
        "param x=1;backward(x*x);adam_step(0.1,0.9,0.999,0);",
        "param x=1;backward(x*x);adam_step(0.1);adam_step(0.1);",
    ] {
        assert!(run(s).is_err());
    }
    for s in [
        "adam_step(0.1,0.9);",
        "adam_step([0.1]);",
        "adam_step(0.1,0.9,[0.999],1e-8);",
    ] {
        assert!(ml_lang::compile(s).is_err());
    }
}
#[test]
fn adam_custom_hyperparameters_and_unconnected_leaves() {
    let r = run("param x=1;param unused=3;backward(x*x);adam_step(0.1,0,0,1e-8);").unwrap();
    assert!((r.variables["x"].data[0] - 0.9).abs() < 1e-9);
    assert_eq!(r.variables["unused"].data, vec![3.]);
    assert!(!r.checkpoint.adam.contains_key("unused"));
}
#[test]
fn adam_multiclass_model_trains_and_generalizes_to_toy_queries() {
    let r = run(include_str!("../examples/adam_multiclass.ml")).unwrap();
    assert_eq!(r.output.len(), 4);
    assert!(r.output[1].data[0] < 0.01);
    assert_eq!(r.output[3].shape, vec![3, 3]);
}
