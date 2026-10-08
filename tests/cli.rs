use std::{path::PathBuf, process::Command};
fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ml"))
}
fn example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}
#[test]
fn check_and_run() {
    for cmd in ["check", "run", "ir"] {
        let result = binary()
            .arg(cmd)
            .arg(example("matmul.ml"))
            .output()
            .unwrap();
        assert!(result.status.success(), "{:?}", result);
    }
}
#[test]
fn missing_file_fails() {
    let result = binary()
        .args(["run", "does-not-exist.ml"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("error:"));
}
#[test]
fn invalid_command_fails() {
    assert!(!binary()
        .arg("oops")
        .arg(example("matmul.ml"))
        .status()
        .unwrap()
        .success());
}
#[test]
fn benchmark_executes_requested_runs() {
    let result = binary()
        .arg("bench")
        .arg(example("matmul.ml"))
        .arg("2")
        .output()
        .unwrap();
    assert!(result.status.success());
    let s = String::from_utf8(result.stdout).unwrap();
    assert!(s.contains("\"iterations\":2"));
    assert!(s.contains("peak_retained_tensor_bytes"));
}

fn temporary_directory() -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let directory = std::env::temp_dir().join(format!(
        "ml-lang-cli-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    directory
}
#[test]
fn checkpoint_training_and_inference_use_separate_processes() {
    let directory = temporary_directory();
    let checkpoint = directory.join("model.mlckpt");
    let train = binary()
        .arg("run")
        .arg(example("adam_multiclass.ml"))
        .arg("--save")
        .arg(&checkpoint)
        .output()
        .unwrap();
    assert!(
        train.status.success(),
        "{}",
        String::from_utf8_lossy(&train.stderr)
    );
    let infer = binary()
        .arg("run")
        .arg(example("adam_infer.ml"))
        .arg("--load")
        .arg(&checkpoint)
        .output()
        .unwrap();
    assert!(
        infer.status.success(),
        "{}",
        String::from_utf8_lossy(&infer.stderr)
    );
    let trained = String::from_utf8(train.stdout).unwrap();
    let inferred = String::from_utf8(infer.stdout).unwrap();
    assert_eq!(trained.lines().last(), inferred.lines().next());
    let original = std::fs::read(&checkpoint).unwrap();
    let duplicate = binary()
        .arg("run")
        .arg(example("adam_multiclass.ml"))
        .arg("--save")
        .arg(&checkpoint)
        .output()
        .unwrap();
    assert!(!duplicate.status.success());
    assert_eq!(original, std::fs::read(&checkpoint).unwrap());
    std::fs::remove_dir_all(directory).unwrap();
}
#[test]
fn checkpoint_cli_rejects_missing_corrupt_and_ambiguous_options() {
    let directory = temporary_directory();
    let bad = directory.join("bad.mlckpt");
    std::fs::write(&bad, b"invalid").unwrap();
    let corrupt = binary()
        .arg("run")
        .arg(example("adam_infer.ml"))
        .arg("--load")
        .arg(&bad)
        .output()
        .unwrap();
    assert!(!corrupt.status.success());
    for options in [
        vec!["--load"],
        vec!["--unknown", "x"],
        vec!["--load", "x", "--load", "y"],
        vec!["--save", "x", "--save", "y"],
    ] {
        let out = binary()
            .arg("run")
            .arg(example("matmul.ml"))
            .args(options)
            .output()
            .unwrap();
        assert!(!out.status.success());
    }
    let missing = binary()
        .arg("run")
        .arg(example("adam_infer.ml"))
        .arg("--load")
        .arg(directory.join("missing.mlckpt"))
        .output()
        .unwrap();
    assert!(!missing.status.success());
    std::fs::remove_dir_all(directory).unwrap();
}
#[test]
fn checkpoint_options_are_not_accepted_by_check_or_bench() {
    for command in ["check", "bench", "ir"] {
        let out = binary()
            .arg(command)
            .arg(example("matmul.ml"))
            .args(["--load", "x"])
            .output()
            .unwrap();
        assert!(!out.status.success());
    }
}

#[test]
fn cli_resume_matches_uninterrupted_optimizer_state() {
    let directory = temporary_directory();
    let source = directory.join("train.ml");
    let full = directory.join("full.mlckpt");
    let first = directory.join("first.mlckpt");
    let resumed = directory.join("resumed.mlckpt");
    let program =
        |n: usize| format!("param x=[1,-2];repeat {n} {{backward(sum(x*x));adam_step(0.01);}}");
    std::fs::write(&source, program(20)).unwrap();
    assert!(binary()
        .arg("run")
        .arg(&source)
        .arg("--save")
        .arg(&full)
        .status()
        .unwrap()
        .success());
    std::fs::write(&source, program(7)).unwrap();
    assert!(binary()
        .arg("run")
        .arg(&source)
        .arg("--save")
        .arg(&first)
        .status()
        .unwrap()
        .success());
    std::fs::write(&source, program(13)).unwrap();
    assert!(binary()
        .arg("run")
        .arg(&source)
        .arg("--load")
        .arg(&first)
        .arg("--save")
        .arg(&resumed)
        .status()
        .unwrap()
        .success());
    assert_eq!(
        std::fs::read(&full).unwrap(),
        std::fs::read(&resumed).unwrap()
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn inference_flag_is_checked_and_runs_forward_programs() {
    assert!(binary()
        .arg("run")
        .arg(example("matmul.ml"))
        .arg("--inference")
        .status()
        .unwrap()
        .success());
    for (file, args) in [
        ("autodiff.ml", vec!["--inference"]),
        ("matmul.ml", vec!["--inference", "--save", "unused.mlckpt"]),
        ("matmul.ml", vec!["--inference", "--inference"]),
    ] {
        let result = binary()
            .arg("run")
            .arg(example(file))
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
    }
}

#[test]
fn inference_plan_and_optimized_run_are_available() {
    let plan = binary()
        .arg("plan")
        .arg(example("inference_memory.ml"))
        .output()
        .unwrap();
    assert!(plan.status.success());
    let dump = String::from_utf8(plan.stdout).unwrap();
    assert!(dump.contains("Tensor<f64,cpu>[256, 256]"));
    assert!(dump.contains("release"));
    let reference = binary()
        .arg("run")
        .arg(example("inference_memory.ml"))
        .arg("--inference")
        .output()
        .unwrap();
    let actual = binary()
        .arg("run")
        .arg(example("inference_memory.ml"))
        .args(["--inference", "--optimize"])
        .output()
        .unwrap();
    assert!(reference.status.success());
    assert!(actual.status.success());
    assert_eq!(reference.stdout, actual.stdout);
    let directory = temporary_directory();
    let checkpoint = directory.join("model.mlckpt");
    let train = binary()
        .arg("run")
        .arg(example("minibatch.ml"))
        .arg("--save")
        .arg(&checkpoint)
        .output()
        .unwrap();
    assert!(train.status.success());
    let actual = binary()
        .arg("run")
        .arg(example("adam_infer.ml"))
        .arg("--load")
        .arg(&checkpoint)
        .args(["--inference", "--optimize"])
        .output()
        .unwrap();
    assert!(actual.status.success());
    std::fs::remove_dir_all(directory).unwrap();
}
#[test]
fn optimized_benchmark_reports_separate_lowering_and_execution_mode() {
    let actual = binary()
        .arg("bench")
        .arg(example("matmul.ml"))
        .args(["--inference", "2", "--optimize"])
        .output()
        .unwrap();
    assert!(actual.status.success());
    let json = String::from_utf8(actual.stdout).unwrap();
    assert!(json.contains("\"execution_mode\":\"inference-ssa\""));
    assert!(json.contains("\"lower_seconds\":"));
    assert!(json.contains("\"iterations\":2"));
}
#[test]
fn inference_optimization_flags_and_effects_are_checked() {
    for (command, file, args) in [
        ("run", "matmul.ml", vec!["--optimize"]),
        (
            "run",
            "matmul.ml",
            vec!["--inference", "--optimize", "--optimize"],
        ),
        ("run", "autodiff.ml", vec!["--inference", "--optimize"]),
        ("bench", "matmul.ml", vec!["--optimize"]),
        ("bench", "matmul.ml", vec!["--inference", "0"]),
        ("bench", "matmul.ml", vec!["2", "3"]),
        ("bench", "matmul.ml", vec!["--inference", "--inference"]),
        ("plan", "autodiff.ml", vec![]),
        ("plan", "matmul.ml", vec!["--optimize"]),
    ] {
        let result = binary()
            .arg(command)
            .arg(example(file))
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success(), "{command} {file}");
        assert!(result.stdout.is_empty());
    }
}
#[test]
fn backend_selection_is_supported_by_training_and_optimized_inference() {
    for flags in [
        vec!["--backend", "tiled"],
        vec!["--backend", "tiled", "--inference", "--optimize"],
    ] {
        let actual = binary()
            .arg("run")
            .arg(example("matmul.ml"))
            .args(flags)
            .output()
            .unwrap();
        assert!(actual.status.success());
    }
    let report = binary()
        .arg("bench")
        .arg(example("matmul.ml"))
        .args(["2", "--backend", "tiled", "--inference", "--optimize"])
        .output()
        .unwrap();
    assert!(report.status.success());
    assert!(String::from_utf8(report.stdout)
        .unwrap()
        .contains("\"backend\":\"tiled\""));
    for args in [
        vec!["--backend"],
        vec!["--backend", "cuda"],
        vec!["--backend", "tiled", "--backend", "cpu"],
    ] {
        let actual = binary()
            .arg("run")
            .arg(example("matmul.ml"))
            .args(args)
            .output()
            .unwrap();
        assert!(!actual.status.success());
    }
}
