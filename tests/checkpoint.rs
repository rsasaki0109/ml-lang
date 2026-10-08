use ml_lang::{checkpoint::Checkpoint, compile, run, vm::execute_with_checkpoint};
#[test]
fn checkpoint_round_trip_preserves_weights_and_optimizer_bits() {
    let r = run("param x=[1,-2];repeat 5 {backward(sum(x*x));adam_step(0.01);}").unwrap();
    let bytes = r.checkpoint.encode().unwrap();
    let decoded = Checkpoint::decode(&bytes).unwrap();
    assert_eq!(decoded, r.checkpoint);
    assert_eq!(decoded.encode().unwrap(), bytes);
}
#[test]
fn adam_resume_matches_uninterrupted_training_exactly() {
    let source =
        |n: usize| format!("param x=[1,-2];repeat {n} {{backward(sum(x*x));adam_step(0.01);}}");
    let full = run(&source(20)).unwrap();
    let first = run(&source(7)).unwrap();
    let checkpoint = Checkpoint::decode(&first.checkpoint.encode().unwrap()).unwrap();
    let resumed = execute_with_checkpoint(&compile(&source(13)).unwrap(), &checkpoint).unwrap();
    assert_eq!(full.variables["x"], resumed.variables["x"]);
    assert_eq!(full.checkpoint, resumed.checkpoint);
}
#[test]
fn resumed_inference_uses_new_batch_specialization() {
    let trained = run(include_str!("../examples/adam_multiclass.ml")).unwrap();
    let program = compile(include_str!("../examples/adam_infer.ml")).unwrap();
    let inferred = execute_with_checkpoint(&program, &trained.checkpoint).unwrap();
    assert_eq!(inferred.output[0], trained.output[3]);
}
#[test]
fn mismatched_names_and_shapes_fail_before_inference() {
    let saved = run("param x=[1,2];").unwrap().checkpoint;
    for s in [
        "param y=[1,2];",
        "param x=[1,2,3];",
        "param x=[1,2];param y=1;",
        "let x=[1,2];",
    ] {
        assert!(execute_with_checkpoint(&compile(s).unwrap(), &saved).is_err());
    }
}
#[test]
fn malformed_and_corrupt_files_are_rejected() {
    let valid = run("param x=[1,2];").unwrap().checkpoint.encode().unwrap();
    for len in 0..valid.len() {
        assert!(Checkpoint::decode(&valid[..len]).is_err());
    }
    let mut corrupt = valid.clone();
    corrupt[15] ^= 1;
    assert!(Checkpoint::decode(&corrupt)
        .unwrap_err()
        .contains("checksum"));
    let mut unknown = valid;
    unknown[7] = b'2';
    assert!(Checkpoint::decode(&unknown)
        .unwrap_err()
        .contains("version"));
}
#[test]
fn invalid_in_memory_states_cannot_be_saved() {
    let good = run("param x=[1,2];backward(sum(x*x));adam_step(0.01);")
        .unwrap()
        .checkpoint;
    let mut bad = good.clone();
    bad.adam.get_mut("x").unwrap().second[0] = -1.;
    assert!(bad.encode().is_err());
    let mut bad = good.clone();
    bad.parameters.get_mut("x").unwrap().data[0] = f64::NAN;
    assert!(bad.encode().is_err());
    let mut bad = good.clone();
    bad.adam_config = None;
    assert!(bad.encode().is_err());
    let mut bad = good;
    bad.adam.get_mut("x").unwrap().step = 0;
    assert!(bad.encode().is_err());
}
#[test]
fn checksum_valid_payloads_still_require_well_formed_metadata() {
    let mut bytes = run("param x=[1,2];").unwrap().checkpoint.encode().unwrap();
    // Parameter count begins after magic and optional-config flag.
    bytes[9..13].copy_from_slice(&u32::MAX.to_le_bytes());
    let n = bytes.len() - 8;
    let hash = bytes[..n].iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    bytes[n..].copy_from_slice(&hash.to_le_bytes());
    assert!(Checkpoint::decode(&bytes).unwrap_err().contains("too many"));
}

#[test]
fn file_save_load_and_no_overwrite() {
    let directory = std::env::temp_dir().join(format!("ml-lang-file-test-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("model.mlckpt");
    let checkpoint = run("param x=[1,2];backward(sum(x*x));adam_step(0.01);")
        .unwrap()
        .checkpoint;
    checkpoint.save(&path).unwrap();
    assert_eq!(Checkpoint::load(&path).unwrap(), checkpoint);
    let original = std::fs::read(&path).unwrap();
    assert!(checkpoint.save(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let user_file = directory.join("user.txt");
    std::fs::write(&user_file, b"preserve").unwrap();
    assert!(checkpoint.save(&user_file).is_err());
    assert_eq!(std::fs::read(&user_file).unwrap(), b"preserve");
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 2);
    std::fs::remove_dir_all(directory).unwrap();
}
#[test]
fn checked_decoding_rejects_invalid_ranks_sizes_and_flags() {
    let original = run("param x=[1,2];").unwrap().checkpoint.encode().unwrap();
    for (offset, replacement) in [
        (8, vec![2]),
        (13, u32::MAX.to_le_bytes().to_vec()),
        (18, 9u32.to_le_bytes().to_vec()),
        (22, 0u64.to_le_bytes().to_vec()),
        (30, u64::MAX.to_le_bytes().to_vec()),
    ] {
        let mut bytes = original.clone();
        bytes[offset..offset + replacement.len()].copy_from_slice(&replacement);
        let n = bytes.len() - 8;
        let hash = bytes[..n].iter().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        });
        bytes[n..].copy_from_slice(&hash.to_le_bytes());
        assert!(Checkpoint::decode(&bytes).is_err());
    }
}

#[test]
fn explicit_inference_disables_gradients_and_optimizer_state() {
    let trained = run(include_str!("../examples/adam_multiclass.ml")).unwrap();
    let program = compile(include_str!("../examples/adam_infer.ml")).unwrap();
    let inferred = ml_lang::vm::execute_inference(&program, Some(&trained.checkpoint)).unwrap();
    assert_eq!(inferred.output[0], trained.output[3]);
    assert!(inferred.checkpoint.adam.is_empty());
    assert!(inferred.checkpoint.adam_config.is_none());
    let forward_loop = |iterations| {
        let source = format!(
            "param x=[1,2];repeat {iterations} {{let y=x*x;assert_close(y,[1,4],0);}}print(x);"
        );
        ml_lang::vm::execute_inference(&compile(&source).unwrap(), None).unwrap()
    };
    let short = forward_loop(2);
    let long = forward_loop(2000);
    assert_eq!(short.output, long.output);
    assert_eq!(short.peak_tensor_bytes, long.peak_tensor_bytes);
    assert!(long.checkpoint.adam.is_empty());
    for source in [
        "param x=1;backward(x*x);",
        "param x=1;let g=grad(x);",
        "param x=1;repeat 0 {backward(x*x);adam_step(0.1);}",
    ] {
        assert!(ml_lang::vm::execute_inference(&compile(source).unwrap(), None).is_err());
    }
}
