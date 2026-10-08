use ml_lang::{compile, inference, ir, vm};
fn agree(source: &str) {
    let program = compile(source).unwrap();
    let planned = ir::lower(&program).unwrap();
    let reference = vm::execute_inference(&program, None).unwrap();
    for _ in 0..2 {
        let actual = inference::execute(&planned, None).unwrap();
        assert_eq!(actual.output, reference.output);
        assert_eq!(actual.variables, reference.variables);
        assert_eq!(actual.checkpoint, reference.checkpoint);
    }
}
#[test]
fn primitive_and_composed_forward_values_match_vm() {
    for source in [
        include_str!("../examples/matmul.ml"),
        "param x=[[0.2,-0.4],[0.7,0.3]];let a=x+[[1,2]];let b=x*x-x/2;let c=tanh(a);let d=relu(b);let e=sigmoid(x);let f=transpose(x);let g=reshape(f,4);let h=rows(x,1,1);let i=sum(x);let j=mean(x);let k=softmax(x);let l=bce_logits(x,[[0,1],[1,0]]);let m=cross_entropy(x,[[0,1],[1,0]]);let n=mse(x,x);let o=detach(x);print(k);print(m);",
        "fn f(x:Tensor[B,2])->Tensor[B,3]{let h=tanh(x@rand(2,3,42));return softmax(h);}let a=f([[1,2]]);let b=f([[1,2],[3,4]]);print(a);print(b);",
    ] { agree(source); }
}
#[test]
fn nested_regions_keep_captures_and_release_iteration_values() {
    agree("param x=rand(2,2,42);let alias=x;repeat 3{let y=tanh(x);repeat 2{let z=rows(y,0,1);print(z);assert_close(z,z,0);}let other=x+y;print(other);}print(alias);");
    agree("let x=ones(2,2);repeat 0{let y=x/0;print(y);}repeat 2{repeat 0{print(x/0);}print(x);}let final=x+1;");
}
#[test]
fn duplicate_operands_aliases_and_snapshot_outputs_match() {
    agree("param x=[1,2];let a=x;let b=detach(x);let c=detach(a)*b+x*x;print(c);print(c);assert_close(a,b,0);");
    agree(
        "fn f(x:Tensor[2])->Tensor[]{let a=x*x;let b=a+a;return sum(b);}let y=f([1,2]);print(y);",
    );
}
#[test]
fn checkpoint_model_inference_matches_reference_and_ignores_adam() {
    let trained = ml_lang::run(include_str!("../examples/minibatch.ml")).unwrap();
    let program = compile(include_str!("../examples/adam_infer.ml")).unwrap();
    let reference = vm::execute_inference(&program, Some(&trained.checkpoint)).unwrap();
    let planned = ir::lower(&program).unwrap();
    let actual = inference::execute(&planned, Some(&trained.checkpoint)).unwrap();
    assert_eq!(actual.output, reference.output);
    assert_eq!(actual.variables, reference.variables);
    assert_eq!(actual.checkpoint, reference.checkpoint);
    assert!(actual.checkpoint.adam.is_empty());
    let wrong = ml_lang::run("param other=1;").unwrap();
    assert!(inference::execute(&planned, Some(&wrong.checkpoint)).is_err());
    let mut wrong = trained.checkpoint;
    wrong.parameters.get_mut("w1").unwrap().shape = vec![4, 4];
    assert!(inference::execute(&planned, Some(&wrong)).is_err());
}
#[test]
fn last_use_plan_reduces_temporary_payload_without_changing_outputs() {
    let source="param x=ones(128,128);let result=mean(tanh(tanh(tanh(tanh(tanh(tanh(x)))))));print(result);";
    let program = compile(source).unwrap();
    let actual = inference::execute(&ir::lower(&program).unwrap(), None).unwrap();
    let reference = vm::execute_inference(&program, None).unwrap();
    assert_eq!(actual.output, reference.output);
    assert_eq!(actual.variables, reference.variables);
    assert!(
        actual.peak_tensor_bytes < reference.peak_tensor_bytes / 2,
        "{} vs {}",
        actual.peak_tensor_bytes,
        reference.peak_tensor_bytes
    );
}
#[test]
fn region_payload_is_bounded_with_no_printed_loop_values() {
    let source = |n| {
        format!("param x=ones(8,8);repeat {n} {{let y=tanh(tanh(x));assert_close(y,y,0);}}print(sum(x));")
    };
    let execute =
        |n| inference::execute(&ir::lower(&compile(&source(n)).unwrap()).unwrap(), None).unwrap();
    let short = execute(2);
    let long = execute(2000);
    assert_eq!(short.output, long.output);
    assert_eq!(short.peak_tensor_bytes, long.peak_tensor_bytes);
}
#[test]
fn lowering_rejects_training_effects_even_in_zero_loops() {
    for source in [
        "param x=1;backward(x*x);",
        "param x=1;let g=grad(x);",
        "repeat 0{step(0.1);}",
        "repeat 0{adam_step(0.1);}",
        "repeat 0{clip_grad_norm(1);}",
    ] {
        assert!(ir::lower(&compile(source).unwrap()).is_err());
    }
}
#[test]
fn last_use_reclamation_preserves_unused_operation_errors_and_print_failures() {
    for source in [
        "fn f(x:Tensor[])->Tensor[]{let bad=x/0;return x;}let y=f(1);",
        "let x=1;print(x);assert_close(x,2,0);",
        "let x=bce_logits([1],[2]);",
        "let x=1;assert_close(x,x,-1);",
    ] {
        let program = compile(source).unwrap();
        assert!(vm::execute_inference(&program, None).is_err());
        assert!(inference::execute(&ir::lower(&program).unwrap(), None).is_err());
    }
}
#[test]
fn typed_dump_is_repeatable_and_contains_captures_and_releases() {
    let source = "param x=ones(2,2);repeat 2{let y=tanh(x);print(y);}let z=sum(x);";
    let first = ir::lower(&compile(source).unwrap()).unwrap();
    let second = ir::lower(&compile(source).unwrap()).unwrap();
    assert_eq!(first.display(), second.display());
    assert!(first.display().contains("Tensor<f64,cpu>[2, 2]"));
    assert!(first.display().contains("captures="));
    assert!(first.display().contains("release"));
    assert!(first.value_count() > 0);
}
#[test]
fn malformed_bytecode_is_rejected_before_execution() {
    use ml_lang::compiler::Instruction;
    let mut p = compile("let x=1;").unwrap();
    p.code.insert(0, Instruction::Load(999));
    assert!(ir::lower(&p).is_err());
    let mut p = compile("let x=1;").unwrap();
    p.code
        .push(Instruction::Constant(ml_lang::tensor::Tensor::scalar(1.)));
    assert!(ir::lower(&p).is_err());
    let mut p = compile("let x=1;").unwrap();
    p.code.extend([
        Instruction::Constant(ml_lang::tensor::Tensor::scalar(2.)),
        Instruction::Store(0, false),
    ]);
    assert!(ir::lower(&p).is_err());
    let mut p = compile("let x=1;").unwrap();
    p.bindings.insert("bad".into(), 999);
    assert!(ir::lower(&p).is_err());
}
