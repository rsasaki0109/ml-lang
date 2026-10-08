//! Independent frontend, shape-checked bytecode compiler, and CPU tensor runtime.
pub mod backend;
pub mod checkpoint;
pub mod compiler;
pub mod inference;
pub mod ir;
pub mod optimizer;
pub mod syntax;
pub mod tensor;
pub mod tiled;
mod typecheck;
pub mod vm;

pub fn compile(source: &str) -> Result<compiler::Program, String> {
    compiler::compile(syntax::parse(source)?)
}
pub fn run(source: &str) -> Result<vm::Execution, String> {
    vm::execute(&compile(source)?)
}
