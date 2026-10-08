use std::{env, fs, process, time::Instant};
fn main() {
    if let Err(e) = command() {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
fn command() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" {
        println!("ml-lang 0.1.0\nUsage: ml <run|check|ir|plan|bench> <file.ml> [iterations]\n       ml run <file.ml> [--load checkpoint] [--save checkpoint] [--inference [--optimize]] [--backend cpu|tiled]\n  run    Compile and execute on the CPU\n  check  Parse and statically check tensor shapes\n  ir     Print typed stack bytecode\n  plan   Print typed inference SSA and last-use releases\n  bench  Accepts [iterations] [--inference [--optimize]] [--backend cpu|tiled]; warm up, then time repeated executions (default: 5)");
        return Ok(());
    }
    if args.len() < 2 {
        return Err("expected command and source file; see --help".into());
    }
    if !["run", "check", "ir", "plan", "bench"].contains(&args[0].as_str()) {
        return Err("unknown command; see --help".into());
    }
    let mut inference = false;
    let mut optimize = false;
    let mut backend_name = None;
    let mut load = None;
    let mut save = None;
    let mut iterations = None;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--inference" if ["run", "bench"].contains(&args[0].as_str()) && !inference => {
                inference = true;
            }
            "--optimize" if ["run", "bench"].contains(&args[0].as_str()) && !optimize => {
                optimize = true;
            }
            "--backend"
                if ["run", "bench"].contains(&args[0].as_str()) && backend_name.is_none() =>
            {
                let name = args
                    .get(index + 1)
                    .ok_or("--backend requires cpu or tiled")?;
                if !["cpu", "tiled"].contains(&name.as_str()) {
                    return Err("unsupported backend; expected cpu or tiled".into());
                }
                backend_name = Some(name.clone());
                index += 1;
            }
            "--load" | "--save" if args[0] == "run" => {
                let value = args
                    .get(index + 1)
                    .filter(|path| !path.starts_with("--"))
                    .ok_or("checkpoint option requires a path")?
                    .clone();
                let destination = if args[index] == "--load" {
                    &mut load
                } else {
                    &mut save
                };
                if destination.replace(value).is_some() {
                    return Err("duplicate checkpoint option".into());
                }
                index += 1;
            }
            count if args[0] == "bench" && iterations.is_none() && !count.starts_with("--") => {
                iterations = Some(
                    count
                        .parse::<usize>()
                        .map_err(|_| "invalid iteration count")?,
                );
            }
            _ => return Err("unknown, duplicate or unexpected command argument".into()),
        }
        index += 1;
    }
    if optimize && !inference {
        return Err("--optimize requires --inference; SSA training is not implemented".into());
    }
    if inference && save.is_some() {
        return Err("--save is not supported in inference mode; use a training run to preserve optimizer state".into());
    }
    let backend_name = backend_name.as_deref().unwrap_or("cpu");
    let backend: &dyn ml_lang::backend::Backend = if backend_name == "tiled" {
        &ml_lang::tiled::TiledCpuBackend
    } else {
        &ml_lang::backend::CpuBackend
    };
    let source = fs::read_to_string(&args[1]).map_err(|e| format!("{}: {e}", args[1]))?;
    let start = Instant::now();
    let program = ml_lang::compile(&source)?;
    let compile_seconds = start.elapsed().as_secs_f64();
    let lower_start = Instant::now();
    let planned = if optimize || args[0] == "plan" {
        Some(ml_lang::ir::lower(&program)?)
    } else {
        None
    };
    let lower_seconds = if planned.is_some() {
        lower_start.elapsed().as_secs_f64()
    } else {
        0.0
    };
    match args[0].as_str() {
        "check" => println!("OK: parsed and shape-checked {}", args[1]),
        "ir" => println!("{:#?}", program.code),
        "plan" => print!("{}", planned.as_ref().unwrap().display()),
        "run" => {
            let saved = load
                .map(ml_lang::checkpoint::Checkpoint::load)
                .transpose()?;
            let result = if let Some(planned) = &planned {
                ml_lang::inference::execute_with_backend(planned, saved.as_ref(), backend)
            } else {
                ml_lang::vm::execute_with_options(&program, backend, saved.as_ref(), inference)
            }?;
            if let Some(path) = save {
                result.checkpoint.save(path)?;
            }
            for tensor in result.output {
                println!("Tensor{:?} {:?}", tensor.shape, tensor.data);
            }
        }
        "bench" => {
            let iterations = iterations.unwrap_or(5);
            if iterations == 0 || iterations > 10000 {
                return Err("benchmark iterations must be 1..10000".into());
            }
            let execute = || {
                if let Some(planned) = &planned {
                    ml_lang::inference::execute_with_backend(planned, None, backend)
                } else if inference {
                    ml_lang::vm::execute_with_options(&program, backend, None, true)
                } else {
                    ml_lang::vm::execute_with_options(&program, backend, None, false)
                }
            };
            let warm = execute()?;
            let mut samples = vec![];
            let mut peak = warm.peak_tensor_bytes;
            drop(warm);
            for _ in 0..iterations {
                let start = Instant::now();
                let result = execute()?;
                samples.push(start.elapsed().as_secs_f64());
                peak = peak.max(result.peak_tensor_bytes);
                std::hint::black_box(result);
            }
            samples.sort_by(f64::total_cmp);
            let median = if iterations % 2 == 0 {
                (samples[iterations / 2 - 1] + samples[iterations / 2]) / 2.0
            } else {
                samples[iterations / 2]
            };
            let mode = if optimize {
                "inference-ssa"
            } else if inference {
                "inference-vm"
            } else {
                "vm"
            };
            let rss = peak_rss().map_or("null".into(), |n| n.to_string());
            println!("{{\"backend\":\"{backend_name}\",\"execution_mode\":\"{mode}\",\"lower_seconds\":{lower_seconds},\"compile_seconds\":{compile_seconds},\"median_seconds\":{median},\"min_seconds\":{},\"iterations\":{iterations},\"peak_retained_tensor_bytes\":{peak},\"peak_rss_bytes\":{rss}}}",samples[0]);
        }
        _ => return Err("unknown command; see --help".into()),
    }
    Ok(())
}

fn peak_rss() -> Option<usize> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace()
        .nth(1)?
        .parse::<usize>()
        .ok()?
        .checked_mul(1024)
}
