#!/usr/bin/env python3
"""Isolated-process CPU f64 benchmarks. Standard-library-only controller.

Install requirements.txt in a virtual environment, build cargo --release,
then run: python benchmarks/compare.py --samples 5 --output /tmp/results.json
"""
import argparse
import ast
import json
import hashlib
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MASK = (1 << 64) - 1


def random_values(n, seed):
    values = []
    state = seed
    for _ in range(n):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
        values.append((z >> 11) / (1 << 53) * 2 - 1)
    return values


def torch_workload(workload, size):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dtype = torch.float64
    if workload == "matmul":
        def run():
            a, b = torch.ones((size, size), dtype=dtype), torch.ones((size, size), dtype=dtype)
            results = [float((a @ b).sum()) for _ in range(10)]
            assert all(x == size ** 3 for x in results)
            return results
        return run, torch.__version__
    binary = workload == "binary-xor"
    minibatch = workload == "minibatch"
    adam = workload in ("adam-multiclass", "minibatch")
    multiclass = workload in ("multiclass", "adam-multiclass", "minibatch")
    outputs = 3 if multiclass else 1
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=dtype)
    y = torch.tensor([[-1.], [1.], [1.], [-1.]], dtype=dtype)
    if binary:
        y = (y + 1) / 2
    if multiclass:
        x = torch.tensor([[-1,-1],[-0.8,-1.2],[1,-1],[1.2,-0.8],[0,1],[0.2,1.2]],dtype=dtype)
        y = torch.tensor([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]],dtype=dtype)
    def run():
        w1 = torch.tensor(random_values(16, 42), dtype=dtype).reshape(2, 8).requires_grad_()
        b1 = torch.zeros((1, 8), dtype=dtype, requires_grad=True)
        w2 = torch.tensor(random_values(8*outputs, 123), dtype=dtype).reshape(8, outputs).requires_grad_()
        b2 = torch.zeros((1, outputs), dtype=dtype, requires_grad=True)
        params = [w1, b1, w2, b2]
        optimizer = torch.optim.Adam(params,lr=0.03,betas=(0.9,0.999),eps=1e-8,foreach=False,fused=False) if adam else None
        def predict(data=x):
            z = torch.tanh(data @ w1 + b1) @ w2 + b2
            return z if binary or multiclass else torch.tanh(z)
        def objective(z,target=y):
            if multiclass:
                return torch.nn.functional.cross_entropy(z,target)
            return torch.nn.functional.binary_cross_entropy_with_logits(z,target) if binary else ((z-target)**2).mean()
        initial = float(objective(predict()).detach())
        for iteration in range(300 if adam else 1000 if multiclass else 2000):
            if minibatch:
                start = (iteration % 3) * 2
                loss = objective(predict(x[start:start+2]), y[start:start+2])
            else:
                loss = objective(predict())
            loss.backward()
            if minibatch:
                # Exact epsilon-free global clipping policy, unlike torch's +1e-6 default.
                norm = torch.sqrt(sum(p.grad.square().sum() for p in params))
                factor = 1.0 / norm.clamp_min(1.0)
                with torch.no_grad():
                    for p in params:
                        p.grad.mul_(factor)
            if adam:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    for p in params:
                        p -= 0.1 * p.grad
                        p.grad = None
        z = predict().detach()
        final = float(objective(z))
        prediction = torch.softmax(z,dim=-1) if multiclass else torch.sigmoid(z) if binary else z
        assert float((prediction - y).abs().max()) < (0.02 if binary or multiclass else 0.06)
        result = [initial, final, prediction.flatten().tolist()]
        if adam:
            query = torch.tensor([[-1.1,-0.9],[0.9,-1.1],[0.1,0.9]],dtype=dtype)
            prob = torch.softmax(torch.tanh(query@w1+b1)@w2+b2,dim=-1).detach()
            target=torch.eye(3,dtype=dtype)
            assert float((prob-target).abs().max())<0.03
            result.extend([prob.flatten().tolist(),[float(v) for p in params for v in p.detach().flatten()]])
        return result
    return run, torch.__version__


def jax_workload(workload, size):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    if workload == "matmul":
        kernel = jax.jit(lambda a, b: jnp.sum(a @ b))
        def run():
            a, b = jnp.ones((size, size), dtype=jnp.float64), jnp.ones((size, size), dtype=jnp.float64)
            results = [float(kernel(a, b).block_until_ready()) for _ in range(10)]
            assert all(v == size ** 3 for v in results)
            return results
        return run, jax.__version__
    binary = workload == "binary-xor"
    minibatch = workload == "minibatch"
    adam = workload in ("adam-multiclass", "minibatch")
    multiclass = workload in ("multiclass", "adam-multiclass", "minibatch")
    outputs = 3 if multiclass else 1
    x = jnp.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=jnp.float64)
    y = jnp.array([[-1.], [1.], [1.], [-1.]], dtype=jnp.float64)
    if binary:
        y = (y + 1) / 2
    if multiclass:
        x = jnp.array([[-1,-1],[-0.8,-1.2],[1,-1],[1.2,-0.8],[0,1],[0.2,1.2]],dtype=jnp.float64)
        y = jnp.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]],dtype=jnp.float64)
    def predict(p,data=x):
        w1, b1, w2, b2 = p
        z = jnp.tanh(data @ w1 + b1) @ w2 + b2
        return z if binary or multiclass else jnp.tanh(z)
    def loss(p,data=x,target=y):
        z = predict(p,data)
        if multiclass:
            return -jnp.mean(jnp.sum(target*jax.nn.log_softmax(z,axis=-1),axis=-1))
        return jnp.mean(jax.nn.softplus(z) - z*target) if binary else jnp.mean((z-target)**2)
    gradient = jax.grad(loss)
    @jax.jit
    def train(params):
        def step(_, p):
            g = gradient(p)
            return tuple(a - 0.1*b for a, b in zip(p, g))
        if adam:
            zero = tuple(jnp.zeros_like(p) for p in params)
            def adam_update(i,state):
                p,m,v=state
                if minibatch:
                    start = (i % 3) * 2
                    xb = jax.lax.dynamic_slice_in_dim(x,start,2,axis=0)
                    yb = jax.lax.dynamic_slice_in_dim(y,start,2,axis=0)
                    g = gradient(p,xb,yb)
                    norm = jnp.sqrt(sum(jnp.sum(a*a) for a in g))
                    factor = 1.0 / jnp.maximum(norm,1.0)
                    g = tuple(a * factor for a in g)
                else:
                    g = gradient(p)
                m=tuple(0.9*a+0.1*b for a,b in zip(m,g))
                v=tuple(0.999*a+0.001*b*b for a,b in zip(v,g))
                t=i+1
                p=tuple(a-0.03*(b/(1-0.9**t))/(jnp.sqrt(c/(1-0.999**t))+1e-8) for a,b,c in zip(p,m,v))
                return p,m,v
            return jax.lax.fori_loop(0,300,adam_update,(params,zero,zero))[0]
        return jax.lax.fori_loop(0, 1000 if multiclass else 2000, step, params)
    def run():
        p = (jnp.array(random_values(16, 42), dtype=jnp.float64).reshape(2, 8),
             jnp.zeros((1, 8), dtype=jnp.float64),
             jnp.array(random_values(8*outputs, 123), dtype=jnp.float64).reshape(8, outputs),
             jnp.zeros((1, outputs), dtype=jnp.float64))
        initial = float(loss(p).block_until_ready())
        p = train(p)
        z = predict(p)
        prediction = (jax.nn.softmax(z,axis=-1) if multiclass else jax.nn.sigmoid(z) if binary else z).block_until_ready()
        final = float(loss(p).block_until_ready())
        assert float(jnp.max(jnp.abs(prediction - y))) < (0.02 if binary or multiclass else 0.06)
        result = [initial, final, prediction.flatten().tolist()]
        if adam:
            w1,b1,w2,b2=p
            query=jnp.array([[-1.1,-0.9],[0.9,-1.1],[0.1,0.9]],dtype=jnp.float64)
            prob=jax.nn.softmax(jnp.tanh(query@w1+b1)@w2+b2,axis=-1).block_until_ready()
            assert float(jnp.max(jnp.abs(prob-jnp.eye(3))))<0.03
            result.extend([prob.flatten().tolist(),[float(v) for a in p for v in a.flatten()]])
        return result
    return run, jax.__version__


def worker(args):
    factory = torch_workload if args.backend == "torch-eager" else jax_workload
    run, version = factory(args.workload, args.size)
    start = time.perf_counter()
    validation = run()  # Includes initialization, first execution and JIT compilation.
    cold = time.perf_counter() - start
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        result = run()
        samples.append(time.perf_counter() - start)
    assert result == validation
    print(json.dumps({"median_seconds": statistics.median(samples), "min_seconds": min(samples),
                      "cold_seconds": cold, "iterations": args.samples, "version": version,
                      "validation": validation, "peak_rss_bytes": peak_rss()}))


def peak_rss():
    status = pathlib.Path("/proc/self/status").read_text()
    return int(next(line.split()[1] for line in status.splitlines() if line.startswith("VmHWM:"))) * 1024


def isolated(command, env):
    # wait4 gives this child's high-water RSS, not cumulative RUSAGE_CHILDREN.
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
        start = time.perf_counter()
        p = subprocess.Popen(command, env=env, cwd=ROOT, stdout=stdout, stderr=stderr)
        _, status, usage = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        elapsed = time.perf_counter() - start
        stdout.seek(0); stderr.seek(0)
        output, errors = stdout.read(), stderr.read()
        if p.returncode:
            raise RuntimeError(f"{command} exited {p.returncode}: {errors}\n{output}")
        result = json.loads(output)
        result["launch_peak_rss_bytes"] = usage.ru_maxrss * 1024  # Linux reports KiB.
        result["process_seconds"] = elapsed
        return result


def controller(args):
    if sys.platform != "linux":
        raise SystemExit("RSS and CPU-affinity protocol currently requires Linux")
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})  # All workers inherit one logical CPU.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "cpu_affinity": [cpu],
              "harness_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              "binary_sha256": hashlib.sha256((ROOT / "target/release/ml").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(b"".join(path.name.encode() + path.read_bytes() for path in sorted((ROOT / "src").glob("*.rs")))).hexdigest(), "dtype": "f64", "device": "cpu", "samples": args.samples, "date": "2026-10-08", "rust": "1.85.1",
              "cpu_model": next((line.split(':', 1)[1].strip() for line in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown'),
              "protocol": "One warmup; median of complete workload executions. RSS is self-reported /proc/self/status VmHWM after warmup/JIT and measured runs, including imports; launch RSS is separately reported by wait4. No GPU results.", "results": []}
    binary = str(ROOT / "target/release/ml")
    with tempfile.TemporaryDirectory(prefix="ml-lang-benchmark-") as directory:
        for workload, size in [("matmul", 64), ("matmul", 128), ("matmul", 256), ("xor", 0), ("binary-xor", 0), ("multiclass", 0), ("adam-multiclass",0), ("minibatch",0)]:
            if workload == "matmul":
                source = pathlib.Path(directory) / f"matmul-{size}.ml"
                source.write_text(f"let a=ones({size},{size}); let b=ones({size},{size}); repeat 10 {{ let s=sum(a@b); print(s); assert_close(s,{size**3},1e-12); }}")
            else:
                source = ROOT / {"xor":"examples/xor.ml", "binary-xor":"examples/binary_xor.ml", "multiclass":"examples/multiclass.ml", "adam-multiclass":"examples/adam_multiclass.ml", "minibatch":"examples/minibatch.ml"}[workload]
            if workload in ("adam-multiclass", "minibatch"):
                model_source=source.read_text()
                source=pathlib.Path(directory)/f"{workload}.ml"
                source.write_text(model_source+"\nprint(w1);print(b1);print(w2);print(b2);\n")
            native_output = subprocess.check_output([binary, "run", str(source)], env=env, text=True)
            native_values = [ast.literal_eval(line[line.index('] ') + 2:]) for line in native_output.splitlines()]
            native = isolated([binary, "bench", str(source), str(args.samples)], env)
            native.update(backend="ml-lang", workload=workload, size=size, validation=native_values, workload_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            report["results"].append(native)
            for backend in ["torch-eager", "jax-jit"]:
                command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker", "--backend", backend,
                           "--workload", workload, "--size", str(size), "--samples", str(args.samples)]
                result = isolated(command, env)
                if workload != "matmul":
                    # Compare losses and every prediction; Adam workloads also compare weights.
                    import math
                    target = [native_values[0][0], native_values[1][0], *native_values[2]]
                    baseline = [result['validation'][0], result['validation'][1], *result['validation'][2]]
                    if workload in ("adam-multiclass", "minibatch"):
                        target.extend(native_values[3]); target.extend(v for t in native_values[4:] for v in t)
                        baseline.extend(result["validation"][3]); baseline.extend(result["validation"][4])
                    assert len(target)==len(baseline), (len(target),len(baseline))
                    assert all(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10) for a,b in zip(target,baseline)), (target,baseline)
                result.update(backend=backend, workload=workload, size=size)
                report["results"].append(result)
            print(f"validated {workload} {size}", file=sys.stderr)
    data = json.dumps(report, indent=2) + '\n'
    if args.output:
        pathlib.Path(args.output).write_text(data)
    print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=["torch-eager", "jax-jit"])
    parser.add_argument("--workload", choices=["matmul", "xor", "binary-xor", "multiclass", "adam-multiclass", "minibatch"])
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.worker:
        worker(args)
    else:
        controller(args)
