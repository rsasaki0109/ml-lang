#!/usr/bin/env python3
"""CPU f64 GEMM/VJP comparison: reference/tiled ml-lang, PyTorch eager, JAX JIT."""
import argparse
import ast
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
from compare import isolated
ROOT = pathlib.Path(__file__).resolve().parents[1]

def framework_worker(args):
    if args.backend == 'torch-eager':
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        def run():
            tracked=args.workload=='backward'
            a=torch.ones((args.size,args.size),dtype=torch.float64,requires_grad=tracked)
            b=torch.ones_like(a,requires_grad=tracked)
            if not tracked:
                results=[]
                for _ in range(10):
                    z=a@b;assert bool(torch.all(z==args.size));results.append(float(z.sum()))
                return results
            z=a@b;loss=(z*z).mean();loss.backward()
            assert bool(torch.all(a.grad==2)) and bool(torch.all(b.grad==2))
            return [float(loss.detach()),float(a.grad.sum()),float(b.grad.sum())]
        version=torch.__version__
    else:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64',True)
        def loss(a,b):
            z=a@b;return jnp.mean(z*z)
        kernel=jax.jit(jax.value_and_grad(loss,argnums=(0,1))) if args.workload=='backward' else jax.jit(lambda a,b:a@b)
        def run():
            a=jnp.ones((args.size,args.size),dtype=jnp.float64);b=jnp.ones_like(a)
            if args.workload=='matmul':
                results=[]
                for _ in range(10):
                    z=kernel(a,b).block_until_ready();assert bool(jnp.all(z==args.size));results.append(float(z.sum()))
                return results
            value,(ga,gb)=kernel(a,b)
            value.block_until_ready()
            assert bool(jnp.all(ga==2)) and bool(jnp.all(gb==2))
            return [float(value),float(ga.sum()),float(gb.sum())]
        version=jax.__version__
    import statistics,time
    start=time.perf_counter();validation=run();cold=time.perf_counter()-start;samples=[]
    for _ in range(args.samples):
        start=time.perf_counter();value=run();samples.append(time.perf_counter()-start);assert value==validation
    from compare import peak_rss
    print(json.dumps({'version':version,'validation':validation,'median_seconds':statistics.median(samples),'min_seconds':min(samples),'iterations':args.samples,'cold_seconds':cold,'peak_rss_bytes':peak_rss()}))

def controller(args):
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    binary=ROOT/'target/release/ml'
    report={'date':'2026-10-08','platform':platform.platform(),'cpu_affinity':[cpu],'dtype':'f64','device':'cpu','samples':args.samples,
            'python':platform.python_version(),'rust':'1.85.1','tiles':{'rows':16,'reduction':32,'columns':128,'transpose':32},'small_work_fallback':32768,
            'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((ROOT/'src').glob('*.rs')))).hexdigest(),
            'harness_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            'isolation_harness_sha256':hashlib.sha256((ROOT/'benchmarks/compare.py').read_bytes()).hexdigest(),
            'protocol':'CPU f64, one inherited CPU affinity and framework thread, one warmup and median complete executions. Compile/lower excluded from warm timings. Process VmHWM includes warmup/imports/JIT/result snapshots. VJP transpose scratch is not in retained-payload counter. Ones workloads have exact analytical results, validated elementwise.',
            'results':[]}
    with tempfile.TemporaryDirectory(prefix='ml-lang-kernel-bench-') as directory:
        for size in (128,256,512):
            for workload in ('matmul','backward'):
                if workload=='matmul':
                    source=f'let a=ones({size},{size});let b=ones({size},{size});repeat 10{{let c=a@b;assert_close(c,ones({size},{size})*{size},0);print(sum(c));}}'
                    expected=[float(size**3)]*10
                else:
                    source=f'param a=ones({size},{size});param b=ones({size},{size});let z=a@b;let loss=mean(z*z);backward(loss);assert_close(grad(a),ones({size},{size})*2,0);assert_close(grad(b),ones({size},{size})*2,0);print(loss);print(sum(grad(a)));print(sum(grad(b)));'
                    expected=[float(size**2),float(2*size**2),float(2*size**2)]
                path=pathlib.Path(directory)/f'{workload}-{size}.ml';path.write_text(source)
                for backend in ('cpu','tiled'):
                    flags=[] if workload=='backward' else ['--inference']
                    command=[str(binary),'run',str(path),'--backend',backend,*flags]
                    output=subprocess.check_output(command,env=env,text=True)
                    values=[ast.literal_eval(line[line.index('] ')+2:])[0] for line in output.splitlines()]
                    assert values==expected,(workload,size,backend)
                    result=isolated([str(binary),'bench',str(path),str(args.samples),'--backend',backend,*flags],env)
                    result.update(backend=f'ml-{backend}',workload=workload,size=size,validation=values,workload_source_sha256=hashlib.sha256(source.encode()).hexdigest())
                    report['results'].append(result)
                for backend in ('torch-eager','jax-jit'):
                    result=isolated([sys.executable,str(pathlib.Path(__file__).resolve()),'--worker','--backend',backend,'--workload',workload,'--size',str(size),'--samples',str(args.samples)],env)
                    assert result['validation']==expected,(backend,workload,size,result['validation'])
                    result.update(backend=backend,workload=workload,size=size);report['results'].append(result)
                print(f'validated {workload} {size}',flush=True)
    pathlib.Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=int,default=5);parser.add_argument('--output');parser.add_argument('--worker',action='store_true');parser.add_argument('--backend',choices=['torch-eager','jax-jit']);parser.add_argument('--workload',choices=['matmul','backward']);parser.add_argument('--size',type=int,default=128)
    args=parser.parse_args()
    if args.samples<1 or args.samples>10000:parser.error('samples must be 1..10000')
    if args.worker:
        framework_worker(args)
    else:
        if not args.output:parser.error('--output is required')
        controller(args)
