#!/usr/bin/env python3
"""Toy unmasked single-head attention training, CPU f64, isolated processes."""
import argparse
import ast
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from compare import isolated, peak_rss, random_values
ROOT=pathlib.Path(__file__).resolve().parents[1]
X=[[[1.,0.],[0.,1.],[1.,0.]],[[0.,1.],[1.,0.],[0.,1.]]]
QUERY=[[[1.,0.],[0.,1.],[0.,1.]]]
def torch_run():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    x=torch.tensor(X,dtype=torch.float64);query=torch.tensor(QUERY,dtype=torch.float64)
    def run():
        wq=torch.tensor(random_values(8,42),dtype=torch.float64).reshape(2,4).requires_grad_()
        wk=torch.tensor(random_values(8,123),dtype=torch.float64).reshape(2,4).requires_grad_()
        wv=torch.tensor(random_values(4,99),dtype=torch.float64).reshape(2,2).requires_grad_()
        params=[wq,wk,wv];optimizer=torch.optim.Adam(params,lr=0.03,betas=(0.9,0.999),eps=1e-8,foreach=False,fused=False)
        def predict(data):
            q=data@wq;k=data@wk;v=data@wv
            return torch.softmax(torch.bmm(q,k.transpose(-2,-1))*0.5,dim=-1)@v
        initial=float(((predict(x)-x)**2).mean().detach())
        for _ in range(500):
            loss=((predict(x)-x)**2).mean();loss.backward()
            norm=torch.sqrt(sum(p.grad.square().sum() for p in params));factor=1.0/norm.clamp_min(1.0)
            with torch.no_grad():
                for p in params:p.grad.mul_(factor)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        prediction=predict(x).detach();result=predict(query).detach()
        assert float((prediction-x).abs().max())<0.03 and float((result-query).abs().max())<0.03
        return [initial,float(((prediction-x)**2).mean()),*prediction.flatten().tolist(),*result.flatten().tolist(),*[float(v) for p in params for v in p.detach().flatten()]]
    return run,torch.__version__
def jax_run():
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    x=jnp.array(X,dtype=jnp.float64);query=jnp.array(QUERY,dtype=jnp.float64)
    def predict(params,data):
        wq,wk,wv=params;q=data@wq;k=data@wk;v=data@wv
        return jax.nn.softmax((q@jnp.swapaxes(k,-2,-1))*0.5,axis=-1)@v
    def loss(params):return jnp.mean((predict(params,x)-x)**2)
    gradient=jax.grad(loss)
    @jax.jit
    def train(params):
        zeros=tuple(jnp.zeros_like(p) for p in params)
        def update(i,state):
            p,m,v=state;g=gradient(p);norm=jnp.sqrt(sum(jnp.sum(a*a) for a in g));factor=1.0/jnp.maximum(norm,1.0);g=tuple(a*factor for a in g)
            m=tuple(0.9*a+0.1*b for a,b in zip(m,g));v=tuple(0.999*a+0.001*b*b for a,b in zip(v,g));t=i+1
            p=tuple(a-0.03*(b/(1-0.9**t))/(jnp.sqrt(c/(1-0.999**t))+1e-8) for a,b,c in zip(p,m,v));return p,m,v
        return jax.lax.fori_loop(0,500,update,(params,zeros,zeros))[0]
    def run():
        params=(jnp.array(random_values(8,42),dtype=jnp.float64).reshape(2,4),jnp.array(random_values(8,123),dtype=jnp.float64).reshape(2,4),jnp.array(random_values(4,99),dtype=jnp.float64).reshape(2,2))
        initial=float(loss(params).block_until_ready());params=train(params);prediction=predict(params,x).block_until_ready();result=predict(params,query).block_until_ready()
        assert float(jnp.max(jnp.abs(prediction-x)))<0.03 and float(jnp.max(jnp.abs(result-query)))<0.03
        return [initial,float(loss(params)),*[float(v) for v in prediction.flatten()],*[float(v) for v in result.flatten()],*[float(v) for p in params for v in p.flatten()]]
    return run,jax.__version__
def worker(args):
    run,version=(torch_run if args.backend=='torch-eager' else jax_run)()
    start=time.perf_counter();validation=run();cold=time.perf_counter()-start;samples=[]
    for _ in range(args.samples):
        start=time.perf_counter();values=run();samples.append(time.perf_counter()-start);assert values==validation
    print(json.dumps({'version':version,'validation':validation,'median_seconds':statistics.median(samples),'min_seconds':min(samples),'cold_seconds':cold,'iterations':args.samples,'peak_rss_bytes':peak_rss()}))
def main(args):
    import math
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    binary=ROOT/'target/release/ml';source=(ROOT/'examples/attention.ml').read_text()+'\nprint(wq);print(wk);print(wv);\n'
    report={'date':'2026-10-08','platform':platform.platform(),'python':platform.python_version(),'rust':'1.85.1','dtype':'f64','device':'cpu','cpu_affinity':[cpu],'samples':args.samples,
            'source_sha256':hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((ROOT/'src').glob('*.rs')))).hexdigest(),
            'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'harness_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'isolation_harness_sha256':hashlib.sha256((ROOT/'benchmarks/compare.py').read_bytes()).hexdigest(),
            'workload_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'protocol':'Single-head unmasked toy attention, 500 Adam steps, clipping=1 without epsilon, shared projections, CPU f64, one affinity/thread, one warmup then median samples. Initial/final loss, all 18 predictions and 20 weights compared at rtol=1e-9, atol=1e-10. Process VmHWM includes imports/JIT/results. No accuracy or production-performance claim.','results':[]}
    with tempfile.TemporaryDirectory(prefix='ml-lang-attention-bench-') as d:
        path=pathlib.Path(d)/'attention.ml';path.write_text(source);reference=None
        for backend in ('cpu','tiled'):
            out=subprocess.check_output([str(binary),'run',str(path),'--backend',backend],env=env,text=True)
            arrays=[ast.literal_eval(line[line.index('] ')+2:]) for line in out.splitlines()];values=[v for array in arrays for v in array]
            assert len(values)==40
            if reference is None:reference=values
            assert values==reference
            result=isolated([str(binary),'bench',str(path),str(args.samples),'--backend',backend],env);result.update(backend='ml-'+backend,validation=values);report['results'].append(result)
        for backend in ('torch-eager','jax-jit'):
            result=isolated([sys.executable,str(pathlib.Path(__file__).resolve()),'--worker','--backend',backend,'--samples',str(args.samples)],env)
            assert len(result['validation'])==len(reference)
            assert all(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10) for a,b in zip(result['validation'],reference)),backend
            result.update(backend=backend);report['results'].append(result)
    pathlib.Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print('validated attention losses, 18 predictions and 20 weights across all four backends')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=int,default=5);parser.add_argument('--output');parser.add_argument('--worker',action='store_true');parser.add_argument('--backend',choices=['torch-eager','jax-jit']);args=parser.parse_args()
    if args.samples<1 or args.samples>10000:parser.error('samples must be 1..10000')
    if args.worker:worker(args)
    else:
        if not args.output:parser.error('--output is required')
        main(args)
