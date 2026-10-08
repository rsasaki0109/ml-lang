#!/usr/bin/env python3
"""Reference inference VM vs last-use SSA: Linux CPU f64, isolated processes."""
import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import subprocess
import tempfile
from compare import isolated
ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=5)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.samples<1 or args.samples>10000:parser.error('samples must be 1..10000')
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    binary=ROOT/'target/release/ml'
    report={'date':'2026-10-08','platform':platform.platform(),'cpu_affinity':[cpu],
            'dtype':'f64','device':'cpu','samples':args.samples,
            'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((ROOT/'src').glob('*.rs')))).hexdigest(),
            'harness_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            'isolation_harness_sha256':hashlib.sha256((ROOT/'benchmarks/compare.py').read_bytes()).hexdigest(),
            'protocol':'One warmup, median of complete executions, one CPU, separate process per mode. Compile/lower time separate. Linux self VmHWM includes IR and result snapshots. No GPU or buffer-reuse claims.',
            'results':[]}
    expected=1.0
    for _ in range(6):expected=math.tanh(expected)
    with tempfile.TemporaryDirectory(prefix='ml-lang-inference-bench-') as directory:
        workloads=[]
        for size in (256,512,1024):
            source=f'param x=ones({size},{size});let result=mean(tanh(tanh(tanh(tanh(tanh(tanh(x)))))));print(result);assert_close(result,{expected!r},1e-10);'
            workloads.append((f'activation-chain-{size}',source))
        # A forward-only, randomly initialized network, not an accuracy benchmark.
        source='''fn predict(x:Tensor[B,32],w1:Tensor[32,128],b1:Tensor[1,128],w2:Tensor[128,16])->Tensor[B,16]{let hidden=relu(x@w1+b1);return softmax(hidden@w2);}let x=rand(64,32,9);param w1=rand(32,128,42);param b1=zeros(1,128);param w2=rand(128,16,123);let prediction=predict(x,w1,b1,w2);print(prediction);assert_close(sum(prediction),64,1e-10);'''
        workloads.append(('dense-network',source))
        for workload,source in workloads:
            path=pathlib.Path(directory)/f'{workload}.ml';path.write_text(source)
            validation=None
            for mode,flags in [('inference-vm',[]),('inference-ssa',['--optimize'])]:
                command=[str(binary),'run',str(path),'--inference',*flags]
                output=subprocess.check_output(command,text=True)
                if validation is None:validation=output
                assert output==validation,workload
                result=isolated([str(binary),'bench',str(path),str(args.samples),'--inference',*flags],dict(os.environ))
                assert result['execution_mode']==mode
                result.update(workload=workload,workload_source_sha256=hashlib.sha256(source.encode()).hexdigest(),validation_sha256=hashlib.sha256(output.encode()).hexdigest())
                report['results'].append(result)
            print(f'validated {workload}',flush=True)
    pathlib.Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
