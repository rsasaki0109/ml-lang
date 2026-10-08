#!/usr/bin/env python3
"""Optional two-head CPU f64 correctness comparison, not a benchmark."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
if hasattr(os,"sched_getaffinity"):
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import numpy as np
import torch
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64",True)
torch.set_num_threads(1)
root=Path(__file__).resolve().parents[1]
f=(root/"examples/multihead_attention.ml").read_text().split("let x =")[0]
for batch,tokens in [(1,2),(2,3)]:
    x=np.arange(batch*tokens*2,dtype=np.float64).reshape(batch,tokens,2)/7-0.3
    weights=[np.arange(8,dtype=np.float64).reshape(2,4)/13-0.2 for _ in range(3)]
    weights.append(np.arange(8,dtype=np.float64).reshape(4,2)/11-0.3)
    source=f+f"let x={x.tolist()};"
    for name,w in zip(["wq","wk","wv","wo"],weights):source+=f"param {name}={w.tolist()};"
    source+="let y=attend(x,wq,wk,wv,wo);print(y);let loss=mse(y,x);print(loss);backward(loss);print(grad(wq));print(grad(wk));print(grad(wv));print(grad(wo));"
    tx=torch.tensor(x,dtype=torch.float64)
    tw=[torch.tensor(w,dtype=torch.float64,requires_grad=True) for w in weights]
    q,k,v=[(tx@w).reshape(batch,tokens,2,2).transpose(1,2) for w in tw[:3]]
    scores=q@k.transpose(-1,-2)*0.7071067811865476
    mask=torch.triu(torch.ones(tokens,tokens,dtype=torch.bool),diagonal=1)
    probability=torch.softmax(scores.masked_fill(mask,float('-inf')),dim=-1)
    ty=(probability@v).transpose(1,2).reshape(batch,tokens,4)@tw[3]
    tl=((ty-tx)**2).mean();tl.backward()
    def model(ws):
        q,k,v=[jnp.swapaxes((jnp.array(x)@w).reshape(batch,tokens,2,2),1,2) for w in ws[:3]]
        s=q@jnp.swapaxes(k,-1,-2)*0.7071067811865476
        p=jax.nn.softmax(jnp.where(jnp.triu(jnp.ones((tokens,tokens),dtype=bool),1),-jnp.inf,s),axis=-1)
        return jnp.swapaxes(p@v,1,2).reshape(batch,tokens,4)@ws[3]
    jw=tuple(jnp.array(w) for w in weights)
    jl,jg=jax.value_and_grad(lambda ws:jnp.mean((model(ws)-jnp.array(x))**2))(jw)
    references=[[ty.detach().numpy(),tl.detach().numpy(),*[w.grad.numpy() for w in tw]], [np.array(model(jw)),np.array(jl),*[np.array(g) for g in jg]]]
    with tempfile.TemporaryDirectory(prefix="ml-multihead-") as tmp:
        path=Path(tmp)/"check.ml";path.write_text(source)
        for backend in ["cpu","tiled"]:
            result=subprocess.run([str(root/"target/release/ml"),"run",str(path),"--backend",backend],check=True,capture_output=True,text=True)
            actual=[np.array(ast.literal_eval(s.split("] ",1)[1])) for s in result.stdout.splitlines()]
            assert len(actual)==6
            for reference in references:
                for a,b in zip(actual,reference):np.testing.assert_allclose(a,b.ravel(),rtol=1e-10,atol=1e-10)
print("2 shapes, 2 backends: multihead outputs, loss and all 32 parameter gradients agree with PyTorch/JAX (rtol=atol=1e-10)")
