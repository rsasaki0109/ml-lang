#!/usr/bin/env python3
"""Optional CPU f64 Transformer numerical comparison; no timing or RSS claims."""
import ast
import os
from pathlib import Path
import re
import subprocess
import tempfile
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
if hasattr(os,"sched_getaffinity"):
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import numpy as np
import torch
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64",True)
torch.set_num_threads(1)
root=Path(__file__).resolve().parents[1]
names=['wq','wk','wv','wo','gain1','bias1','gain2','bias2','w1','b1','w2','b2','readout','readbias']
prefix=(root/'examples/transformer.ml').read_text().split('let initial =')[0]
full_x=np.array([[[1,0,1,0],[0,1,0,1],[1,0,1,1]],[[0,1,1,0],[1,0,0,1],[0,1,1,1]]],dtype=np.float64)
full_target=np.array([[[1,0],[0,1],[1,0]],[[0,1],[1,0],[0,1]]],dtype=np.float64)
def parse(line):
    shape,values=line.removeprefix('Tensor').split('] ',1)
    return np.array(ast.literal_eval(values)).reshape(ast.literal_eval(shape+']'))
def pytorch_model(x,w):
    batch,tokens,_=x.shape
    n=torch.nn.functional.layer_norm(x,(4,),eps=1e-5)*w[4]+w[5]
    q,k,v=[(n@p).reshape(batch,tokens,2,2).transpose(1,2) for p in w[:3]]
    s=q@k.transpose(-1,-2)*0.7071067811865476
    mask=torch.triu(torch.ones(tokens,tokens,dtype=torch.bool),1)
    context=torch.softmax(s.masked_fill(mask,float('-inf')),dim=-1)@v
    residual=x+context.transpose(1,2).reshape(batch,tokens,4)@w[3]
    n=torch.nn.functional.layer_norm(residual,(4,),eps=1e-5)*w[6]+w[7]
    hidden=torch.nn.functional.gelu(n@w[8]+w[9],approximate='tanh')
    return (residual+hidden@w[10]+w[11])@w[12]+w[13]
def jax_model(x,w):
    batch,tokens,_=x.shape
    def norm(x):
        z=x-jnp.mean(x,axis=-1,keepdims=True)
        return z/jnp.sqrt(jnp.mean(z*z,axis=-1,keepdims=True)+1e-5)
    n=norm(x)*w[4]+w[5]
    q,k,v=[jnp.swapaxes((n@p).reshape(batch,tokens,2,2),1,2) for p in w[:3]]
    s=q@jnp.swapaxes(k,-1,-2)*0.7071067811865476
    mask=jnp.triu(jnp.ones((tokens,tokens),dtype=bool),1)
    context=jax.nn.softmax(jnp.where(mask,-jnp.inf,s),axis=-1)@v
    residual=x+jnp.swapaxes(context,1,2).reshape(batch,tokens,4)@w[3]
    n=norm(residual)*w[6]+w[7]
    hidden=jax.nn.gelu(n@w[8]+w[9],approximate=True)
    return (residual+hidden@w[10]+w[11])@w[12]+w[13]
for batch,tokens in [(1,2),(2,3)]:
    x=full_x[:batch,:tokens];target=full_target[:batch,:tokens]
    source=re.sub(r'^let x = .*;$',f'let x = {x.tolist()};',prefix,flags=re.M)
    source=re.sub(r'^let target = .*;$',f'let target = {target.tolist()};',source,flags=re.M)
    source+=''.join(f'print({name});' for name in names)
    source+=f'let logits=model(x,{",".join(names)});let loss=cross_entropy(reshape(logits,{batch*tokens},2),reshape(target,{batch*tokens},2));print(logits);print(loss);backward(loss);'
    source+=''.join(f'print(grad({name}));' for name in names)
    with tempfile.TemporaryDirectory(prefix='ml-transformer-') as tmp:
        path=Path(tmp)/'check.ml';path.write_text(source)
        for backend in ['cpu','tiled']:
            result=subprocess.run([str(root/'target/release/ml'),'run',str(path),'--backend',backend],check=True,capture_output=True,text=True)
            outputs=[parse(line) for line in result.stdout.splitlines()];assert len(outputs)==30
            weights=outputs[:14];actual=outputs[14:]
            tw=[torch.tensor(w,dtype=torch.float64,requires_grad=True) for w in weights]
            ty=pytorch_model(torch.tensor(x),tw)
            tl=torch.nn.functional.cross_entropy(ty.reshape(-1,2),torch.tensor(target).reshape(-1,2));tl.backward()
            jw=tuple(jnp.array(w) for w in weights)
            def loss(ws):return -jnp.mean(jnp.sum(jnp.array(target)*jax.nn.log_softmax(jax_model(jnp.array(x),ws),axis=-1),axis=-1))
            jl,jg=jax.value_and_grad(loss)(jw)
            references=[[ty.detach().numpy(),tl.detach().numpy(),*[w.grad.numpy() for w in tw]],[np.array(jax_model(jnp.array(x),jw)),np.array(jl),*[np.array(g) for g in jg]]]
            for reference in references:
                for a,b in zip(actual,reference):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
print('2 shapes, 2 CPU backends: Transformer logits, cross entropy and all 166 parameter gradients agree with PyTorch/JAX (rtol=atol=1e-10)')
