#!/usr/bin/env python3
"""Optional CPU f64 forward/VJP comparison; no timing or memory claims."""
import argparse
import ast
import os
from pathlib import Path
import subprocess

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import numpy as np
import torch
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/ml"))
    args = parser.parse_args()
    binary = args.binary.resolve()
    cases = 0
    for shape in [(1,), (3,), (2, 3), (2, 2, 3)]:
        for constant in [False, True]:
            x = np.full(shape, 0.7) if constant else np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 7 - 0.3
            g = np.arange(x.size, dtype=np.float64).reshape(shape) / 11 - 0.2
            source = f"param x={x.tolist()};let y=layer_norm(x);print(y);backward(sum(y*{g.tolist()}));print(grad(x));"
            # Run through the public CLI using an owned temporary source.
            import tempfile
            with tempfile.TemporaryDirectory(prefix="ml-layer-norm-") as tmp:
                path = Path(tmp) / "check.ml"
                path.write_text(source)
                result = subprocess.run([str(binary), "run", str(path)], check=True, capture_output=True, text=True)
            lines = result.stdout.splitlines()
            assert len(lines) == 2, result.stdout
            actual = [np.array(ast.literal_eval(line.split("] ", 1)[1])).reshape(shape) for line in lines]
            tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            ty = torch.nn.functional.layer_norm(tx, (shape[-1],), eps=1e-5)
            (ty * torch.tensor(g)).sum().backward()
            def norm(a):
                centered = a - jnp.mean(a, axis=-1, keepdims=True)
                return centered / jnp.sqrt(jnp.mean(centered * centered, axis=-1, keepdims=True) + 1e-5)
            jy = norm(jnp.array(x))
            jg = jax.grad(lambda a: jnp.sum(norm(a) * jnp.array(g)))(jnp.array(x))
            for expected in [(ty.detach().numpy(), tx.grad.numpy()), (np.array(jy), np.array(jg))]:
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
            cases += 1
    print(f"{cases} CPU f64 cases: forward and every input VJP agree with PyTorch/JAX (rtol=atol=1e-10)")


if __name__ == "__main__":
    main()
