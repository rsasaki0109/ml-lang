#!/usr/bin/env python3
"""CPU permutation ablation in fresh single-affinity processes; no framework claims."""
import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
import statistics
import subprocess
import tempfile
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    root = Path(__file__).resolve().parents[1]
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    with tempfile.TemporaryDirectory(prefix="ml-axis-swap-") as tmp:
        binary = Path(tmp) / "axis-swap"
        start = time.perf_counter()
        subprocess.run(["rustc", "--edition=2021", "-C", "opt-level=3", str(root / "benchmarks/axis_swap.rs"), "-o", str(binary)], check=True)
        compile_seconds = time.perf_counter() - start
        rows = []
        for workload in ["heads", "last", "identity"]:
            for mode in ["forward", "vjp"]:
                for backend in ["legacy", "blocks"]:
                    samples = []
                    for _ in range(args.samples):
                        result = subprocess.run([str(binary), backend, mode, workload], check=True, text=True, capture_output=True)
                        samples.append(json.loads(result.stdout))
                    shape, axes = {"heads": ([8,64,8,32], [1,2]), "last": ([8,8,64,32], [2,3]), "identity": ([8,8,64,32], [1,1])}[workload]
                    rows.append(dict(workload=workload,shape=shape,axes=axes,mode=mode,backend=backend,samples=samples,median_ms=statistics.median(s["milliseconds"] for s in samples),median_peak_rss_mib=statistics.median(s["peak_rss_kib"] / 1024 for s in samples)))
        cpu_model = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
        report = dict(schema=1,date_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu_model=cpu_model,cpu_affinity=cpu,rustc=subprocess.check_output(["rustc","--version"],text=True).strip(),compile_seconds=compile_seconds,binary_sha256=digest(binary),tensor_sha256=digest(root/"src/tensor.rs"),harness_sha256=digest(root/"benchmarks/axis_swap.rs"),controller_sha256=digest(Path(__file__)),method="One warmup; 20 iterations per sample; fresh processes; median; correctness over every output/gradient element after timing; VmHWM includes input, zero output-shape fixture, upstream and result plus validation and allocator state; no GPU memory.",results=rows)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        for row in rows:
            print(f"{row['workload']:8} {row['mode']:7} {row['backend']:6} {row['median_ms']:.3f} ms {row['median_peak_rss_mib']:.2f} MiB")


if __name__ == "__main__":
    main()
