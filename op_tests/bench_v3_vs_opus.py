#!/usr/bin/env python3
"""Benchmark CK FMHA V3 bf16 fwd vs opus_attn with matching parameters."""

import torch
import time
import argparse
import subprocess
import os

def flops(B, N, H, D, causal):
    f = 4 * B * N * N * H * D
    return f // 2 if causal else f

def bench_ck_v3(B, N, H, H_KV, D, causal, warmup=10, iters=50):
    """Benchmark CK FMHA V3 via aiter fmha_v3_fwd_ck_func."""
    from aiter.ops.mha import fmha_v3_fwd_ck_func

    torch.manual_seed(0)
    q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, N, H_KV, D, dtype=torch.bfloat16, device="cuda")

    # warmup
    for _ in range(warmup):
        _ = fmha_v3_fwd_ck_func(q, k, v, causal=causal)
    torch.cuda.synchronize()

    # timed
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        _ = fmha_v3_fwd_ck_func(q, k, v, causal=causal)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = sum(times_ms) / len(times_ms)
    f = flops(B, N, H, D, causal)
    tflops = f / (avg_ms * 1e-3) / 1e12
    return avg_ms, tflops

def bench_opus(B, N, H, H_KV, D, causal, exe_path):
    """Benchmark opus_attn via subprocess."""
    cmd = [exe_path, f"-b={B}", f"-n={N}", f"-h={H}", f"--hkv={H_KV}", f"-d={D}"]
    if not causal:
        cmd.append("--no-causal")
    result = subprocess.run(cmd, capture_output=True, text=True)
    # parse output: "GQA ... Performance: avg_time=X.XXX ms, Y.YY TFlops"
    for line in result.stdout.split("\n"):
        if "Performance" in line:
            parts = line.split("avg_time=")[1]
            ms_str, rest = parts.split(" ms, ")
            tflops_str = rest.split(" TFlops")[0]
            return float(ms_str), float(tflops_str)
    print("opus_attn output:", result.stdout)
    print("opus_attn stderr:", result.stderr)
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", type=int, default=16)
    parser.add_argument("-H", type=int, default=64)
    parser.add_argument("--hkv", type=int, default=8)
    parser.add_argument("-d", type=int, default=128)
    parser.add_argument("--seqlens", type=str, default="1024,2048,4096,8192,16384")
    parser.add_argument("--opus-exe", type=str,
                        default="/root/workspace/repo/gcnasm/opus_attn/build/gqa_attn.exe")
    parser.add_argument("--skip-opus", action="store_true")
    parser.add_argument("--skip-ck", action="store_true")
    args = parser.parse_args()

    seqlens = [int(s) for s in args.seqlens.split(",")]
    B, H, H_KV, D = args.b, args.H, args.hkv, args.d

    print(f"Config: B={B}, H={H}, H_KV={H_KV}, D={D}")
    print(f"{'':>6} | {'CK V3 Causal':>20} | {'opus Causal':>20} | {'CK V3 Non-causal':>20} | {'opus Non-causal':>20}")
    print(f"{'N':>6} | {'ms':>8} {'TFlops':>10} | {'ms':>8} {'TFlops':>10} | {'ms':>8} {'TFlops':>10} | {'ms':>8} {'TFlops':>10}")
    print("-" * 102)

    for N in seqlens:
        row = f"{N:>6} |"
        for causal in [True, False]:
            if not args.skip_ck:
                ck_ms, ck_tf = bench_ck_v3(B, N, H, H_KV, D, causal)
                row += f" {ck_ms:>8.3f} {ck_tf:>10.1f} |"
            else:
                row += f" {'--':>8} {'--':>10} |"

            if not args.skip_opus:
                op_ms, op_tf = bench_opus(B, N, H, H_KV, D, causal, args.opus_exe)
                row += f" {op_ms:>8.3f} {op_tf:>10.1f} |"
            else:
                row += f" {'--':>8} {'--':>10} |"
        print(row)

if __name__ == "__main__":
    main()
