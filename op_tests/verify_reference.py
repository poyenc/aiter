# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Verification script to compare three attention implementations:
1. attention_fp8_ref_online() - new online softmax reference
2. attention_fp8_ref() - old batch softmax reference
3. BF16 kernel (aiter.flash_attn_func) - proven production kernel

This script verifies that the FP8 reference implementations are correct
before using them to debug the FP8 kernel.
"""

import math
import torch
import aiter
from aiter import dtypes
from aiter import per_tensor_quant
from aiter.test_common import run_perftest
from einops import repeat

from test_mha_fp8 import attention_fp8_ref_online, attention_fp8_ref


def run_bf16_kernel(q_bf16, k_bf16, v_bf16, causal=False, window_size=(-1, -1)):
    """Run the proven BF16 flash attention kernel."""
    out, _ = run_perftest(
        aiter.flash_attn_func,
        q_bf16,
        k_bf16,
        v_bf16,
        dropout_p=0.0,
        causal=causal,
        window_size=window_size,
        bias=None,
        alibi_slopes=None,
        deterministic=True,
        return_lse=False,
        return_attn_probs=False,
        num_iters=2,
        num_warmup=0,
    )
    return out


def compare_implementations(
    batch_size,
    seqlen_q,
    seqlen_k,
    nheads,
    nheads_k,
    d,
    d_v,
    causal,
    kv_tile_size=64,
):
    """Compare all three implementations and report differences."""
    torch.manual_seed(42)

    # Generate BF16 inputs
    q_bf16 = torch.rand(batch_size, seqlen_q, nheads, d, device="cuda", dtype=torch.bfloat16)
    k_bf16 = torch.rand(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=torch.bfloat16)
    v_bf16 = torch.rand(batch_size, seqlen_k, nheads_k, d_v, device="cuda", dtype=torch.bfloat16)

    # Quantize to FP8
    q_fp8, q_descale = per_tensor_quant(q_bf16, quant_dtype=dtypes.fp8)
    k_fp8, k_descale = per_tensor_quant(k_bf16, quant_dtype=dtypes.fp8)
    v_fp8, v_descale = per_tensor_quant(v_bf16, quant_dtype=dtypes.fp8)

    # Run all three implementations
    # 1. BF16 kernel (ground truth)
    out_bf16 = run_bf16_kernel(q_bf16, k_bf16, v_bf16, causal=causal)

    # 2. FP8 reference (batch softmax)
    out_fp8_batch = attention_fp8_ref(
        q_fp8, k_fp8, v_fp8,
        q_descale.item(), k_descale.item(), v_descale.item(),
        causal=causal,
    )

    # 3. FP8 reference (online softmax)
    out_fp8_online = attention_fp8_ref_online(
        q_fp8, k_fp8, v_fp8,
        q_descale.item(), k_descale.item(), v_descale.item(),
        causal=causal,
        kv_tile_size=kv_tile_size,
    )

    # Compute differences
    diff_online_vs_batch = (out_fp8_online - out_fp8_batch).abs()
    diff_online_vs_bf16 = (out_fp8_online.float() - out_bf16.float()).abs()
    diff_batch_vs_bf16 = (out_fp8_batch.float() - out_bf16.float()).abs()

    max_online_vs_batch = diff_online_vs_batch.max().item()
    max_online_vs_bf16 = diff_online_vs_bf16.max().item()
    max_batch_vs_bf16 = diff_batch_vs_bf16.max().item()

    return {
        "online_vs_batch": max_online_vs_batch,
        "online_vs_bf16": max_online_vs_bf16,
        "batch_vs_bf16": max_batch_vs_bf16,
    }


def main():
    print("=" * 70)
    print("Verification: Comparing attention implementations")
    print("=" * 70)
    print()
    print("Implementations:")
    print("  1. attention_fp8_ref_online() - Online softmax with KV tiling")
    print("  2. attention_fp8_ref()        - Batch softmax (full sequence)")
    print("  3. aiter.flash_attn_func      - BF16 production kernel")
    print()

    # Test configurations
    configs = [
        # (seqlen_q, seqlen_k, causal, description)
        (64, 32, False, "Single KV tile (32 < 64), no mask"),
        (64, 64, False, "Single KV tile (64 == 64), full"),
        (256, 64, True, "Single KV tile with causal"),
        (256, 128, False, "Two KV tiles (128 = 2*64)"),
        (256, 256, True, "Multiple tiles with causal"),
        (128, 217, False, "Non-aligned seqlen_k"),
        (113, 203, True, "Non-aligned both, causal"),
    ]

    # Fixed parameters
    batch_size = 2
    nheads = 8
    nheads_k = 1  # GQA
    d = 128
    d_v = 128

    print(f"Fixed params: batch={batch_size}, nheads={nheads}, nheads_k={nheads_k}, d={d}, d_v={d_v}")
    print()

    # Run comparisons
    print("-" * 70)
    print(f"{'Config':<35} {'Online vs Batch':>12} {'Online vs BF16':>14} {'Batch vs BF16':>13}")
    print("-" * 70)

    all_passed = True

    for seqlen_q, seqlen_k, causal, desc in configs:
        results = compare_implementations(
            batch_size=batch_size,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            nheads=nheads,
            nheads_k=nheads_k,
            d=d,
            d_v=d_v,
            causal=causal,
        )

        # Format config description
        causal_str = "causal" if causal else "non-causal"
        config_str = f"q={seqlen_q}, k={seqlen_k}, {causal_str}"

        # Check if online is within expected tolerance
        # Online softmax with FP8 quantization per-tile differs from batch softmax
        # because P values are quantized at different running max values.
        # Tolerance of 0.02 allows for FP8 quantization differences across tiles.
        online_batch_ok = results["online_vs_batch"] < 0.02
        status = "OK" if online_batch_ok else "FAIL"

        if not online_batch_ok:
            all_passed = False

        print(f"{config_str:<35} {results['online_vs_batch']:>12.6f} {results['online_vs_bf16']:>14.4f} {results['batch_vs_bf16']:>13.4f}  [{status}]")

    print("-" * 70)
    print()

    # Summary
    print("Interpretation:")
    print("  - 'Online vs Batch' can differ due to FP8 quantization at different row max values")
    print("    (Online softmax quantizes P per-tile with running max, batch uses global max)")
    print("  - Small differences (<0.02) are expected and acceptable")
    print("  - 'Online vs BF16' and 'Batch vs BF16' differ due to FP8 quantization loss")
    print()

    if all_passed:
        print("RESULT: All tests PASSED - attention_fp8_ref_online() is within tolerance")
        print("        The online softmax reference is suitable for kernel debugging")
    else:
        print("RESULT: Some tests FAILED - differences exceed expected tolerance (0.02)")
        print("        Need to investigate the online softmax implementation")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
