# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Debug script for FP8 FMHA v3 pipeline.
# This script tests whether K position 4 contributes to PV GEMM output.

import torch
import aiter
from aiter import dtypes, per_tensor_quant
from aiter.test_common import run_perftest


def test_k_position_contribution():
    """
    Test which K positions contribute to PV GEMM output.

    If setting V[K=N] to 0 produces the same output as original V,
    then K position N is NOT contributing to the computation.
    """
    torch.random.manual_seed(0)

    # Test parameters
    seqlen_q, seqlen_k, d, d_v = 5, 5, 128, 128
    nheads = 1

    # Generate random Q, K, V
    q = torch.rand(1, seqlen_q, nheads, d, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(1, seqlen_k, nheads, d, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(1, seqlen_k, nheads, d_v, device="cuda", dtype=torch.bfloat16)

    # Quantize original inputs
    q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
    k_quant, k_descale = per_tensor_quant(k, quant_dtype=dtypes.fp8)
    v_quant, v_descale = per_tensor_quant(v, quant_dtype=dtypes.fp8)

    print("=" * 70)
    print("Testing K position contribution to PV GEMM")
    print(f"seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, d={d}, d_v={d_v}")
    print("=" * 70)

    # Run kernel with original V
    print("\n[1] Running kernel with original V...")
    out_orig, _ = run_perftest(
        aiter.flash_attn_fp8_pertensor_func,
        q_quant, k_quant, v_quant,
        q_descale, k_descale, v_descale,
        causal=False,
        window_size=(-1, -1),
        num_iters=2,
        num_warmup=0,
    )

    # Test each K position
    # IMPORTANT: Modify v_quant directly (not v), and use SAME v_descale
    # This ensures quantization doesn't affect the comparison
    print("\n[2] Testing each K position contribution:")
    print("    (Modifying v_quant directly, using same v_descale)")
    for k_pos in range(seqlen_k):
        # Create modified v_quant with K position k_pos set to 0
        v_mod_quant = v_quant.clone()
        v_mod_quant[:, k_pos, :, :] = 0  # Zero out in FP8 representation

        # Run kernel with SAME v_descale
        out_mod, _ = run_perftest(
            aiter.flash_attn_fp8_pertensor_func,
            q_quant, k_quant, v_mod_quant,
            q_descale, k_descale, v_descale,  # Use original v_descale
            causal=False,
            window_size=(-1, -1),
            num_iters=2,
            num_warmup=0,
        )

        # Compare
        diff = (out_orig - out_mod).abs().max().item()

        if diff < 0.001:
            status = "NOT contributing (BUG!)"
        else:
            status = "contributing (OK)"

        print(f"  V[K={k_pos}]=0 -> diff={diff:.6f} -> K position {k_pos} is {status}")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("If any K position shows 'NOT contributing', the PV GEMM is buggy.")
    print("=" * 70)


if __name__ == "__main__":
    test_k_position_contribution()
