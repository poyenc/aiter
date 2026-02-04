# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Debug script for FP8 FMHA v3 pipeline.
# This script tests whether K position 4 contributes to PV GEMM output.

import torch
import aiter
from aiter import dtypes, per_tensor_quant
from aiter.test_common import run_perftest


def test_k_position_contribution(seqlen_q=5, seqlen_k=5):
    """
    Test which K positions contribute to PV GEMM output.

    If setting V[K=N] to 0 produces the same output as original V,
    then K position N is NOT contributing to the computation.
    """
    torch.random.manual_seed(0)

    # Test parameters
    d, d_v = 128, 128
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

    # Return list of non-contributing K positions
    non_contributing = [k_pos for k_pos in range(seqlen_k)
                        if k_pos >= 4]  # placeholder, will be updated
    return non_contributing


def test_k_key_contribution(seqlen_k=5):
    """
    Test if K position contributes to attention by modifying K (not V).

    If setting K[pos]=0 doesn't change output, then P[pos] is not contributing.
    This tests whether the attention weights (P) are correctly computed and used.
    """
    torch.random.manual_seed(0)

    d, d_v = 128, 128
    seqlen_q = seqlen_k
    nheads = 1

    q = torch.rand(1, seqlen_q, nheads, d, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(1, seqlen_k, nheads, d, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(1, seqlen_k, nheads, d_v, device="cuda", dtype=torch.bfloat16)

    q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
    k_quant, k_descale = per_tensor_quant(k, quant_dtype=dtypes.fp8)
    v_quant, v_descale = per_tensor_quant(v, quant_dtype=dtypes.fp8)

    print("=" * 70)
    print(f"Testing K (key) position contribution to attention")
    print(f"seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print("=" * 70)

    # Run with original K
    print("\n[1] Running kernel with original K...")
    out_orig, _ = run_perftest(
        aiter.flash_attn_fp8_pertensor_func,
        q_quant, k_quant, v_quant,
        q_descale, k_descale, v_descale,
        causal=False, window_size=(-1, -1),
        num_iters=2, num_warmup=0,
    )

    print("\n[2] Testing each K position (modifying K, not V):")
    for k_pos in range(seqlen_k):
        k_mod_quant = k_quant.clone()
        k_mod_quant[:, k_pos, :, :] = 0  # Zero out K at this position

        out_mod, _ = run_perftest(
            aiter.flash_attn_fp8_pertensor_func,
            q_quant, k_mod_quant, v_quant,
            q_descale, k_descale, v_descale,
            causal=False, window_size=(-1, -1),
            num_iters=2, num_warmup=0,
        )

        diff = (out_orig - out_mod).abs().max().item()

        if diff < 0.001:
            status = "NOT contributing (BUG in P!)"
        else:
            status = "contributing (OK)"

        print(f"  K[pos={k_pos}]=0 -> diff={diff:.6f} -> {status}")


def test_multiple_seqlen_k():
    """Test K position contribution for various seqlen_k values."""
    print("=" * 70)
    print("Testing K position contribution for various seqlen_k values")
    print("=" * 70)

    results = {}
    for seqlen_k in [4, 5, 8, 9, 10, 11, 12, 16, 20, 24, 28, 32]:
        torch.random.manual_seed(0)

        d, d_v = 128, 128
        seqlen_q = seqlen_k  # Use same seqlen_q for simplicity
        nheads = 1

        q = torch.rand(1, seqlen_q, nheads, d, device="cuda", dtype=torch.bfloat16)
        k = torch.rand(1, seqlen_k, nheads, d, device="cuda", dtype=torch.bfloat16)
        v = torch.rand(1, seqlen_k, nheads, d_v, device="cuda", dtype=torch.bfloat16)

        q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
        k_quant, k_descale = per_tensor_quant(k, quant_dtype=dtypes.fp8)
        v_quant, v_descale = per_tensor_quant(v, quant_dtype=dtypes.fp8)

        # Run with original V
        out_orig, _ = run_perftest(
            aiter.flash_attn_fp8_pertensor_func,
            q_quant, k_quant, v_quant,
            q_descale, k_descale, v_descale,
            causal=False, window_size=(-1, -1),
            num_iters=2, num_warmup=0,
        )

        # Test each K position
        non_contributing = []
        for k_pos in range(seqlen_k):
            v_mod_quant = v_quant.clone()
            v_mod_quant[:, k_pos, :, :] = 0

            out_mod, _ = run_perftest(
                aiter.flash_attn_fp8_pertensor_func,
                q_quant, k_quant, v_mod_quant,
                q_descale, k_descale, v_descale,
                causal=False, window_size=(-1, -1),
                num_iters=2, num_warmup=0,
            )

            diff = (out_orig - out_mod).abs().max().item()
            if diff < 0.001:
                non_contributing.append(k_pos)

        results[seqlen_k] = non_contributing
        if non_contributing:
            print(f"  seqlen_k={seqlen_k:2d}: K positions NOT contributing: {non_contributing}")
        else:
            print(f"  seqlen_k={seqlen_k:2d}: All K positions contributing (OK)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for seqlen_k, non_contrib in results.items():
        status = "BUG" if non_contrib else "OK"
        print(f"  seqlen_k={seqlen_k:2d}: {status} - missing: {non_contrib if non_contrib else 'none'}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        test_multiple_seqlen_k()
    elif len(sys.argv) > 1 and sys.argv[1] == "--key":
        test_k_key_contribution()
    else:
        test_k_position_contribution()
