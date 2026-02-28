# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
import torch
import aiter
from aiter import dtypes
from aiter.test_common import run_perftest, perftest
from aiter import per_tensor_quant
from einops import repeat
import pytest
import pandas as pd
import argparse

benchmark = {}


def attention_fp8_ref_online(
    q_fp8,
    k_fp8,
    v_fp8,
    q_descale: float,
    k_descale: float,
    v_descale: float,
    causal=False,
    window_size=(-1, -1),
    logits_soft_cap=0.0,
    kv_tile_size=128,
):
    """
    Reference implementation simulating online softmax with KV tiling.

    This matches the kernel's tile-by-tile computation for debugging.
    KV tile size = 64 (kN0 in the v3 pipeline).
    """
    if causal:
        window_size = (window_size[0], 0)

    batch_size, seqlen_q, nheads, d = q_fp8.shape
    seqlen_k = k_fp8.shape[1]
    d_v = v_fp8.shape[-1]

    # FP8 E4M3 max value
    fp8_max = torch.finfo(dtypes.fp8).max  # 448.0

    # log2(e) for FAST_EXP2 path
    log2e = math.log2(math.e)

    # Dequantize fp8 inputs to fp32
    q = q_fp8.float()
    k = k_fp8.float()
    v = v_fp8.float()

    # Handle GQA
    k = repeat(k, "b s h d -> b s (h g) d", g=nheads // k_fp8.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=nheads // v_fp8.shape[2])

    # scale_s = (1/sqrt(d)) * log2(e) * q_descale * k_descale
    scale_s = (1.0 / math.sqrt(d)) * log2e * q_descale * k_descale
    raw_scale = (1.0 / math.sqrt(d)) * q_descale * k_descale
    scale_p = fp8_max

    # Initialize online softmax state
    # m: row max (for numerical stability), shape [batch, nheads, seqlen_q]
    # l: row sum of exp, shape [batch, nheads, seqlen_q]
    # o_acc: output accumulator, shape [batch, seqlen_q, nheads, d_v]
    m = torch.full(
        (batch_size, nheads, seqlen_q),
        float("-inf"),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    l = torch.zeros(
        (batch_size, nheads, seqlen_q), device=q_fp8.device, dtype=torch.float32
    )
    o_acc = torch.zeros(
        (batch_size, seqlen_q, nheads, d_v), device=q_fp8.device, dtype=torch.float32
    )

    num_kv_tiles = (seqlen_k + kv_tile_size - 1) // kv_tile_size

    for tile_idx in range(num_kv_tiles):
        kv_start = tile_idx * kv_tile_size
        kv_end = min(kv_start + kv_tile_size, seqlen_k)

        # Get K, V tiles: [batch, tile_len, nheads, d]
        k_tile = k[:, kv_start:kv_end, :, :]
        v_tile = v[:, kv_start:kv_end, :, :]

        # Step 1: Compute S = Q @ K_tile^T for this tile
        # scores: [batch, nheads, seqlen_q, tile_len]
        scores = torch.einsum("bthd,bshd->bhts", q, k_tile)

        # Apply logits soft cap (before masking, on natural-scale logits)
        if logits_soft_cap > 0.0:
            raw_scores = scores * raw_scale
            raw_scores = logits_soft_cap * torch.tanh(raw_scores / logits_soft_cap)
            scores = raw_scores * log2e
            use_scale_s = 1.0
        else:
            use_scale_s = scale_s

        # Apply causal mask for this tile
        if window_size[0] >= 0 or window_size[1] >= 0:
            row_idx = torch.arange(
                seqlen_q, device=q_fp8.device, dtype=torch.long
            ).view(-1, 1)
            col_idx = torch.arange(
                kv_start, kv_end, device=q_fp8.device, dtype=torch.long
            )
            if window_size[0] < 0:
                # Causal only
                mask = col_idx > row_idx + seqlen_k - seqlen_q + window_size[1]
            else:
                mask = torch.logical_or(
                    col_idx > row_idx + seqlen_k - seqlen_q + window_size[1],
                    col_idx < row_idx + seqlen_k - seqlen_q - window_size[0],
                )
            scores.masked_fill_(mask, float("-inf"))

        # Step 2: Online softmax update
        # Compute local row max for this tile
        m_tile = scores.max(dim=-1).values  # [batch, nheads, seqlen_q]

        # Handle fully masked rows
        m_tile = torch.where(
            torch.isinf(m_tile), torch.full_like(m_tile, float("-inf")), m_tile
        )

        # New global max
        m_new = torch.maximum(m, m_tile)
        # Handle case where both are -inf
        m_new = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)

        # Rescaling factors
        # alpha = exp2(use_scale_s * (m_old - m_new)) - rescale old accumulator
        # For rows where m was -inf, alpha should be 0
        alpha = torch.exp2(use_scale_s * (m - m_new))
        alpha = torch.where(torch.isinf(m), torch.zeros_like(alpha), alpha)

        # Compute P = exp2(use_scale_s * (scores - m_new))
        p_compute = torch.exp2(use_scale_s * (scores - m_new.unsqueeze(-1)))

        # Step 3: Rescale o_acc by alpha
        # o_acc shape: [batch, seqlen_q, nheads, d_v]
        # alpha shape: [batch, nheads, seqlen_q] -> [batch, seqlen_q, nheads, 1]
        alpha_for_o = alpha.permute(0, 2, 1).unsqueeze(-1)
        o_acc = o_acc * alpha_for_o

        # Step 4: Rescale l by alpha and add new sum
        l = l * alpha + p_compute.sum(dim=-1)

        # Step 5: Quantize P to fp8 and accumulate PV
        p_fp8 = (p_compute * scale_p).to(dtypes.fp8)
        p_dequant = p_fp8.float()

        # PV GEMM: [batch, nheads, seqlen_q, tile_len] @ [batch, tile_len, nheads, d_v]
        # -> [batch, seqlen_q, nheads, d_v]
        o_tile = torch.einsum("bhts,bshd->bthd", p_dequant, v_tile)
        o_acc = o_acc + o_tile

        # Update m
        m = m_new

    # Final normalization: O = o_acc / l
    l_for_div = l.permute(0, 2, 1).unsqueeze(-1).clamp(min=1e-9)
    output = o_acc / l_for_div

    # Apply output scale
    scale_o = v_descale / scale_p
    output = (output * scale_o).to(torch.bfloat16)

    return output


def attention_fp8_ref(
    q_fp8,
    k_fp8,
    v_fp8,
    q_descale: float,
    k_descale: float,
    v_descale: float,
    causal=False,
    window_size=(-1, -1),
):
    """
    Reference implementation for FP8 FMHA kernel computation.

    Simulates the FP8 flash attention kernel computation flow with FAST_EXP2 path:
    1. QK^T GEMM: fp8 x fp8 -> fp32 (no scaling yet)
    2. Online softmax using exp2: scale_s includes log2(e) factor
       - scale_s = (1/sqrt(d)) * log2(e) * q_descale * k_descale
       - p = exp2(scale_s * (scores - row_max))
    3. P quantization: P_fp8 = (P_fp32 * scale_p).to(fp8), where scale_p = fp8_max
    4. PV GEMM: fp8 x fp8 -> fp32
    5. Output conversion: O_bf16 = (O_fp32 * scale_o).to(bf16), where scale_o = v_descale / scale_p

    Arguments:
        q_fp8: (batch_size, seqlen_q, nheads, head_dim) - fp8 quantized query
        k_fp8: (batch_size, seqlen_k, nheads_k, head_dim) - fp8 quantized key
        v_fp8: (batch_size, seqlen_k, nheads_k, head_dim_v) - fp8 quantized value
        q_descale: scale factor for dequantizing q
        k_descale: scale factor for dequantizing k
        v_descale: scale factor for dequantizing v
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size, -1 means infinite

    Returns:
        output: (batch_size, seqlen_q, nheads, head_dim_v) - bf16
    """
    if causal:
        window_size = (window_size[0], 0)

    seqlen_q, seqlen_k = q_fp8.shape[1], k_fp8.shape[1]
    d = q_fp8.shape[-1]

    # FP8 E4M3 max value
    fp8_max = torch.finfo(dtypes.fp8).max  # 448.0

    # log2(e) for FAST_EXP2 path
    log2e = math.log2(math.e)  # 1.4426950408889634

    # Dequantize fp8 inputs to fp32 for GEMM simulation
    q = q_fp8.float()
    k = k_fp8.float()
    v = v_fp8.float()

    # Handle GQA (grouped query attention)
    k = repeat(k, "b s h d -> b s (h g) d", g=q_fp8.shape[2] // k_fp8.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q_fp8.shape[2] // v_fp8.shape[2])

    # Step 1: QK^T GEMM (fp8 x fp8 -> fp32)
    # In FAST_EXP2 path, scores are NOT scaled here
    scores = torch.einsum("bthd,bshd->bhts", q, k)

    # Apply causal/local mask (aligned with attention_ref implementation)
    if window_size[0] >= 0 or window_size[1] >= 0:
        row_idx = torch.arange(seqlen_q, device=q_fp8.device, dtype=torch.long).view(
            -1, 1
        )
        col_idx = torch.arange(seqlen_k, device=q_fp8.device, dtype=torch.long)
        if window_size[0] < 0:
            # Causal only (no left window limit)
            mask = col_idx > row_idx + seqlen_k - seqlen_q + window_size[1]
        else:
            # Sliding window attention
            mask = torch.logical_or(
                col_idx > row_idx + seqlen_k - seqlen_q + window_size[1],
                col_idx < row_idx + seqlen_k - seqlen_q - window_size[0],
            )
        scores.masked_fill_(mask, float("-inf"))

    # Step 2: Softmax using exp2 (FAST_EXP2 path) - online softmax style
    # scale_s = (1/sqrt(d)) * log2(e) * q_descale * k_descale
    scale_s = (1.0 / math.sqrt(d)) * log2e * q_descale * k_descale

    # Compute row max for numerical stability
    row_max = scores.max(dim=-1, keepdim=True).values
    # Handle fully masked rows (all -inf) to avoid NaN from -inf - (-inf)
    row_max = torch.where(torch.isinf(row_max), torch.zeros_like(row_max), row_max)

    # IMPORTANT: This is UNNORMALIZED - kernel does NOT divide by sum before quantization
    p_compute = torch.exp2(scale_s * (scores - row_max))

    # Step 3: P quantization (fp32 -> fp8)
    # Kernel quantizes UNNORMALIZED p_compute: P_fp8 = (p_compute * scale_p).to(fp8)
    scale_p = fp8_max
    p_fp8 = (p_compute * scale_p).to(dtypes.fp8)

    # Step 4: PV GEMM (fp8 x fp8 -> fp32)
    # Dequantize p_fp8 back to float for computation
    p_dequant = p_fp8.float()
    output = torch.einsum("bhts,bshd->bthd", p_dequant, v)

    # Step 5: Output normalization and scaling
    # In kernel: o_acc *= 1/l (where l is sum of unnormalized p)
    # Then: o_acc *= scale_o where scale_o = v_descale / scale_p
    p_sum = p_compute.sum(dim=-1, keepdim=True)  # Sum of unnormalized p
    # p_sum shape is [b, h, t, 1], but output shape is [b, t, h, d]
    # Need to permute p_sum to match: [b, h, t, 1] -> [b, t, h, 1]
    p_sum_for_div = p_sum.permute(0, 2, 1, 3)
    output = output / p_sum_for_div.clamp(min=1e-9)

    scale_o = v_descale / scale_p
    output = (output * scale_o).to(torch.bfloat16)

    return output


def attention_varlen_fp8_ref_online(
    q_fp8,
    k_fp8,
    v_fp8,
    cu_seqlens_q,
    cu_seqlens_k,
    q_descale: float,
    k_descale: float,
    v_descale: float,
    causal=False,
    logits_soft_cap=0.0,
    kv_tile_size=128,
):
    """
    Reference implementation simulating online softmax with KV tiling for varlen.

    Processes each sequence independently (group_mode), same FP8 computation
    flow as the kernel (FAST_EXP2 path with online softmax).

    Arguments:
        q_fp8: (total_q, nheads, head_dim) - fp8 quantized query
        k_fp8: (total_k, nheads_k, head_dim) - fp8 quantized key
        v_fp8: (total_k, nheads_k, head_dim) - fp8 quantized value
        cu_seqlens_q: (batch_size + 1,) cumulative Q sequence lengths
        cu_seqlens_k: (batch_size + 1,) cumulative K sequence lengths
        q_descale, k_descale, v_descale: per-tensor descale factors (scalar)
        causal: whether to apply causal masking
        logits_soft_cap: soft cap for logits (0.0 = disabled)
        kv_tile_size: tile size for KV dimension in online softmax

    Returns:
        output: (total_q, nheads, head_dim) - bf16
    """
    nheads = q_fp8.shape[1]
    nheads_k = k_fp8.shape[1]
    d = q_fp8.shape[2]
    batch_size = len(cu_seqlens_q) - 1

    fp8_max = torch.finfo(dtypes.fp8).max
    log2e = math.log2(math.e)

    scale_s = (1.0 / math.sqrt(d)) * log2e * q_descale * k_descale
    scale_p = fp8_max
    gqa_ratio = nheads // nheads_k

    total_q = q_fp8.shape[0]
    output = torch.zeros(total_q, nheads, d, dtype=torch.bfloat16, device=q_fp8.device)

    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

        # Dequantize
        q = q_fp8[q_start:q_end].float()  # [sq, h, d]
        k = k_fp8[k_start:k_end].float()  # [sk, hk, d]
        v = v_fp8[k_start:k_end].float()  # [sk, hk, d]

        # GQA expansion
        if gqa_ratio > 1:
            k = repeat(k, "s h d -> s (h g) d", g=gqa_ratio)
            v = repeat(v, "s h d -> s (h g) d", g=gqa_ratio)

        # Initialize online softmax state for this sequence
        m = torch.full(
            (nheads, seqlen_q), float("-inf"), device=q_fp8.device, dtype=torch.float32
        )
        l = torch.zeros((nheads, seqlen_q), device=q_fp8.device, dtype=torch.float32)
        o_acc = torch.zeros(
            (seqlen_q, nheads, d), device=q_fp8.device, dtype=torch.float32
        )

        num_kv_tiles = (seqlen_k + kv_tile_size - 1) // kv_tile_size

        for tile_idx in range(num_kv_tiles):
            kv_start = tile_idx * kv_tile_size
            kv_end = min(kv_start + kv_tile_size, seqlen_k)

            k_tile = k[kv_start:kv_end]  # [tile_len, h, d]
            v_tile = v[kv_start:kv_end]  # [tile_len, h, d]

            # QK^T: [h, sq, tile_len]
            scores = torch.einsum("qhd,khd->hqk", q, k_tile)

            # Logits soft cap (applied before exp2 scaling)
            if logits_soft_cap > 0.0:
                raw_scale = (1.0 / math.sqrt(d)) * q_descale * k_descale
                raw_scores = scores * raw_scale
                raw_scores = logits_soft_cap * torch.tanh(raw_scores / logits_soft_cap)
                # Convert to log2 scale for exp2
                scores_scaled = raw_scores * log2e
                use_scale_s = 1.0
            else:
                scores_scaled = scores
                use_scale_s = scale_s

            # Causal mask
            if causal:
                row_idx = torch.arange(
                    seqlen_q, device=q_fp8.device, dtype=torch.long
                ).view(-1, 1)
                col_idx = torch.arange(
                    kv_start, kv_end, device=q_fp8.device, dtype=torch.long
                )
                mask = col_idx > row_idx + seqlen_k - seqlen_q
                scores_scaled.masked_fill_(mask, float("-inf"))

            # Online softmax update
            m_tile = scores_scaled.max(dim=-1).values  # [h, sq]
            m_tile = torch.where(
                torch.isinf(m_tile), torch.full_like(m_tile, float("-inf")), m_tile
            )

            m_new = torch.maximum(m, m_tile)
            m_new = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)

            alpha = torch.exp2(use_scale_s * (m - m_new))
            alpha = torch.where(torch.isinf(m), torch.zeros_like(alpha), alpha)

            p_compute = torch.exp2(use_scale_s * (scores_scaled - m_new.unsqueeze(-1)))

            # Rescale o_acc: [sq, h, d] *= alpha [h, sq] -> [sq, h, 1]
            alpha_for_o = alpha.permute(1, 0).unsqueeze(-1)
            o_acc = o_acc * alpha_for_o

            l = l * alpha + p_compute.sum(dim=-1)

            # Quantize P to fp8
            p_fp8 = (p_compute * scale_p).to(dtypes.fp8)
            p_dequant = p_fp8.float()

            # PV: [h, sq, tile_len] @ [tile_len, h, d] -> [sq, h, d]
            o_tile = torch.einsum("hqk,khd->qhd", p_dequant, v_tile)
            o_acc = o_acc + o_tile

            m = m_new

        # Final normalization
        l_for_div = l.permute(1, 0).unsqueeze(-1).clamp(min=1e-9)  # [sq, h, 1]
        o_seq = o_acc / l_for_div

        scale_o = v_descale / scale_p
        output[q_start:q_end] = (o_seq * scale_o).to(torch.bfloat16)

    return output


@perftest()
def profile_func(target_func, *args, **kwargs):
    return target_func(*args, **kwargs)


def run_ck(
    q,
    k,
    v,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    q_descale=None,
    k_descale=None,
    v_descale=None,
    logits_soft_cap=0.0,
    profile=False,
):
    if q.dtype == dtypes.fp8 and k.dtype == dtypes.fp8 and v.dtype == dtypes.fp8:
        if profile:
            return profile_func(
                aiter.flash_attn_fp8_pertensor_func,
                q,
                k,
                v,
                q_descale,
                k_descale,
                v_descale,
                causal=causal,
                window_size=window_size,
                logits_soft_cap=logits_soft_cap,
            )
        return run_perftest(
            aiter.flash_attn_fp8_pertensor_func,
            q,
            k,
            v,
            q_descale,
            k_descale,
            v_descale,
            causal=causal,
            window_size=window_size,
            logits_soft_cap=logits_soft_cap,
            num_iters=2,
            num_warmup=0,
        )
    else:
        if profile:
            return profile_func(
                aiter.flash_attn_func,
                q,
                k,
                v,
                dropout_p=0.0,
                causal=causal,
                window_size=window_size,
                logits_soft_cap=logits_soft_cap,
                bias=None,
                alibi_slopes=None,
                deterministic=True,
                return_lse=False,
                return_attn_probs=False,
            )
        return run_perftest(
            aiter.flash_attn_func,
            q,
            k,
            v,
            dropout_p=0.0,
            causal=causal,
            window_size=window_size,
            logits_soft_cap=logits_soft_cap,
            bias=None,
            alibi_slopes=None,
            deterministic=True,
            return_lse=False,
            return_attn_probs=False,
            num_iters=2,
            num_warmup=0,
        )


def run_ck_varlen(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=False,
    logits_soft_cap=0.0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    profile=False,
):
    if q.dtype == dtypes.fp8 and k.dtype == dtypes.fp8 and v.dtype == dtypes.fp8:
        if profile:
            return profile_func(
                aiter.flash_attn_varlen_fp8_pertensor_func,
                q,
                k,
                v,
                q_descale,
                k_descale,
                v_descale,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                min_seqlen_q=0,
                causal=causal,
                logits_soft_cap=logits_soft_cap,
                window_size=(-1, -1),
            )
        return run_perftest(
            aiter.flash_attn_varlen_fp8_pertensor_func,
            q,
            k,
            v,
            q_descale,
            k_descale,
            v_descale,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=0,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            window_size=(-1, -1),
            num_iters=2,
            num_warmup=0,
        )
    else:
        if profile:
            return profile_func(
                aiter.flash_attn_varlen_func,
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=causal,
                logits_soft_cap=logits_soft_cap,
                return_lse=False,
            )
        return run_perftest(
            aiter.flash_attn_varlen_func,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            return_lse=False,
            num_iters=2,
            num_warmup=0,
        )


# @pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("nheads, nheads_k", [(8, 1), (40, 8), (32, 8), (5, 1)])
@pytest.mark.parametrize(
    "d,d_v",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        (4096, 4096),
    ],
)
def test_flash_attn_fp8_output(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    causal,
    local,
    logits_soft_cap,
    kv_tile_size=128,
    profile=False,
):
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    dtype = torch.bfloat16
    quant_dtype = dtypes.fp8

    q = torch.rand(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.rand(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device="cuda",
        dtype=dtype,
    )
    v = torch.rand(
        batch_size,
        seqlen_k,
        nheads_k,
        d_v,
        device="cuda",
        dtype=dtype,
    )

    q_quant, q_descale = per_tensor_quant(q, quant_dtype=quant_dtype)
    k_quant, k_descale = per_tensor_quant(k, quant_dtype=quant_dtype)
    v_quant, v_descale = per_tensor_quant(v, quant_dtype=quant_dtype)

    out, us_quant_fwd = run_ck(
        q_quant,
        k_quant,
        v_quant,
        causal,
        window_size,
        q_descale,
        k_descale,
        v_descale,
        logits_soft_cap=logits_soft_cap,
        profile=profile,
    )

    out_ref = attention_fp8_ref_online(
        q_quant,
        k_quant,
        v_quant,
        q_descale.item(),
        k_descale.item(),
        v_descale.item(),
        causal=causal,
        window_size=window_size,
        logits_soft_cap=logits_soft_cap,
        kv_tile_size=kv_tile_size,
    )

    max_diff = (out - out_ref).abs().max().item()
    print(f"Output max diff: {max_diff}")
    threshold = 0.025 if logits_soft_cap > 0.0 else 0.02
    assert max_diff < threshold

    fwd_flop = (
        batch_size
        * nheads
        * (seqlen_q * seqlen_k * d * 2 + seqlen_q * seqlen_k * d_v * 2)
        // (2 if causal else 1)
    )

    quant_dtype_bytes = torch.finfo(quant_dtype).bits // 8
    quant_fwd_num_bytes = (
        batch_size
        * nheads
        * quant_dtype_bytes
        * (seqlen_q * d + seqlen_k * d + seqlen_k * d_v + seqlen_q * d_v)
    )

    benchmark["quant_fwd_us"] = us_quant_fwd
    benchmark["quant_fwd_tflops"] = (fwd_flop) / 1.0e6 / us_quant_fwd
    benchmark["quant_fwd_gb_per_sec"] = (quant_fwd_num_bytes) / 1.0e3 / us_quant_fwd


def convert_lens_to_indptr(lens):
    return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()


@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("nheads, nheads_k", [(8, 1), (40, 8), (32, 8), (5, 1)])
@pytest.mark.parametrize(
    "d,d_v",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        (4096, 4096),
    ],
)
def test_flash_attn_varlen_fp8_output(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    causal,
    logits_soft_cap,
    kv_tile_size=128,
    profile=False,
):
    """
    Test flash_attn_varlen_fp8_pertensor_func against online-softmax reference.

    Uses the same FP8 computation flow as test_flash_attn_fp8_output but with
    varlen (group_mode) layout: (total_tokens, nheads, head_dim).
    """
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    dtype = torch.bfloat16
    quant_dtype = dtypes.fp8

    # Build variable-length sequences
    if batch_size > 1:
        qo_lens = torch.randint(1, seqlen_q + 1, (batch_size,)).int()
        kv_lens = torch.randint(1, seqlen_k + 1, (batch_size,)).int()
        if causal:
            kv_lens = torch.maximum(qo_lens, kv_lens)
    else:
        qo_lens = torch.full((batch_size,), seqlen_q).int()
        kv_lens = torch.full((batch_size,), seqlen_k).int()

    total_q = qo_lens.sum().item()
    total_k = kv_lens.sum().item()
    max_sq = qo_lens.max().item()
    max_sk = kv_lens.max().item()

    cu_seqlens_q = convert_lens_to_indptr(qo_lens).cuda()
    cu_seqlens_k = convert_lens_to_indptr(kv_lens).cuda()

    # Create BF16 tensors in varlen layout, then quantize
    q = torch.rand(total_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.rand(total_k, nheads_k, d, device="cuda", dtype=dtype)
    v = torch.rand(total_k, nheads_k, d_v, device="cuda", dtype=dtype)

    q_quant, q_descale = per_tensor_quant(q, quant_dtype=quant_dtype)
    k_quant, k_descale = per_tensor_quant(k, quant_dtype=quant_dtype)
    v_quant, v_descale = per_tensor_quant(v, quant_dtype=quant_dtype)

    out, us_quant_fwd = run_ck_varlen(
        q_quant,
        k_quant,
        v_quant,
        cu_seqlens_q,
        cu_seqlens_k,
        max_sq,
        max_sk,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        profile=profile,
    )

    out_ref = attention_varlen_fp8_ref_online(
        q_quant,
        k_quant,
        v_quant,
        cu_seqlens_q.cpu(),
        cu_seqlens_k.cpu(),
        q_descale.item(),
        k_descale.item(),
        v_descale.item(),
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        kv_tile_size=kv_tile_size,
    )

    max_diff = (out - out_ref).abs().max().item()
    print(
        f"Varlen FP8 | b={batch_size} sq={seqlen_q} sk={seqlen_k} "
        f"h={nheads}/{nheads_k} causal={causal} lsc={logits_soft_cap} | "
        f"max diff: {max_diff}"
    )
    assert max_diff < 0.02

    fwd_flop = (
        batch_size
        * nheads
        * (max_sq * max_sk * d * 2 + max_sq * max_sk * d_v * 2)
        // (2 if causal else 1)
    )

    benchmark["varlen_quant_fwd_us"] = us_quant_fwd
    benchmark["varlen_quant_fwd_tflops"] = (fwd_flop) / 1.0e6 / us_quant_fwd


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["regular", "varlen"],
    default="regular",
    help="""Test mode: 'regular' for dense format, 'varlen' for variable length format.
    e.g.: --mode varlen""",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=2,
    help="""Batch size. Default is 2.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nheads",
    type=int,
    default=5,
    help="""Number of heads. Default is 5.
    e.g.: -n 8""",
)
parser.add_argument(
    "-nk",
    "--nheads_k",
    type=int,
    default=-1,
    help="""Number of KV heads. -1 means equal to n (nheads).
    e.g.: -nk 1""",
)
parser.add_argument(
    "-q",
    "--seqlen_q",
    type=int,
    default=512,
    help="""Sequence length for query. Default is 512.
    e.g.: -q 1024""",
)
parser.add_argument(
    "-k",
    "--seqlen_k",
    type=int,
    default=-1,
    help="""Sequence length for key. -1 means equal to q (seqlen_q).
    e.g.: -k 1024""",
)
parser.add_argument(
    "-d",
    "--d_qk",
    type=int,
    default=128,
    help="""Dimension of query and key. Default is 128.
    e.g.: -d 128""",
)
parser.add_argument(
    "-dv",
    "--d_v",
    type=int,
    default=-1,
    help="""Dimension of value. -1 means equal to d (d_qk).
    e.g.: -dv 128""",
)
parser.add_argument(
    "-c",
    "--causal",
    action="store_true",
    help="""Causal attention. Default is False.
    -c or --causal    # enable causal attention""",
)
parser.add_argument(
    "-l",
    "--local",
    action="store_true",
    help="""Local attention. Default is False.
    -l or --local    # enable local attention""",
)
parser.add_argument(
    "--logits_soft_cap",
    type=float,
    default=0.0,
    help="""Logits soft cap. Default is 0.0 (disabled).
    e.g.: --logits_soft_cap 30.0""",
)
parser.add_argument(
    "-p",
    "--profile",
    action="store_true",
    help="""Profile mode: run kernel without warmup for profiling.
    -p or --profile    # enable profile mode""",
)

if __name__ == "__main__":
    args = parser.parse_args()

    nheads_k = args.nheads_k if args.nheads_k > 0 else args.nheads
    seqlen_k = args.seqlen_k if args.seqlen_k > 0 else args.seqlen_q
    d_v = args.d_v if args.d_v > 0 else args.d_qk

    if args.mode == "regular":
        test_flash_attn_fp8_output(
            args.batch_size,
            args.nheads,
            nheads_k,
            args.seqlen_q,
            seqlen_k,
            args.d_qk,
            d_v,
            args.causal,
            args.local,
            args.logits_soft_cap,
            profile=args.profile,
        )
    elif args.mode == "varlen":
        test_flash_attn_varlen_fp8_output(
            args.batch_size,
            args.nheads,
            nheads_k,
            args.seqlen_q,
            seqlen_k,
            args.d_qk,
            d_v,
            args.causal,
            args.logits_soft_cap,
            profile=args.profile,
        )

    df = pd.DataFrame([benchmark])
    aiter.logger.info(f"mha summary:\n{df}")
