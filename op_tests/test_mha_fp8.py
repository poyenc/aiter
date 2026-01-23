# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
import torch
import aiter
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter import per_tensor_quant
from einops import repeat
import pytest
import pandas as pd
import argparse

benchmark = {}


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

    Simulates the FP8 flash attention kernel computation flow:
    1. QK^T GEMM: fp8 x fp8 -> fp32, scaled by (1/sqrt(d)) * q_descale * k_descale
    2. Softmax: fp32 -> fp32
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

    # Dequantize fp8 inputs to fp32 for GEMM simulation
    q = q_fp8.float()
    k = k_fp8.float()
    v = v_fp8.float()

    # Handle GQA (grouped query attention)
    k = repeat(k, "b s h d -> b s (h g) d", g=q_fp8.shape[2] // k_fp8.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q_fp8.shape[2] // v_fp8.shape[2])

    # Step 1: QK^T GEMM (fp8 x fp8 -> fp32)
    # Combined scale: scale_s = (1/sqrt(d)) * q_descale * k_descale
    scale_s = (1.0 / math.sqrt(d)) * q_descale * k_descale
    scores = torch.einsum("bthd,bshd->bhts", q, k) * scale_s

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

    # Step 2: Softmax (fp32 -> fp32)
    p = torch.softmax(scores, dim=-1)

    # Step 3: P quantization (fp32 -> fp8)
    # scale_p = fp8_max, P_fp8 = (P * scale_p).to(fp8)
    scale_p = fp8_max
    p_fp8 = (p * scale_p).to(dtypes.fp8)

    # Step 4: PV GEMM (fp8 x fp8 -> fp32)
    # Dequantize p_fp8 back to float for computation
    p_dequant = p_fp8.float()
    output = torch.einsum("bhts,bshd->bthd", p_dequant, v)

    # Step 5: Output scaling and conversion (fp32 -> bf16)
    # scale_o = v_descale / scale_p
    scale_o = v_descale / scale_p
    output = (output * scale_o).to(torch.bfloat16)

    return output


def run_ck(
    q,
    k,
    v,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    if q.dtype == dtypes.fp8 and k.dtype == dtypes.fp8 and v.dtype == dtypes.fp8:
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
        )
    else:
        return run_perftest(
            aiter.flash_attn_func,
            q,
            k,
            v,
            dropout_p=0.0,
            causal=causal,
            window_size=window_size,
            bias=None,
            alibi_slopes=None,
            deterministic=True,
            return_lse=False,
            return_attn_probs=False,
        )


# @pytest.mark.parametrize("local", [False, True])
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
def test_flash_attn_output(
    batch_size, nheads, nheads_k, seqlen_q, seqlen_k, d, d_v, causal, local
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
    )
    out_ref, us_fwd = run_ck(q, k, v, causal, window_size)

    max_diff = (out - out_ref).abs().max().item()
    print(f"Output max diff: {max_diff}")
    assert max_diff < 0.055

    fwd_flop = (
        batch_size
        * nheads
        * (seqlen_q * seqlen_k * d * 2 + seqlen_q * seqlen_k * d_v * 2)
    )

    dtype_bytes = torch.finfo(dtype).bits // 8
    quant_dtype_bytes = torch.finfo(quant_dtype).bits // 8

    fwd_num_bytes = (
        batch_size
        * nheads
        * dtype_bytes
        * (seqlen_q * d + seqlen_k * d + seqlen_k * d_v + seqlen_q * d_v)
    )
    quant_fwd_num_bytes = (
        batch_size
        * nheads
        * quant_dtype_bytes
        * (seqlen_q * d + seqlen_k * d + seqlen_k * d_v + seqlen_q * d_v)
    )

    benchmark["quant_fwd_us"] = us_quant_fwd
    benchmark["quant_fwd_tflops"] = (fwd_flop) / 1.0e6 / us_quant_fwd
    benchmark["quant_fwd_gb_per_sec"] = (quant_fwd_num_bytes) / 1.0e3 / us_quant_fwd
    benchmark["fwd_us"] = us_fwd
    benchmark["fwd_tflops"] = (fwd_flop) / 1.0e6 / us_fwd
    benchmark["fwd_gb_per_sec"] = (fwd_num_bytes) / 1.0e3 / us_fwd


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
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
    help="""Number of heads. -1 means equal to n (nheads).
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
    help="""Dimension of query and key. -1 means equal to d (d_qk).
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

if __name__ == "__main__":
    args = parser.parse_args()

    nheads_k = args.nheads_k if args.nheads_k > 0 else args.nheads
    seqlen_k = args.seqlen_k if args.seqlen_k > 0 else args.seqlen_q
    d_v = args.d_v if args.d_v > 0 else args.d_qk

    collected = []
    test_flash_attn_output(
        args.batch_size,
        args.nheads,
        nheads_k,
        args.seqlen_q,
        seqlen_k,
        args.d_qk,
        d_v,
        args.causal,
        args.local,
    )
    collected.append(benchmark)

    df = pd.DataFrame(collected)
    aiter.logger.info(f"mha summary:\n{df}")
