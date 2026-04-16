# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark CK FMHA V3 kernels across all three paths:
  1. batch_prefill  — paged KV (linear layout)
  2. fmha_regular   — contiguous KV, batch/group mode (FP8 per-tensor)
  3. fmha_varlen    — contiguous KV, variable-length mode

Supports BF16 and FP8 input dtypes. Reports time (us) and TFlops.

Usage:
    # Single problem size
    python op_tests/bench_fmha_v3.py -b 1 -n 6 -nk 1 -q 1024 -c --dtype fp8

    # Compare all three modes side-by-side
    python op_tests/bench_fmha_v3.py -b 1 -n 6 -nk 1 -q 4096 -c --dtype fp8 --mode all

    # Batch prefill only (ps=1, sglang, pertensor — default)
    python op_tests/bench_fmha_v3.py -b 1 -n 6 -nk 1 -q 4096 -c --dtype fp8 --mode batch_prefill

    # Batch prefill with vllm lookup table
    python op_tests/bench_fmha_v3.py --sweep --dtype fp8 --mode batch_prefill --lookup_table vllm

    # Batch prefill with page_size=1024 and blockscale
    python op_tests/bench_fmha_v3.py --sweep --dtype fp8 --mode batch_prefill --quant kv_blockscale --page_size 1024

    # Sweep predefined problem sizes
    python op_tests/bench_fmha_v3.py --sweep --dtype fp8
"""

import argparse
import torch

import aiter
from aiter import dtypes
from aiter import per_tensor_quant
from aiter.test_common import perftest

from aiter.test_mha_common import (
    generate_random_padding_mask,
    generate_qkv,
)

from test_batch_prefill import (
    build_block_table,
    build_paged_kv_cache,
    convert_lens_to_indptr,
    split_kv_pages,
    per_page_quant,
)


@perftest()
def profile_func(target_func, *args, **kwargs):
    return target_func(*args, **kwargs)


def flops(batch, seqlen_q, seqlen_k, headdim, nheads_q, nheads_k, causal):
    mask_area = seqlen_q * seqlen_k // (2 if causal else 1)
    qk = 2 * batch * mask_area * nheads_q * headdim
    pv = 2 * batch * mask_area * nheads_q * headdim
    return qk + pv


def tflops(flop, time_us):
    return flop / time_us / 1e6


def run_batch_prefill(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    head_dim,
    causal,
    logits_soft_cap,
    is_fp8,
    page_size=1,
    kv_lookup_table="sglang",
):
    # Match fmha_varlen: padded Q + random padding mask + unpad
    dtype = torch.bfloat16

    q_padded = torch.randn(
        batch_size, seqlen_q, nheads, head_dim, device="cuda", dtype=dtype
    )
    query_padding_mask = generate_random_padding_mask(
        seqlen_q, batch_size, "cuda", mode="random"
    )
    qo_lens = query_padding_mask.sum(dim=-1).to(torch.int32).cpu()
    total_q = qo_lens.sum().item()
    max_qo_len = qo_lens.max().item()
    # Unpad Q: select valid tokens per sequence
    q = torch.cat(
        [q_padded[i, : qo_lens[i]] for i in range(batch_size)], dim=0
    )  # [total_q, nheads, head_dim]

    kv_lens = torch.full((batch_size,), seqlen_k).int()

    kv_cache = build_paged_kv_cache(
        batch_size,
        seqlen_k,
        page_size,
        nheads_k,
        head_dim,
        kv_lens,
        None if is_fp8 else -5,
        None if is_fp8 else 5,
        dtype,
        use_uniform=is_fp8,
        contiguous_kv=True,
    )

    q_indptr = convert_lens_to_indptr(qo_lens).cuda()
    kv_indptr = kv_cache["kv_indptr_cpu"].cuda()
    kv_indices = kv_cache["kv_indices_cpu"].cuda()
    kv_last_page_len = kv_cache["kv_last_page_len_cpu"].cuda()

    k_cache_ref, v_cache_ref = split_kv_pages(kv_cache["kv_data"])
    k_cache = k_cache_ref.contiguous()
    v_cache = v_cache_ref.contiguous()

    # vllm lookup table uses block_table instead of kv_indices
    block_table_gpu = None
    seqlen_k_tensor = None
    if kv_lookup_table == "vllm":
        max_num_pages = (seqlen_k + page_size - 1) // page_size
        block_table_cpu = build_block_table(
            kv_cache["kv_indptr_cpu"],
            kv_cache["kv_indices_cpu"],
            batch_size,
            max_num_pages,
        )
        block_table_gpu = block_table_cpu.cuda()
        seqlen_k_tensor = kv_lens.cuda().int()

    if is_fp8:
        q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
        k_quant, k_descale = per_tensor_quant(k_cache.to(dtype), quant_dtype=dtypes.fp8)
        v_quant, v_descale = per_tensor_quant(v_cache.to(dtype), quant_dtype=dtypes.fp8)
        q_quant, q_descale = q_quant.detach(), q_descale.detach()
        k_quant, k_descale = k_quant.detach(), k_descale.detach()
        v_quant, v_descale = v_quant.detach(), v_descale.detach()

        out, time_us = profile_func(
            aiter.mha_batch_prefill_func,
            q_quant,
            k_quant,
            v_quant,
            q_indptr,
            kv_indptr,
            kv_indices,
            max_qo_len,
            seqlen_k,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            kv_last_page_lens=kv_last_page_len,
            block_table=block_table_gpu,
            seqlen_k=seqlen_k_tensor,
        )
    else:
        out, time_us = profile_func(
            aiter.mha_batch_prefill_func,
            q,
            k_cache,
            v_cache,
            q_indptr,
            kv_indptr,
            kv_indices,
            max_qo_len,
            seqlen_k,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            kv_last_page_lens=kv_last_page_len,
            block_table=block_table_gpu,
            seqlen_k=seqlen_k_tensor,
        )

    return time_us


def run_batch_prefill_blockscale(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    head_dim,
    causal,
    logits_soft_cap,
    is_fp8,
    page_size=1024,
    kv_lookup_table="sglang",
):
    assert is_fp8, "kv_blockscale only supports fp8"
    dtype = torch.bfloat16

    q_padded = torch.randn(
        batch_size, seqlen_q, nheads, head_dim, device="cuda", dtype=dtype
    )
    query_padding_mask = generate_random_padding_mask(
        seqlen_q, batch_size, "cuda", mode="random"
    )
    qo_lens = query_padding_mask.sum(dim=-1).to(torch.int32).cpu()
    max_qo_len = qo_lens.max().item()
    q = torch.cat([q_padded[i, : qo_lens[i]] for i in range(batch_size)], dim=0)

    kv_lens = torch.full((batch_size,), seqlen_k).int()

    kv_cache = build_paged_kv_cache(
        batch_size,
        seqlen_k,
        page_size,
        nheads_k,
        head_dim,
        kv_lens,
        None,
        None,
        dtype,
        use_uniform=True,
        contiguous_kv=True,
    )

    q_indptr = convert_lens_to_indptr(qo_lens).cuda()
    kv_indptr = kv_cache["kv_indptr_cpu"].cuda()
    kv_indices = kv_cache["kv_indices_cpu"].cuda()
    kv_last_page_len = kv_cache["kv_last_page_len_cpu"].cuda()

    k_cache_ref, v_cache_ref = split_kv_pages(kv_cache["kv_data"])
    k_cache = k_cache_ref.contiguous()
    v_cache = v_cache_ref.contiguous()

    # Per-tensor Q quantization
    q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
    q_quant, q_descale = q_quant.detach(), q_descale.detach()

    # Per-page K/V quantization
    # k_cache/v_cache are [num_pages, page_size, nheads_k, head_dim]
    k_paged_fp8, k_descales = per_page_quant(k_cache.to(dtype), page_size, dtypes.fp8)
    v_paged_fp8, v_descales = per_page_quant(v_cache.to(dtype), page_size, dtypes.fp8)
    kv_block_descale = torch.stack([k_descales, v_descales], dim=-1)

    # Build block_table and seqlen_k tensor (required by kv_blockscale path)
    max_num_pages = (seqlen_k + page_size - 1) // page_size
    block_table = torch.zeros((batch_size, max_num_pages), dtype=torch.int32)
    kv_indices_cpu = kv_cache["kv_indices_cpu"]
    kv_indptr_cpu = kv_cache["kv_indptr_cpu"]
    for i in range(batch_size):
        start, end = kv_indptr_cpu[i].item(), kv_indptr_cpu[i + 1].item()
        block_table[i, : (end - start)] = kv_indices_cpu[start:end]
    block_table = block_table.cuda()
    seqlen_k_tensor = kv_lens.cuda().int()

    out, time_us = profile_func(
        aiter.mha_batch_prefill_func,
        q_quant,
        k_paged_fp8,
        v_paged_fp8,
        q_indptr,
        kv_indptr,
        kv_indices,
        max_qo_len,
        seqlen_k,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        q_descale=q_descale,
        kv_block_descale=kv_block_descale,
        kv_last_page_lens=kv_last_page_len,
        block_table=block_table,
        seqlen_k=seqlen_k_tensor,
    )

    return time_us


def run_fmha_regular(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    head_dim,
    causal,
    logits_soft_cap,
    is_fp8,
):
    dtype = torch.bfloat16

    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(
        batch_size, seqlen_k, nheads_k, head_dim, device="cuda", dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads_k, head_dim, device="cuda", dtype=dtype
    )

    if is_fp8:
        q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
        k_quant, k_descale = per_tensor_quant(k, quant_dtype=dtypes.fp8)
        v_quant, v_descale = per_tensor_quant(v, quant_dtype=dtypes.fp8)
        q_quant, q_descale = q_quant.detach(), q_descale.detach()
        k_quant, k_descale = k_quant.detach(), k_descale.detach()
        v_quant, v_descale = v_quant.detach(), v_descale.detach()

        out, time_us = profile_func(
            aiter.flash_attn_fp8_pertensor_func,
            q_quant,
            k_quant,
            v_quant,
            q_descale,
            k_descale,
            v_descale,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
        )
    else:
        out, time_us = profile_func(
            aiter.fmha_v3_fwd_ck_func,
            q,
            k,
            v,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
        )

    return time_us


def convert_lens_to_indptr(lens):
    return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()


def run_fmha_varlen(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    head_dim,
    causal,
    logits_soft_cap,
    is_fp8,
):
    dtype = torch.bfloat16

    if is_fp8:
        # Match test_mha_fp8.py: create tensors directly in varlen layout
        qo_lens = torch.full((batch_size,), seqlen_q).int()
        kv_lens = torch.full((batch_size,), seqlen_k).int()
        total_q = qo_lens.sum().item()
        total_k = kv_lens.sum().item()

        cu_seqlens_q = convert_lens_to_indptr(qo_lens).cuda()
        cu_seqlens_k = convert_lens_to_indptr(kv_lens).cuda()

        q = torch.rand(total_q, nheads, head_dim, device="cuda", dtype=dtype)
        k = torch.rand(total_k, nheads_k, head_dim, device="cuda", dtype=dtype)
        v = torch.rand(total_k, nheads_k, head_dim, device="cuda", dtype=dtype)

        q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
        k_quant, k_descale = per_tensor_quant(k, quant_dtype=dtypes.fp8)
        v_quant, v_descale = per_tensor_quant(v, quant_dtype=dtypes.fp8)
        q_quant, q_descale = q_quant.detach(), q_descale.detach()
        k_quant, k_descale = k_quant.detach(), k_descale.detach()
        v_quant, v_descale = v_quant.detach(), v_descale.detach()

        out, time_us = profile_func(
            aiter.flash_attn_varlen_fp8_pertensor_func,
            q_quant,
            k_quant,
            v_quant,
            q_descale,
            k_descale,
            v_descale,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            min_seqlen_q=0,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
        )
    else:
        # BF16: keep padded tensors + random padding mask + unpad (match test_fmha_v3_fwd_ck.py)
        q = torch.randn(
            batch_size, seqlen_q, nheads, head_dim, device="cuda", dtype=dtype
        )
        k = torch.randn(
            batch_size, seqlen_k, nheads_k, head_dim, device="cuda", dtype=dtype
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads_k, head_dim, device="cuda", dtype=dtype
        )

        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, "cuda", mode="random"
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, "cuda", mode="random"
        )
        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

        out, time_us = profile_func(
            aiter.fmha_v3_varlen_fwd_ck_func,
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
        )

    return time_us


MODE_RUNNERS = {
    "batch_prefill": run_batch_prefill,
    "batch_prefill_blockscale": run_batch_prefill_blockscale,
    "fmha_regular": run_fmha_regular,
    "fmha_varlen": run_fmha_varlen,
}

# Modes that accept page_size / kv_lookup_table kwargs
PAGED_MODES = {"batch_prefill", "batch_prefill_blockscale"}

ALL_MODES = ["batch_prefill", "fmha_regular", "fmha_varlen"]


def print_header():
    print(f"{'Mode':<16} {'Dtype':<5} {'Problem':<45} {'Time(us)':>10} {'TFlops':>8}")
    print("-" * 88)


def print_row(mode, dtype_str, problem_str, time_us, tf):
    print(f"{mode:<16} {dtype_str:<5} {problem_str:<45} {time_us:>10.1f} {tf:>8.1f}")


def problem_str(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    causal,
    lsc,
    page_size=None,
    kv_lookup_table=None,
):
    c = "c" if causal else "nc"
    lsc_str = f" lsc={lsc}" if lsc > 0 else ""
    paged_str = ""
    if page_size is not None:
        paged_str = f" ps={page_size}"
        if kv_lookup_table is not None:
            paged_str += f" {kv_lookup_table}"
    return f"b={batch_size} h={nheads}/{nheads_k} sq={seqlen_q} sk={seqlen_k} {c}{lsc_str}{paged_str}"


def run_sweep(
    dtype_str, modes, quant="pertensor", page_size=1, kv_lookup_table="sglang"
):
    sweep_configs = [
        # (batch, nheads, nheads_k, seqlen_q, seqlen_k, causal, logits_soft_cap)
        (1, 6, 1, 1024, 1024, True, 0.0),
        (1, 6, 1, 2048, 2048, True, 0.0),
        (1, 6, 1, 4096, 4096, True, 0.0),
        (1, 6, 1, 8192, 8192, True, 0.0),
        (1, 6, 1, 16384, 16384, True, 0.0),
        (1, 6, 1, 32768, 32768, True, 0.0),
        (1, 6, 1, 65536, 65536, True, 0.0),
        (1, 6, 1, 131072, 131072, True, 0.0),
        (1, 16, 1, 65536, 65536, True, 0.0),
        (1, 40, 40, 37200, 37200, False, 0.0),
    ]

    # Remap batch_prefill mode based on quant method
    effective_modes = []
    for mode in modes:
        if mode == "batch_prefill" and quant == "kv_blockscale" and dtype_str == "fp8":
            effective_modes.append("batch_prefill_blockscale")
        else:
            effective_modes.append(mode)

    print_header()
    for b, n, nk, sq, sk, causal, lsc in sweep_configs:
        is_fp8 = dtype_str == "fp8"
        for mode in effective_modes:
            # Only pass paged KV args to paged modes
            paged_kwargs = {}
            ps_display = None
            lut_display = None
            if mode in PAGED_MODES:
                paged_kwargs = {
                    "page_size": page_size,
                    "kv_lookup_table": kv_lookup_table,
                }
                ps_display = page_size
                lut_display = kv_lookup_table
            ps_str = problem_str(b, n, nk, sq, sk, causal, lsc, ps_display, lut_display)
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            time_us = MODE_RUNNERS[mode](
                b, n, nk, sq, sk, 128, causal, lsc, is_fp8, **paged_kwargs
            )
            tf = tflops(flops(b, sq, sk, 128, n, nk, causal), time_us)
            # Display original mode name (batch_prefill) for consistency
            display_mode = (
                "batch_prefill" if mode == "batch_prefill_blockscale" else mode
            )
            print_row(display_mode, dtype_str, ps_str, time_us, tf)
        if len(effective_modes) > 1:
            print()


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Benchmark CK FMHA V3 kernels (batch_prefill / fmha_regular / fmha_varlen)",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["batch_prefill", "fmha_regular", "fmha_varlen", "all"],
    default="all",
    help="""Kernel mode to benchmark. Default is 'all'.
    batch_prefill  — paged KV (use --page_size, --lookup_table, --quant to configure)
    fmha_regular   — contiguous KV, batch/group mode
    fmha_varlen    — contiguous KV, variable-length mode
    all            — run all three and compare""",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=1,
    help="Batch size. Default is 1.",
)
parser.add_argument(
    "-n",
    "--nheads",
    type=int,
    default=8,
    help="Number of query heads. Default is 8.",
)
parser.add_argument(
    "-nk",
    "--nheads_k",
    type=int,
    default=-1,
    help="Number of KV heads. -1 means equal to nheads.",
)
parser.add_argument(
    "-q",
    "--seqlen_q",
    type=int,
    default=1024,
    help="Sequence length for query. Default is 1024.",
)
parser.add_argument(
    "-k",
    "--seqlen_k",
    type=int,
    default=-1,
    help="Sequence length for key. -1 means equal to seqlen_q.",
)
parser.add_argument(
    "-d",
    "--head_dim",
    type=int,
    default=128,
    help="Head dimension. Default is 128.",
)
parser.add_argument(
    "-c",
    "--causal",
    action="store_true",
    help="Enable causal attention.",
)
parser.add_argument(
    "--logits_soft_cap",
    type=float,
    default=0.0,
    help="Logits soft cap. Default is 0.0 (disabled).",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["bf16", "fp8"],
    default="bf16",
    help="Input dtype. Default is bf16.",
)
parser.add_argument(
    "--sweep",
    action="store_true",
    help="Run predefined sweep of problem sizes.",
)
parser.add_argument(
    "--quant",
    type=str,
    choices=["pertensor", "kv_blockscale"],
    default="pertensor",
    help="FP8 quantization method for batch_prefill. Default is pertensor.",
)
parser.add_argument(
    "--page_size",
    type=int,
    default=1,
    help="Page size for batch_prefill. Default is 1.",
)
parser.add_argument(
    "--lookup_table",
    type=str,
    choices=["sglang", "vllm"],
    default="sglang",
    help="KV lookup table layout for batch_prefill. Default is sglang.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    nheads_k = args.nheads_k if args.nheads_k > 0 else args.nheads
    seqlen_k = args.seqlen_k if args.seqlen_k > 0 else args.seqlen_q

    modes = ALL_MODES if args.mode == "all" else [args.mode]

    if args.sweep:
        run_sweep(
            args.dtype,
            modes,
            quant=args.quant,
            page_size=args.page_size,
            kv_lookup_table=args.lookup_table,
        )
    else:
        is_fp8 = args.dtype == "fp8"
        # Remap batch_prefill mode based on quant method
        effective_modes = []
        for mode in modes:
            if mode == "batch_prefill" and args.quant == "kv_blockscale" and is_fp8:
                effective_modes.append("batch_prefill_blockscale")
            else:
                effective_modes.append(mode)
        print_header()
        for mode in effective_modes:
            paged_kwargs = {}
            ps_display = None
            lut_display = None
            if mode in PAGED_MODES:
                paged_kwargs = {
                    "page_size": args.page_size,
                    "kv_lookup_table": args.lookup_table,
                }
                ps_display = args.page_size
                lut_display = args.lookup_table
            ps = problem_str(
                args.batch_size,
                args.nheads,
                nheads_k,
                args.seqlen_q,
                seqlen_k,
                args.causal,
                args.logits_soft_cap,
                ps_display,
                lut_display,
            )
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            time_us = MODE_RUNNERS[mode](
                args.batch_size,
                args.nheads,
                nheads_k,
                args.seqlen_q,
                seqlen_k,
                args.head_dim,
                args.causal,
                args.logits_soft_cap,
                is_fp8,
                **paged_kwargs,
            )
            tf = tflops(
                flops(
                    args.batch_size,
                    args.seqlen_q,
                    seqlen_k,
                    args.head_dim,
                    args.nheads,
                    nheads_k,
                    args.causal,
                ),
                time_us,
            )
            display_mode = (
                "batch_prefill" if mode == "batch_prefill_blockscale" else mode
            )
            print_row(display_mode, args.dtype, ps, time_us, tf)
