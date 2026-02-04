# FP8 FMHA v3 Technical Knowledge

Technical knowledge learned during debugging the FP8 FMHA v3 pipeline.

---

## Warp/Row Mapping

**Configuration:** 8 warps, M0=256, 32x32 MFMA

| Warp | Q Rows | Output Rows |
|------|--------|-------------|
| 0 | [0, 32) | [0, 32) |
| 1 | [32, 64) | [32, 64) |
| 2 | [64, 96) | [64, 96) |
| 3 | [96, 128) | [96, 128) |
| 4 | [128, 160) | [128, 160) |
| 5 | [160, 192) | [160, 192) |
| 6 | [192, 224) | [192, 224) |
| 7 | [224, 256) | [224, 256) |

**Lane mapping within warp (64 threads/wavefront):** Both lane N and lane N+32 own row N within the 32-row chunk (for transposed C distribution with 32x32 MFMA).

**Column distribution for row 196 (warp 6):**
- Lane 4 handles: cols 0,1,2,3,8,9,10,11,16,17,...
- Lane 36 handles: cols 4,5,6,7,12,13,14,15,20,21,...

**KV tile iterations:** For seqlen_k=256 with kN0=64:
- Tile 0: K[0:64]
- Tile 1: K[64:128]
- Tile 2: K[128:192]
- Tile 3: K[192:256]

---

## Causal Mask Formula

**Bottom-right alignment:**
```python
# Mask condition from attention_fp8_ref() with causal=True:
# window_size becomes (-1, 0), so window_size[1] = 0
mask = col_idx > row_idx + seqlen_k - seqlen_q + 0
```

Example with seqlen_q=256, seqlen_k=64:
- Rows 0-191: full attention (all K[0:64] valid)
- Row 192: K[0:1] valid (1 position)
- Row 196: K[0:5] valid (5 positions: cols 0,1,2,3,4)
- Row 255: K[0:64] valid (all 64 positions)

**Verification examples from docstrings:**

seqlen_q=2, seqlen_k=5:
```
1 1 1 1 0   (row 0: col > 0+5-2=3, mask col 4)
1 1 1 1 1   (row 1: col > 1+5-2=4, nothing masked)
```

seqlen_q=5, seqlen_k=2:
```
0 0   (row 0: col > 0+2-5=-3, all cols > -3, all masked)
0 0   (row 1: col > 1+2-5=-2, all masked)
0 0   (row 2: col > 2+2-5=-1, all masked)
1 0   (row 3: col > 3+2-5=0, col 1 masked)
1 1   (row 4: col > 4+2-5=1, nothing masked)
```

**Kernel dispatch:** Uses `mask_info::decode("b:...")` which sets `mask_enum::mask_bottom_right`.

---

## sp_compute Buffer Roles

`sp_compute` is reused in-place for multiple stages:

1. **QK GEMM result**: After `gemm(sp_reg_idx, gemm0)`, stores raw Q×K scores
2. **Masked QK result**: After `fmha_mask()`, padding/causal positions set to -inf
3. **sp_delta computed**: After `fmha_alu0()`:
   - Computes `m = rowmax(sp_compute)`
   - Computes `sp_delta = scale_s * (sp_compute - m)` (stored in separate buffer)
   - sp_compute is NOT modified here
4. **Softmax result**: After `fmha_alu1()`, sp_compute = exp2(sp_delta)

This in-place reuse means debugging must check values at the correct stage.

---

## v3 Pipeline Double Buffering Design

The v3 pipeline uses **two register buffers** (`sp[0]` and `sp[1]`) to overlap computation:
- While one buffer does PV GEMM, the other does QK GEMM
- `pi` alternates between 0 and 1 to swap buffer roles

**Buffer Index Mapping:**
```cpp
auto xdl_SP_p01_reg_idx = number<1>{} - pi;  // pi=0 → 1, pi=1 → 0
auto xdl_SP_p23_reg_idx = pi;                 // pi=0 → 0, pi=1 → 1
```

**Core Loop Structure (`core_loop` calls `iteration(0)` then `iteration(1)`):**

| Phase | pi=0 | pi=1 |
|-------|------|------|
| phase0 | cl_calc(1, gemm0) - QK GEMM → buf1 | cl_calc(0, gemm0) - QK GEMM → buf0 |
| phase0 | fmha_alu1(0) - exp2/rowsum on buf0 | fmha_alu1(1) - exp2/rowsum on buf1 |
| phase0 | fmha_logits_trans(1) | fmha_logits_trans(0) |
| phase1 | fmha_mask(1) - mask buf1 | fmha_mask(0) - mask buf0 |
| phase2 | cl_calc(0, gemm1) - PV GEMM w/ buf0 | cl_calc(1, gemm1) - PV GEMM w/ buf1 |
| phase2 | fmha_alu0(1) - rowmax/delta on buf1 | fmha_alu0(0) - rowmax/delta on buf0 |

**Key: fmha_alu0 is called on OTHER buffer during PV GEMM:**
```cpp
auto cl_calc = [&](auto sp_reg_idx, auto gemm_idx) {
    if constexpr(gemm_idx == 1) {
        gemm_1(o_acc, sp(sp_reg_idx).p, v_tile);
        fmha_alu0(number<1>{} - sp_reg_idx);  // Process OTHER buffer!
    }
};
```

**Buffer Lifecycle (tracing buffer 1 across one full loop):**
1. pi=0, phase0: `cl_calc(1, gemm0)` → QK GEMM fills sp_compute[1]
2. pi=0, phase0: `fmha_logits_trans(1)` → transform sp_compute[1]
3. pi=0, phase1: `fmha_mask(1)` → mask sp_compute[1] to -inf
4. pi=0, phase2: `fmha_alu0(1)` → m = rowmax(sp_compute[1]), sp_delta[1]
5. pi=1, phase0: `fmha_alu1(1)` → sp_compute[1] = exp2(sp_delta[1]), rowsum
6. pi=1, phase2: `cl_calc(1, gemm1)` → PV GEMM using sp[1].p

**Single KV Tile Case (num_total_loop=1):**

Uses prologue + post_process instead of main loop:

*Prologue (buffer 0):*
1. `gemm(0, gemm0)` → QK GEMM fills sp_compute[0]
2. `fmha_logits_trans(0)` → optional transform
3. `fmha_mask(0)` → mask sp_compute[0] to -inf
4. `fmha_alu0(0)` → m = rowmax(sp_compute[0]), sp_delta[0]

*Post-process (buffer 0):*
5. `fmha_alu1(0)` → sp_compute[0] = exp2(sp_delta[0]), l = sum
6. `gemm(0, gemm1)` → PV GEMM

---

## V Tile Lane Mapping

**V tile distribution (verified for lanes 0-31):**

```
V^T tensor [d_v=128, seqlen_k] (padded to [128, 32]):

              K positions (seqlen_k, padded to 32)
              0   1   2   3   4   5...31 (zeros)
             ┌───┬───┬───┬───┬───┬─────────┐
  dim  0     │     Lane 0, Group 0        │
  dim  1     │     Lane 1, Group 0        │
  ...        │         ...                │
  dim 31     │     Lane 31, Group 0       │
             ├───────────────────────────-┤
  dim 32     │     Lane 0, Group 1        │
  ...        │         ...                │
  dim 63     │     Lane 31, Group 1       │
             ├───────────────────────────-┤
  dim 64     │     Lane 0, Group 2        │
  ...        │         ...                │
  dim 95     │     Lane 31, Group 2       │
             ├───────────────────────────-┤
  dim 96     │     Lane 0, Group 3        │
  ...        │         ...                │
  dim 127    │     Lane 31, Group 3       │
             └───────────────────────────-┘
```

**Lane N thread_buf_ layout (128 elements):**
- Positions 0-31: Group 0 → V^T dim N
- Positions 32-63: Group 1 → V^T dim N+32
- Positions 64-95: Group 2 → V^T dim N+64
- Positions 96-127: Group 3 → V^T dim N+96

**K position interleaving (within each group of 32):**
- Lane N positions [0:8,8:16,16:24,24:32] = K positions [0:8,16:24,32:40,48:56]
- Lane N+32 positions [0:8,8:16,16:24,24:32] = K positions [8:16,24:32,40:48,56:64]

---

## IsMasking-Dependent Code Paths

| Location | Function | IsMasking=true | IsMasking=false |
|----------|----------|----------------|-----------------|
| `block_masking.hpp:113` | `GetTileRangeAlongX()` | Causal-aware range | `(0, x_total)` |
| `block_masking.hpp:267` | `IsEdgeTile()` | Causal edge check | Padding-only check |
| `block_masking.hpp:214` | `IsOutOfBound()` | Causal mask | Padding-only |
| `pipeline.hpp:1517` | Final normalization | Safe divide: `l==0 ? 0 : 1/l` | Direct: `1/l` |

v3 uses `GenericAttentionMask<true, false>` (IsMasking=true, IsLocal=false):
```cpp
// IsOutOfBound for IsMasking=true:
index_t x_end = min(i_y + x, x_total);
return i_x >= x_end || i_y >= y_total;
```

---

## Golden Reference for Debugging

**IMPORTANT:** Do not compare v3 kernel register values with async_trload pipeline due to different kernel/pipeline designs. Always use Python reference implementations as golden:

| Test Case | Golden Reference | Notes |
|-----------|------------------|-------|
| Single KV tile iteration | `attention_fp8_ref()` | Batch-style softmax (full row at once) |
| Multiple KV tile iterations | `attention_fp8_ref_online()` | Online softmax (tile-by-tile) |

Both references are in `op_tests/test_mha_fp8.py`.

---

## Reference Implementation Verification

**Verification script:** `op_tests/verify_reference.py`

Compares three implementations:
1. `attention_fp8_ref_online()` - online softmax with KV tiling (kv_tile_size=64)
2. `attention_fp8_ref()` - batch softmax (full sequence at once)
3. `aiter.flash_attn_func` - BF16 production kernel

**Key findings:**

| Configuration | Online vs Batch Diff | Notes |
|---------------|---------------------|-------|
| Single KV tile (k ≤ 64) | 0.000 | Identical results |
| Multiple KV tiles (k > 64) | 0.007-0.012 | Expected difference |

**Why the difference for multiple tiles:**
- Online softmax quantizes P to FP8 **per-tile** using a running max
- Batch softmax quantizes P using a **single global max**
- Different FP8 representations for logically equivalent attention weights

**Tolerance:** Tests use 0.02 tolerance to accommodate:
- FP8 quantization differences between online/batch softmax
- Numerical variance in online softmax accumulation

---

## P and V Distribution Mismatch (Root Cause - FIXED)

For PV GEMM to compute correctly, when lane N has P[K=k], the same lane should have V[K=k].

**Observed mismatch (seqlen_q=1, seqlen_k=5):**

| Lane | P values (before fix) | V values | Problem |
|------|----------------------|----------|---------|
| 0 | P[K=0,1,2,3] | V[K=0,1,2,3,4] | V[K=4] exists but P[K=4] missing |
| 32 | P[K=4]=124 | ALL ZEROS | P[K=4] exists but V[K=4] missing |

Result: `P[K=4] × V[K=4] = 124 × 0 = 0`

**Root Cause:**
- Non-SwizzleB warp gemm: kCM1PerLane = 4 contiguous K positions per lane
- V tile distribution: 8 contiguous K positions per lane (due to transpose)
- Mismatch causes P[K=4] (lane 32) to multiply with V[K=?] (zeros in lane 32)

**Fix (2026-02-04):**

Changed QK GEMM warp gemm in `GetQKBlockGemm()`:
```cpp
// Before:
return WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<>{};

// After:
return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<>{};
```

**Why SwizzleB fixes it:**
- SwizzleB (SFactor=2): kCM1PerLane * SFactor = 8 contiguous K positions
- This matches V tile distribution (8 K positions per lane)
- Lane 0 now has P[K=0-7], matching V[K=0-7]

**Test Results:** 176/176 tests pass

---

## Key Files

| Component | File |
|-----------|------|
| v3 Pipeline | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp` |
| v3 Policy | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp` |
| async_trload Policy | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp` |
| Masking | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/block_masking.hpp` |
| Block GEMM | `3rdparty/composable_kernel/include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp` |
| Test | `op_tests/test_mha_fp8.py` |
| Debug Script | `op_tests/debug_mha_fp8.py` |
