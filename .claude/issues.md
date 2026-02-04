# FP8 FMHA v3 Issue Tracker

## Issue Summary

| ID | Issue | Status | Affects | Root Cause |
|----|-------|--------|---------|------------|
| #1 | [K Tile Half-Stride Bug](#issue-1-k-tile-half-stride-bug) | ✓ FIXED | All FP8, all seqlen | SwizzleB warp GEMM half-stride |
| #2 | [PV GEMM Missing K Positions](#issue-2-pv-gemm-missing-k-positions) | ✓ FIXED | seqlen_k % 16 in [5,11] | P/V lane distribution mismatch |
| #3 | [Causal Masking Bug](#issue-3-causal-masking-bug) | ✓ FIXED | seqlen_k≥256 with causal=True | Same as #2 |
| #4 | [V Tile Transpose Load Bug](#issue-4-v-tile-transpose-load-bug) | ⚠️ BYPASSED | FP8 V tile loading | Coordinate mismatch |

> **Note:** Replace `<CONTAINER>` and `<WORKSPACE>` with values from `.claude/user.md`

---

## Issue #1: K Tile Half-Stride Bug

### Status: ✓ FIXED (2026-01-30)

### Description
Lane N receives K row N/2 instead of row N during K tile loading, causing 8-lane offset in output.

### Root Cause
`WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution` has half-stride in `BWarpDstrEncoding`.

### Fix
Changed to `WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<>` in `GetQKBlockGemm()`.

**File:** `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp:267`

```cpp
#if 1  // Set to 0 to reproduce bug
    return WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<>{};
#else
    constexpr index_t swizzle_factor = 4;
    return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<swizzle_factor>{};
#endif
```

### Reproduction
```bash
# 1. Set #if 0 in the fix location above
# 2. Run test
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 1 -q 32 -k 32 -d 128 -dv 128"
```

### Test Results

| Config | With Bug | With Fix |
|--------|----------|----------|
| seqlen=32, causal=False | ❌ FAIL (0.98 diff) | ✓ PASS (0.035 diff) |
| seqlen=32, causal=True | ❌ FAIL (0.98 diff) | ✓ PASS (0.035 diff) |

---

## Issue #2: PV GEMM Missing K Positions

### Status: ✓ FIXED (2026-02-04)

### Description
Certain K positions were NOT contributing to PV GEMM output. The bug followed a predictable pattern based on seqlen_k modulo 16.

**Pattern:** Bug occurred when `(seqlen_k % 16)` was in range **[5, 11]**

### Root Cause

P/V lane distribution mismatch in PV GEMM:
- Non-SwizzleB warp gemm: kCM1PerLane = 4 contiguous K positions per lane
- V tile distribution: 8 contiguous K positions per lane (due to transpose)
- Lane 32 had P[K=4] but V was all zeros, causing `P[K=4] × V = 0`

### Fix

Changed QK GEMM warp gemm in `GetQKBlockGemm()`:
```cpp
// Before:
return WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<>{};

// After:
// Use SwizzleB variant to get 8 contiguous K positions per lane,
// matching the V tile distribution for PV GEMM
return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<>{};
```

**File:** `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp`

### Test Results (After Fix)

**Full pytest suite: 176/176 tests pass**

```bash
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python -m pytest op_tests/test_mha_fp8.py -v"
```

---

## Issue #3: Causal Masking Bug

### Status: ✓ FIXED (2026-02-04) - Same root cause as Issue #2

### Description
Tests failed when `seqlen_k ≥ 256` with `causal=True`, but passed with `causal=False`.

### Root Cause
Same as Issue #2 - P/V lane distribution mismatch. The SwizzleB fix resolves this issue as well.

### Test Results (After Fix)

All causal tests now pass. Full pytest suite: 176/176 tests pass.

---

## Issue #4: V Tile Transpose Load Bug

### Status: ⚠️ BYPASSED

### Description
`load_tile_transpose()` with `ReverseDirection=true` has coordinate space mismatch in `quad_output_ps_minor_offset` calculation.

### Root Cause
When `ReverseDirection=true`:
- `quad_output_ps_to_rhss_major0` uses original coordinate space
- `quad_idx_offset` uses reversed coordinate space
- Mismatch causes wrong LDS addresses

Current: `sequence<2, 3, 2>`, Expected: `sequence<3, 2, 3>`

### Bypass Applied
Validation disabled for FP8 in `load_tile_transpose.hpp`:
```cpp
static constexpr bool distr_encoding_valid =
    (sizeof(DataType_) == 1) || Validator::value;
```

### Reproduction
This issue was initially investigated as the root cause of the 8-lane offset bug, but the actual cause was Issue #1 (K tile half-stride). The V tile issue may still cause problems in certain configurations.

### Related
- May be related to Issue #2 (P/V distribution mismatch)

---

## Test Commands Reference

```bash
# Full pytest suite (source of truth)
python -m pytest op_tests/test_mha_fp8.py -v

# Clean JIT cache (required before each test)
rm -f aiter/jit/*.so

# Quick single test
python op_tests/test_mha_fp8.py -b 1 -n 1 -q 32 -k 32 -d 128 -dv 128

# With causal mask
python op_tests/test_mha_fp8.py -b 1 -n 1 -q 32 -k 32 -d 128 -dv 128 -c

# Specific test pattern
python -m pytest op_tests/test_mha_fp8.py -v -k "256" --tb=short
```

**IMPORTANT:** Always run `pytest op_tests/test_mha_fp8.py` to verify any fix before documenting conclusions.

---

## Key Files

| Component | File |
|-----------|------|
| v3 Kernel | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp` |
| v3 Pipeline | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp` |
| v3 Policy | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp` |
| async_trload Pipeline | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload.hpp` |
| async_trload Policy | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp` |
| Masking | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/block_masking.hpp` |
| Block GEMM | `3rdparty/composable_kernel/include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp` |
| Test | `op_tests/test_mha_fp8.py` |
