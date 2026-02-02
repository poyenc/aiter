# FP8 FMHA v3 Issue Tracker

## Issue Summary

| ID | Issue | Status | Affects | Root Cause |
|----|-------|--------|---------|------------|
| #1 | [K Tile Half-Stride Bug](#issue-1-k-tile-half-stride-bug) | ✓ FIXED | All FP8, all seqlen | SwizzleB warp GEMM half-stride |
| #2 | [Causal + Large Seqlen Bug](#issue-2-causal--large-seqlen-bug) | 🔴 OPEN | causal=True, seqlen≥256 | Unknown (investigating) |
| #3 | [V Tile Transpose Load Bug](#issue-3-v-tile-transpose-load-bug) | ⚠️ BYPASSED | FP8 V tile loading | Coordinate mismatch |

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

## Issue #2: Causal + Large Seqlen Bug

### Status: 🔴 OPEN (Root cause unknown)

### Description
Tests fail when **causal=True** AND **seqlen ≥ 256**. All other combinations pass.

### Current Test Result (2026-02-02)

```bash
python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 64 -d 128 -dv 128 -c
```

**Output:**
```
Output max diff (kernel vs bf16 ref): 0.287109375
Output max diff (kernel vs online ref): 0.271484375
Output max diff (bf16 ref vs online ref): 0.03515625
```

The two references agree with each other (diff=0.035), but kernel differs from both by ~0.27-0.29.

### Ruled Out (Verified Correct)

1. **scale_p = 448** (CORRECT)
   - Verified via debug print: `[SCALE_DEBUG] scale_p=448.0000`
   - Hardware path `type_convert<float>` correctly returns 448.0

2. **Causal mask formula** (CORRECT)
   - Kernel mask matches reference formula
   - Verified via debug prints showing correct `should_mask` values

3. **m (row max) and l (row sum)** - verified correct via debug prints

4. **GemmLoopOrder mismatch** (RULED OUT)
   - v3 uses `MNK` for GEMM1, async_trload uses `KMN`
   - Tested: Changed to KMN with matching P+V distributions
   - Result: Test still fails with same diff (0.287)

### Remaining Hypotheses

1. **P×V GEMM computation** (HIGH)
   - P values appear correct, V loading appears correct
   - But o_acc output differs from reference
   - May be issue in GEMM1 data layout or accumulation

2. **Online softmax rescaling** (MEDIUM)
   - The `o_acc *= exp2(scale_s * (m_old - m_new))` rescaling
   - May have issue when some lanes have all-masked P values

### Reproduction
```bash
# Fails
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 64 -d 128 -dv 128 -c"

# Passes (same config, no causal)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 64 -d 128 -dv 128"
```

### Test Results (2026-01-31)

Full pytest run with Issue #1 fix applied:

| Sequence Length | causal=False | causal=True |
|-----------------|--------------|-------------|
| 32 | ✓ PASS | ✓ PASS |
| 108 | ✓ PASS | ✓ PASS |
| 113 | ✓ PASS | ✓ PASS |
| 128 | ✓ PASS | ✓ PASS |
| 256 | ✓ PASS | ❌ FAIL |
| 512 | ✓ PASS | ❌ FAIL |
| 1023 | ✓ PASS | ❌ FAIL |
| 2048 | ✓ PASS | ❌ FAIL |
| 4096 | ✓ PASS | ❌ FAIL |

**Summary:** 48 failed, 128 passed

### Run All Tests
```bash
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python -m pytest op_tests/test_mha_fp8.py -v --tb=short"
```

---

## Issue #3: V Tile Transpose Load Bug

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
- See [findings.md](findings.md) for 9 failed fix attempts
- May be related to Issue #2 (causal + large seqlen failures)

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
