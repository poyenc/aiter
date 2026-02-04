# FP8 FMHA v3 Issue Tracker

## Issue Summary

| ID | Issue | Status | Affects | Root Cause |
|----|-------|--------|---------|------------|
| #1 | [K Tile Half-Stride Bug](#issue-1-k-tile-half-stride-bug) | ✓ FIXED | All FP8, all seqlen | SwizzleB warp GEMM half-stride |
| #2 | [PV GEMM Missing K Positions](#issue-2-pv-gemm-missing-k-positions) | 🔴 OPEN | seqlen_k % 16 in [5,11] | V[4-7,20-23,...] not accumulated |
| #3 | [Causal Masking Bug](#issue-3-causal-masking-bug) | 🔴 OPEN | seqlen_k≥256 with causal=True | Unknown |
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

### Status: 🔴 OPEN (Root cause identified, fix pending)

### Description
Certain K positions are NOT contributing to PV GEMM output. The bug follows a predictable pattern based on seqlen_k modulo 16.

**Pattern:** Bug occurs when `(seqlen_k % 16)` is in range **[5, 11]**

### Root Cause Analysis (2026-02-03)

**Verified Facts:**
1. **P (attention weights) are CORRECT** - Setting K[pos]=0 changes output, proving P is computed correctly
2. **V values are NOT accumulated for certain K positions** - Setting V[K=4]=0 produces diff=0.000000 (no change)
3. **Bug is in PV GEMM accumulation**, not in P computation or V loading

### Bug Pattern Diagram

```
Each 16-element block has two 8-element groups:
┌─────────────────────────────────────────────────────────────────┐
│  K positions: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15    │
│               └──group 0──┘  └──group 1──┘  └─────────────────  │
│                                                                  │
│  When seqlen_k lands in positions 5-11 of a 16-block:           │
│  - Group 0 (positions 4-7) of that block is NOT accumulated     │
│                                                                  │
│  Example: seqlen_k=5                                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ K pos:  0   1   2   3   4   5   6   7  ...                  ││
│  │        [OK][OK][OK][OK][--][pad][pad][pad]                  ││
│  │                        ↑                                     ││
│  │                   NOT accumulated!                           ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  The kernel processes V in 8-element groups, but when           │
│  seqlen_k % 16 is in [5,11], group 1 of the last 16-block       │
│  (positions 4-7 relative to block start) gets skipped.          │
└─────────────────────────────────────────────────────────────────┘
```

### Complete Test Results (seqlen_k 1-64)

```
seqlen_k= 1: OK
seqlen_k= 2: OK
seqlen_k= 3: OK
seqlen_k= 4: OK
seqlen_k= 5: BUG - missing: [4]
seqlen_k= 6: BUG - missing: [4, 5]
seqlen_k= 7: BUG - missing: [4, 5, 6]
seqlen_k= 8: BUG - missing: [4, 5, 6, 7]
seqlen_k= 9: BUG - missing: [5, 6, 7]       (note: [4] now OK)
seqlen_k=10: BUG - missing: [6, 7]
seqlen_k=11: BUG - missing: [7]
seqlen_k=12: OK
seqlen_k=13: OK
seqlen_k=14: OK
seqlen_k=15: OK
seqlen_k=16: OK
seqlen_k=17: OK
seqlen_k=18: OK
seqlen_k=19: OK
seqlen_k=20: OK
seqlen_k=21: BUG - missing: [20]
seqlen_k=22: BUG - missing: [20, 21]
seqlen_k=23: BUG - missing: [20, 21, 22]
seqlen_k=24: BUG - missing: [20, 21, 22, 23]
seqlen_k=25: BUG - missing: [21, 22, 23]
seqlen_k=26: BUG - missing: [22, 23]
seqlen_k=27: BUG - missing: [23]
seqlen_k=28: OK
...pattern repeats every 16...
```

**Affected ranges:** [5-11], [21-27], [37-43], [53-59], ...

### Verification Test

```bash
# Run the K position contribution test
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && python op_tests/debug_mha_fp8.py"

# Test all seqlen_k from 1 to 64
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && python op_tests/debug_mha_fp8.py --all"

# Verify P is correct (modify K instead of V)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && python op_tests/debug_mha_fp8.py --key"
```

### Ruled Out (Verified Correct)

1. **P (attention weights)** - K[pos]=0 changes output, proving P is correct
2. **scale_p = 448** - Verified via debug print
3. **Causal mask formula** - Both causal=True and causal=False exhibit same bug pattern
4. **m (row max) and l (row sum)** - Verified correct via debug prints

### Hypothesis for Root Cause

The v3 pipeline likely has an off-by-one error or incorrect loop bounds in the PV GEMM phase when handling partial tiles. Specifically:
- V is processed in 8-element groups within 16-element blocks
- When seqlen_k % 16 is in [5, 11], the second 8-element group (positions 4-7) gets skipped
- This suggests a loop iteration or tile slicing bug

### Full pytest Results (2026-01-31)

| Sequence Length | causal=False | causal=True |
|-----------------|--------------|-------------|
| 32 | ✓ PASS | ✓ PASS |
| 108 | ✓ PASS | ✓ PASS |
| 113 | ✓ PASS | ✓ PASS |
| 128 | ✓ PASS | ✓ PASS |
| 256 | ✓ PASS | ❌ FAIL (Issue #3) |
| 512 | ✓ PASS | ❌ FAIL (Issue #3) |
| 1023 | ✓ PASS | ❌ FAIL (Issue #3) |
| 2048 | ✓ PASS | ❌ FAIL (Issue #3) |
| 4096 | ✓ PASS | ❌ FAIL (Issue #3) |

**Note:** Large seqlen causal=True failures are a SEPARATE bug (Issue #3)

### Run All Tests
```bash
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python -m pytest op_tests/test_mha_fp8.py -v --tb=short"
```

---

## Issue #3: Causal Masking Bug

### Status: 🔴 OPEN (Root cause unknown)

### Description
Tests fail when `seqlen_k ≥ 256` with `causal=True`, but pass with `causal=False`. This is a SEPARATE bug from Issue #2.

### Symptoms
- All large seqlen tests pass with causal=False
- All large seqlen tests fail with causal=True
- Small seqlen tests (≤128) pass with both causal modes

### Test Results

| Sequence Length | causal=False | causal=True |
|-----------------|--------------|-------------|
| 128 | ✓ PASS | ✓ PASS |
| 256 | ✓ PASS | ❌ FAIL |
| 512 | ✓ PASS | ❌ FAIL |
| 1023 | ✓ PASS | ❌ FAIL |
| 2048 | ✓ PASS | ❌ FAIL |
| 4096 | ✓ PASS | ❌ FAIL |

### Reproduction
```bash
# Passing case (causal=False)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 256 -d 128 -dv 128"

# Failing case (causal=True)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 256 -d 128 -dv 128 -c"
```

### Notes
- Investigation deferred until Issue #2 is resolved
- May share root cause with Issue #2 or be completely independent

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
