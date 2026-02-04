# FP8 FMHA v3 Debug Session

**Last Updated:** 2026-02-04

---

## Current Status

**FIXED** - FP8 FMHA v3 pipeline now produces correct outputs.

**Root Cause:** P/V lane distribution mismatch in PV GEMM. Lane 32 had P[K=4] but V was all zeros.

**Fix:** Changed QK GEMM warp gemm from `WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<>{}` to `WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<>{}`. The SwizzleB variant provides 8 contiguous K positions per lane (vs 4), aligning P (sp_compute) distribution with V tile distribution.

**Test Results:**
- Minimal reproducer (q=1, k=5): Output max diff = 0.0
- Full pytest suite: **176/176 tests pass**

---

## Fix Details

**File:** `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp`

**Change in GetQKBlockGemm():**
```cpp
// Before:
return WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<>{};

// After:
// Use SwizzleB variant to get 8 contiguous K positions per lane,
// matching the V tile distribution for PV GEMM
return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<>{};
```

**Why this works:**
- Non-SwizzleB: kCM1PerLane = 4 → Lane 0 owns K[0,1,2,3], Lane 32 owns K[4,5,6,7]
- SwizzleB (SFactor=2): kCM1PerLane * SFactor = 8 → Lane 0 owns K[0,1,2,3,4,5,6,7]
- V tile distribution puts all 5 valid K positions in Lane 0
- With SwizzleB, P[K=0-7] is also in Lane 0, so P × V works correctly

---

## TODO

- [x] Verify scale_p → 448 (correct)
- [x] Verify mask formula → correct
- [x] Verify V tile lane mapping → correct
- [x] Identify root cause → P/V distribution mismatch
- [x] Verify attention_fp8_ref_online() is correct (verified 2026-02-04)
- [x] Fix distribution mismatch in v3 policy (SwizzleB variant)
- [x] Run full pytest suite to verify fix (176/176 passed)
- [ ] Commit fix with documentation

---

## Test Commands

```bash
# Full pytest suite (source of truth)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python -m pytest op_tests/test_mha_fp8.py -v"

# Single test (minimal reproducer)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 1 -q 1 -k 5 -d 128 -dv 128"
```

> Replace `<CONTAINER>` and `<WORKSPACE>` with values from `.claude/user.md`

---

## Related Docs

- [issues.md](issues.md) - Issue tracking and reproduction steps
- [knowledge.md](knowledge.md) - Technical knowledge (pipeline design, lane mapping, etc.)

---

## Warnings

### Git Checkout
**ALWAYS ask user before `git checkout` on CK submodule.** Reverting can disable v3 dispatch → false positives.

### Debug Prints
Wrap in `#if 0` blocks. Only enable when needed. Slow compilation otherwise.

### Print One Lane Per Run
Thread buffers have 128+ elements. Print one lane at a time.

### Do Not Modify test_mha_fp8.py
Use `op_tests/debug_mha_fp8.py` for debug code.

---

## Notice

If encounter error 3 times without progress, ask user for help.
