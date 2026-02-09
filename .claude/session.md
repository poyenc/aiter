# FP8 FMHA v3 Debug Session

**Last Updated:** 2026-02-08

---

## Current Status

**FIXED** - All FP8 FMHA v3 issues resolved.

**Test Results:** Full pytest suite: **176/176 tests pass**

---

## Recent Work: FP8 Instruction Scheduling Optimization (2026-02-09)

Added fp8-specific `CoreLoopSchedulerImpl` with asymmetric `sched_group_barrier` patterns for both GEMM0 and GEMM1 phases. Also added `block_gemm_mfma_count_v` variable template.

### GEMM0 (Phase 0) — Fixed 7 back-to-back MFMAs

FP8 GEMM0 has 16 MFMAs (kKIter=2) but the same TRANS work (~32 v_exp_f32) as bf16/fp16 (8 MFMAs). The uniform `MFMA:1, TRANS:2, VALU:2` pattern caused the compiler to front-load all TRANS into MFMA #1, leaving MFMAs #2-8 back-to-back.

**Fix:** Two-phase pattern matching the natural K iteration boundary:
- K iter 0 (MFMAs 1-8): `MFMA:1, TRANS:4, VALU:4` — absorbs all softmax exp + add reduction
- K iter 1 (MFMAs 9-16): `MFMA:1, VALU:6` — absorbs P scale + cvt_pk_fp8 + o_acc rescale

**Result:** Zero back-to-back MFMAs, minimum 6 VALU interleaved.

### GEMM1 (Phase 2) — Reduced back-to-back MFMAs

fmha_alu0's v_fma chain depends on serial max3→permlane→max→mul chain, creating data dependency gap around MFMAs 8-11.

**Fix:** Asymmetric VALU constraints:
- First half (MFMAs 1-8): `MFMA:1, VALU:4` — v_perm + v_max3 + permlane chain
- Second half (MFMAs 9-16): `MFMA:1, VALU:3` — looser constraint for data-dep limited v_fma

### Explored but reverted

- `__builtin_fmaf` replacing `asm volatile fma_impl_vsv` for fp8 sp_delta — interleaved v_fma with MFMAs #8-12 but **profiling showed worse performance** (matrix core contention from v_pk_fma_f32 + VALU quota displacement of critical-path v_perm/v_max3)
- Unwrapping `asm volatile` from o_acc rescaling (`pk_mul_f32`) in Phase 2 — o_acc_scale data dependency (requires v_exp_f32 of row-max diff) prevents actual interleaving
- Removing `sched_barrier(0)` between GEMM1 and `fmha_alu_D_upd()` — merged scheduling regions but o_acc_scale dependency still blocks interleaving

**File:** `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp`

**Test Results:** 176/176 FP8 tests pass.

---

## Recent Work: Fix kMfmaPerWarpGemm Formula (2026-02-09)

Fixed `CoreLoopSchedulingParams` to correctly count hardware MFMAs instead of warp gemm calls.

### Bug
`kMfmaPerWarpGemm = MIterPerWarp * NIterPerWarp * KIterPerWarp` counts **warp gemm calls**, not hardware MFMA instructions. For fp8, each warp gemm (K=32) wraps 2× `v_mfma_f32_32x32x16_fp8_fp8` (base K=16), so `kKIter=2`. The formula gave 8 instead of 16. bf16/fp16 were unaffected (kKIter=1).

### Fix
Multiply by `WarpGemm::kK / WarpGemm::WarpGemmAttribute::Impl::kK` (= kKIter) to account for internal K iterations:

```cpp
static constexpr index_t kMfmaPerWarpGemm0 =
    QKBlockGemm::MIterPerWarp * QKBlockGemm::NIterPerWarp * QKBlockGemm::KIterPerWarp *
    (QKBlockGemm::WarpGemm::kK / QKBlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK);
```

This correctly gives 16 for fp8 (both GEMM0 and GEMM1) and is unchanged for bf16/fp16.

### Assembly impact
- **Phase 0 (QK GEMM):** MFMA barriers doubled 8→16. The extra barriers actually improved instruction interleaving — v_exp_f32 and softmax VALU instructions are now better distributed between MFMAs.
- **Phase 2 (PV GEMM):** MFMA barriers doubled 8→16 but no instruction ordering change. The back-to-back MFMAs 8-11 persist due to data dependency (v_fma_f32 chain depends on serial v_max3→permlane→v_mul chain).

No fp8-specific specialization override needed — the fix is in the generic formula.

**File:** `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp`

**Test Results:** 176/176 FP8 tests pass.

---

## Recent Work: Replace Custom s_waitcnt with CK Core API (2026-02-07)

Replaced custom `s_waitcnt` helpers in the V3 pipeline with CK core's architecture-aware `s_waitcnt` from `arch.hpp`.

### Changes

**File:** `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp`

1. **Deleted** 3 custom member functions (22 lines): `s_waitcnt<Vmcnt, Lgkmcnt, Expcnt>`, `s_waitcnt_vmcnt<Vmcnt>`, `s_waitcnt_lgkmcnt<Lgkmcnt>` — hardcoded GFX9 bit-packing with no validation.

2. **Replaced 12 call sites** with CK core's `ck_tile::s_waitcnt` (from `arch.hpp`), which has `static_assert` validation and GFX9/GFX11/GFX12 support via layout structs.

3. **Critical parameter order difference:** V3's custom API was `<Vmcnt, Lgkmcnt, Expcnt=7>`, CK core is `<vmcnt, expcnt, lgkmcnt>`. All translations handled correctly:
   - `s_waitcnt_lgkmcnt<0>()` → `s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>()`
   - `s_waitcnt_vmcnt<N>()` → `s_waitcnt<N>()`
   - `s_waitcnt<V, L>()` → `s_waitcnt<V, waitcnt_arg::kMaxExpCnt, L>()`

### Assembly verification (2026-02-07)
**Confirmed behavior-preserving.** All 6 V3 kernel variants (fp8 nmask/mask, bf16 nmask/mask, fp16 nmask/mask) produce **identical assembly** (only `__hip_cuid_*` hashes differ).

**Test Results:** 176/176 FP8 tests pass.

---

## Recent Work: CoreLoopScheduler Refactoring (2026-02-07)

Refactored `CoreLoopScheduler` for dtype-aware instruction scheduling.

### Changes

1. **`arch.hpp`**: Added `TRANS = 1 << 10` to `LLVMSchedGroupMask` enum, updated `ALL`. Fixed `0x200` (DS_WRITE) → `0x400` (TRANS) bug in V3 scheduler.

2. **`block_fmha_fwd_v3_pipeline.hpp`**: Replaced 150-line duplicated `CoreLoopScheduler<Problem, bool>` with:
   - `CoreLoopSchedulingParams<Problem>` — auto-derives `kMfmaPerWarpGemm0/1` from tile/gemm config
   - `CoreLoopSchedulerDefaultBase<Problem>` — reusable phase helpers (`schedule_gemm0_compute`, `schedule_gemm1_compute`, `schedule_load_phase`) using `LLVMSchedGroupMask` enum
   - `CoreLoopSchedulerImpl<Problem, Q, K, V>` — dtype-specialized dispatch (bf16/fp16/fp8 specializations)
   - `CoreLoopScheduler<Problem>` — user-facing forwarding template (simplified from 2 template params to 1)

3. **Usage site**: `CoreLoopScheduler<Problem, FmhaMask::IsMasking>` → `CoreLoopScheduler<Problem>`

### Key design decisions
- WG0/WG1 phase-shift pattern factored into single `schedule()` with `effective = (WG==0) ? Phase : (Phase+3)%4`
- Raw hex magic numbers replaced with `LLVMSchedGroupMask::MFMA` / `TRANS` / `VALU` / `SALU` enum
- Hardcoded `8` MFMA count replaced with auto-derived `Params::kMfmaPerWarpGemm0/1`
- bf16, fp16, and fp8 all currently share the same default base; fp8 specialization can be customized independently

### Assembly verification (2026-02-07)
**Confirmed behavior-preserving.** Compared V3 assembly (baseline: TRANS fix only, no refactoring) vs (TRANS fix + refactoring) for 6 kernel variants:
- fp8 nmask, fp8 mask, bf16 nmask, bf16 mask, fp16 nmask, fp16 mask

**Result:** All 6 variants produce **identical assembly** (only `__hip_cuid_*` compilation unit hashes differ — expected per-build randomness).

**Note on TRANS mask (`0x200` → `0x400`):** The `0x400` mask is for LLVM's transcendental unit (`v_exp_f32`, `v_log_f32`, etc.), not LDS transpose reads (`ds_read_b64_tr_b8`). LDS transpose reads are DS operations.

---

## Recent Work: FP8 Code Sinking Fix (2026-02-06)

Fixed Issue #5: FP8 P conversion code was being sunk by compiler to between Phase 1 and Phase 2.

**Fix:** Added `asm volatile` wrapper `detail::cvt_pk_fp8_f32()` for FP8 conversion, matching the pattern used by BF16/FP16.

**Key insight:** Only the final `v_cvt_pk_fp8_f32` instruction needs the `asm volatile` wrapper. All predecessor instructions (scale, etc.) automatically stay in Phase 0 because their results feed into the anchored conversion.

**File:** `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp`

**Assembly verification:** FP8 conversions now in Phase 0 (lines 889-969 inside phase0 marker).

---

## Previous Work: P/V Distribution Fix (2026-02-04)

**Root Cause:** P/V lane distribution mismatch in PV GEMM. Lane 32 had P[K=4] but V was all zeros.

**Fix:** Changed QK GEMM warp gemm from `WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<>{}` to `WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<>{}`. The SwizzleB variant provides 8 contiguous K positions per lane (vs 4), aligning P (sp_compute) distribution with V tile distribution.

See [knowledge.md](knowledge.md) for detailed assembly analysis.

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
- [x] Refactor CoreLoopScheduler for dtype-aware instruction scheduling (176/176 passed)
- [x] Replace custom s_waitcnt with CK core API (176/176 passed, assembly identical)
- [x] Fix kMfmaPerWarpGemm formula to count hardware MFMAs (176/176 passed)
- [x] Add fp8 CoreLoopSchedulerImpl with asymmetric scheduling (176/176 passed)

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
