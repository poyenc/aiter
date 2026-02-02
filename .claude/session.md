# FP8 FMHA v3 Debug Session

**Last Updated:** 2026-02-02 (pytest verified: 48 failed, 128 passed)

## Current Focus

Investigating **Issue #2: Causal + Large Seqlen Bug** - tests fail when causal=True AND seqlen ≥ 256.

See [issues.md](issues.md) for full issue tracking and test results.

---

## Verified Correct (Ruled Out)

1. **scale_p = 448** (CORRECT)
   - Verified via debug print: `[SCALE_DEBUG] scale_p=448.0000`
   - Hardware path `type_convert<float>` correctly returns 448.0
   - The `(float)numeric<fp8_t>::max()` returning 126.0 only affects debug prints, not actual computation

2. **Causal mask formula** (CORRECT)
   - Kernel uses: `x_end = i_y + x` where `x = 1 + right_size + x_tmp`
   - For causal with `window_size[1] = 0`: valid if `col <= row + seqlen_k - seqlen_q`
   - Reference uses: `mask = col > row + seqlen_k - seqlen_q + window_size[1]` with `window_size[1] = 0`
   - Both formulas are equivalent - kernel mask is correct

3. **m (row max) and l (row sum)** verified via debug prints

---

## Current Test Result

### Large Seqlen Case
```bash
python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 64 -d 128 -dv 128 -c
```

**Output:**
```
Output max diff (kernel vs bf16 ref): 0.287109375
Output max diff (kernel vs online ref): 0.271484375
Output max diff (bf16 ref vs online ref): 0.03515625
```

### Small Seqlen Case (Smallest Reproduction)
```bash
python op_tests/debug_mha_fp8.py -b 1 -n 1 -q 32 -k 32 -d 128 -dv 128 -c
```

**Output:**
```
Output max diff (kernel vs bf16 ref): 0.27734375
Output max diff (kernel vs online ref): 0.28125
Output max diff (kernel vs batch ref): 0.28125
Output max diff (online ref vs batch ref): 0.0
```

**Key Finding:** The online and batch references produce identical output (diff=0.0), confirming the bug is in the kernel, not the reference implementations.

### Non-Causal Case (PASSES)
```bash
python op_tests/debug_mha_fp8.py -b 1 -n 1 -q 32 -k 32 -d 128 -dv 128
```

**Output:**
```
Output max diff (kernel vs bf16 ref): 0.03125  # < 0.055 threshold
Output max diff (kernel vs online ref): 0.02734375
Output max diff (kernel vs batch ref): 0.02734375
```

**Conclusion:** Bug is specific to causal masking. Non-causal case passes.

Error distribution by row (for seqlen_q=32, seqlen_k=32, causal=True):
- Row 0: diff=0.027 (attends to 1 col)
- Row 7: diff=0.074 (attends to 8 cols) - **highest error**
- Row 15: diff=0.006 (attends to 16 cols)
- Row 16: diff=0.020 (attends to 17 cols)
- Row 24: diff=0.057 (attends to 25 cols)
- Row 31: diff=0.010 (attends to 32 cols)

---

## Warp/Row Mapping (FP8 v3 Pipeline)

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

**Causal mask (bottom-right alignment):**
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

---

## Progress

### 2026-02-02

1. **Verified scale_p = 448** (CORRECT)
   - Debug print confirmed `scale_p=448.0000`
   - Ruled out FP8 max value interpretation issue

2. **Verified kernel mask is correct**
   - Lane 4: cols 0-3 have `should_mask=0`
   - Lane 36: col 4 has `should_mask=0`, cols 5+ have `should_mask=1`
   - Matches reference formula

3. **Removed all debug prints** from kernel code

4. **Confirmed test failure**
   - `test_mha_fp8.py -b 1 -n 1 -q 256 -k 64 -d 128 -dv 128 -c` fails
   - kernel vs reference diff: 0.27-0.29
   - References agree with each other (diff=0.035)

### 2025-01-31

1. **Fixed Issue #1 (K Tile Half-Stride Bug)**
   - Changed `GetQKBlockGemm()` to use non-SwizzleB warp GEMM
   - Small seqlen tests now pass (32, 108, 113, 128)

2. **Discovered Issue #2 (Causal + Large Seqlen Bug)**
   - 48 failed, 128 passed
   - All failures: causal=True + seqlen≥256

---

## GEMM Config Comparison: v3 vs async_trload

### GetQKBlockGemm (GEMM0: Q × K)

| Aspect | v3 Pipeline | async_trload Pipeline |
|--------|-------------|----------------------|
| WarpGemm | Explicit: `WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<>{}` | Dispatcher: `WarpGemmDispatcher<..., true>` |
| GemmLoopOrder | `MNK` | `MNK` |

Both use `GemmLoopOrder::MNK` - **no difference**.

### GetPVBlockGemm (GEMM1: P × V) - **KEY DIFFERENCES**

| Aspect | v3 Pipeline | async_trload Pipeline |
|--------|-------------|----------------------|
| WGAttrNumAccessEnum | Always `Double` | Conditional: `Double` only for (16×32) or (32×16), else `Single` |
| GemmLoopOrder | **`MNK`** | **`KMN`** |

**GemmLoopOrder difference is significant:**
- `KMN`: Outer loop K → M → N (loops over K first, accumulation-oriented)
- `MNK`: Outer loop M → N → K (loops over M, N first, row/col-oriented)

**Potential impact:** The loop order affects how partial products are accumulated. For P×V with causal masking, when P rows have all-zero masked elements, the order of accumulation may produce different intermediate states.

---

## Hypothesis

For Issue #2 (causal + large seqlen), remaining possibilities:

1. ~~GemmLoopOrder mismatch~~ (RULED OUT)
2. ~~P×V GEMM computation~~ (RULED OUT - works when mask disabled)
3. ~~Online softmax rescaling~~ (RULED OUT - works when mask disabled)
4. ~~FP8 scale_p mismatch~~ (RULED OUT - scale_p = 448 is correct)
5. ~~Causal mask off-by-one~~ (RULED OUT - kernel mask decisions are correct)

6. **`set_tile_if` with IsMasking=true** (HIGH - CURRENT FOCUS)

### Experiment Results (seqlen_q=32, seqlen_k=32)

| Case | set_tile_if Status | causal=True | causal=False |
|------|-------------------|-------------|--------------|
| 1 | Disabled for BOTH | 0.143 (fail) | 0.031 (pass) |
| 2 | Disabled for IsMasking=true only | 0.486 (fail) | 0.031 (pass) |
| Normal | Enabled for BOTH | 0.277 (fail) | 0.031 (pass) |

**Conclusions:**
- Case 1: Kernel outputs identical for both causal flags when no masking applied
- Case 2: IsMasking=false path works correctly with `set_tile_if` enabled
- **Bug is specific to IsMasking=true path**

---

## Next Steps

1. **Investigate why causal=True and causal=False produce different output even with same masking predicate**
   - `IsEdgeTile()` has different logic for IsMasking=true vs false
   - `l == 0 ? 0 : 1/l` vs `1/l` at final normalization (line 1517)
   - `CoreLoopScheduler` has different specializations (scheduling only, shouldn't affect correctness)

2. **Verify sp_compute tile distribution is correct for set_tile_if:**
   - Check if `get_x_indices_from_distributed_indices` returns correct (row, col)

---

## Case 3 Results: IsMasking=true with padding-only predicate

Modified IsMasking=true path to only check `col >= seqlen_k_end`:

| Row | kernel (causal=True) | nocausal | diff |
|-----|---------------------|----------|------|
| 0 | 0.4180 | 0.4395 | 0.023 |
| 8 | 0.4121 | 0.4395 | 0.035 |
| 31 | 0.4180 | 0.4434 | 0.031 |

**Finding:** Outputs still differ by 0.02-0.04 per row even with same masking predicate. Bug is NOT just in the predicate - there's other IsMasking-dependent code.

---

## TODO

- [x] Verify scale_p value at runtime → 448 (correct)
- [x] Verify mask formula matches reference → correct
- [x] Compare GEMM config between v3 and async_trload
- [x] Try changing v3 GEMM1 GemmLoopOrder from MNK to KMN → NOT THE BUG
- [x] Confirm `set_tile_if` works for IsMasking=false path
- [x] Modify IsMasking=true path to only check padding → still differs by 0.02-0.04
- [ ] **Investigate `IsEdgeTile()` differences between IsMasking=true/false**
- [ ] **Check final normalization `l == 0 ? 0 : 1/l` path**
- [ ] Print FULL intermediate values (32+ elements) when debugging

---

## Best Practices

1. **Print more values:** When comparing kernel registers with reference, print 32+ elements, not just first 4-8. Matching on few elements doesn't mean all match.

2. **Batch kernel changes:** Debug prints require expensive recompilation. Plan all changes at once rather than incrementally.

---

## Quick Reference

### Test Commands

> Replace `<CONTAINER>` and `<WORKSPACE>` with values from `.claude/user.md`

```bash
# Full pytest suite (source of truth for pass/fail)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python -m pytest op_tests/test_mha_fp8.py -v"

# Single failing test (Issue #2) - single KV tile
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 64 -d 128 -dv 128 -c"

# Single failing test (Issue #2) - multiple KV tiles
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 256 -d 128 -dv 128 -c"

# Passing test (same config, no causal)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 64 -d 128 -dv 128"
```

**IMPORTANT:** Always run `pytest op_tests/test_mha_fp8.py` to verify any fix before documenting conclusions.

### Key Files
- v3 Kernel: `3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp`
- v3 Pipeline: `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp`
- v3 Policy: `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp`
- async_trload Pipeline: `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload.hpp`
- async_trload Policy: `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp`
- Masking: `3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/block_masking.hpp`
- Block GEMM: `3rdparty/composable_kernel/include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp`
- Test: `op_tests/test_mha_fp8.py`

### Related Docs
- [issues.md](issues.md) - Issue status and reproduction
- [findings.md](findings.md) - Detailed investigation notes

---

## Notice

If encounter error 3 times and cannot make progress, ask user for help and let user decide what to do next.

---

## WARNING: Git Checkout

**ALWAYS ask user before running `git checkout` on files in the CK submodule.**

Reverting certain files can disable v3 kernel dispatch, causing tests to run v2 kernels instead and give **false-positive** results. Critical files that control v3 dispatch:
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` - Controls which kernels are generated
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp` - v3 kernel implementation

If you accidentally revert these, the tests will pass because they fall back to v2 kernels.

---

## WARNING: Debug Prints

**Debug prints significantly increase compilation time.** Before running tests, wrap all debug prints in `#if 0` blocks:

```cpp
#if 0  // Debug: description
    if(get_block_1d_id() == 0 && get_warp_id() == 0 && get_lane_id() == 0)
    {
        printf("[DEBUG] ...\n");
    }
#endif
```

Only enable debug prints (`#if 1`) when you specifically need to see the output. After debugging, disable them again before running the full test suite.

---

## WARNING: Do Not Modify test_mha_fp8.py

**Never modify `op_tests/test_mha_fp8.py` to avoid inaccurate testing results.** If you need to add code to check outputs or debug values, write a separate script instead (e.g., `op_tests/debug_mha_fp8.py`).
