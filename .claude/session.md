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

## Golden Reference for Debugging

**IMPORTANT:** Do not compare v3 kernel register values with async_trload pipeline due to different kernel/pipeline designs. Always use Python reference implementations as golden:

| Test Case | Golden Reference | Notes |
|-----------|------------------|-------|
| Single KV tile iteration | `attention_fp8_ref()` | Batch-style softmax (full row at once) |
| Multiple KV tile iterations | `attention_fp8_ref_online()` | Online softmax (tile-by-tile) |

Both references are in `op_tests/test_mha_fp8.py`.

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

1. Verify `set_tile_if` is masking ALL padding positions (cols 5-63 for seqlen_k=5)
2. Add kernel debug prints to inspect sp_compute values after masking

---

## Investigation Progress (2026-02-03)

### Smallest Reproduction Case Found

**seqlen_k >= 5 fails, seqlen_k <= 4 passes**

| seqlen_q | seqlen_k | Result | Notes |
|----------|----------|--------|-------|
| 4 | 4 | PASS | l_ref/l_kernel ratio = 1.0 |
| 5 | 4 | PASS | Only seqlen_k matters |
| 4 | 5 | FAIL | diff=0.22, l inflated |
| 5 | 5 | FAIL | diff=0.17-0.22 |
| 4 | 8 | FAIL | diff=0.46 (worst) |
| 4 | 16 | borderline | diff=0.04 (below 0.055 threshold) |

Test command:
```bash
python op_tests/debug_mha_fp8.py -q 5 -k 5 -d 128 -dv 128 --detailed
```

### Key Observation: `l` (row sum) is Inflated

For seqlen_k=5, the kernel's `l` value is 14-22% larger than reference:
```
Row 0: l_ref=4.507, implied l_kernel≈5.27 (ratio=1.17)
Row 4: l_ref=4.159, implied l_kernel≈4.76 (ratio=1.14)
```

This causes output to be too small: `output = o_acc / l`

### v3 Pipeline Double Buffering Design

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

**Conclusion:** Masking order is CORRECT in both single-tile and multi-tile cases.

### Masking Logic Verified Correct

For seqlen_q=5, seqlen_k=5, causal=True, row 4:
- `IsEdgeTile()` returns true (64 > 5)
- `IsOutOfBound(4, 5)` returns true (5 >= min(4+1, 5) = 5)
- `should_mask` = true for col 5+
- `sp_compute` should be set to -inf for cols 5-63

For masked positions:
- sp_compute = -inf
- sp_delta = -inf * scale_s - scale_s * m = -inf
- exp2(-inf) = 0
- Contribution to rowsum = 0

**Theory says masking should work, but l is still inflated.**

### Remaining Questions

1. Is `set_tile_if` actually iterating over ALL positions (cols 5-63)?
2. Are there unmasked garbage values from K DRAM load beyond seqlen_k?
3. Is there a race condition or memory issue in the double buffer?

### Key Files for Further Investigation

- `block_fmha_fwd_v3_pipeline.hpp:1125-1157` - fmha_mask lambda
- `static_distributed_tensor.hpp` - set_tile_if implementation
- `block_masking.hpp:214-235` - IsOutOfBound implementation

---

## Debug observations

Mask decisions from debug print appear correct:
- Row 0, col 0: should_mask=0 (valid)
- Row 0, col 1: should_mask=1 (masked)
- Row 1, col 1: should_mask=0 (valid)
- Row 2, col 2: should_mask=0 (valid)

The causal masking formula is working correctly.

---

## Case 3 & 4: Isolating the masking bug

v3 uses `GenericAttentionMask` from `block_masking.hpp`.

### Case 3: Modify predicate in pipeline only

Modified `set_tile_if` predicate for IsMasking=true to use `col >= seqlen_k_end`:

| Test | Result |
|------|--------|
| kernel causal=True vs causal=False | **0.051** (NOT identical) |

### Case 4: Modify IsOutOfBound() in GenericAttentionMask

Modified `IsOutOfBound()` for IsMasking=true to use `return i_x >= x_total`:

| Test | Result |
|------|--------|
| kernel causal=True vs causal=False | **0.0** (IDENTICAL) |

### Observations from Case 3 & 4

- Case 3: Modifying pipeline predicate alone gives diff=0.051 between causal=True and causal=False
- Case 4: Modifying `IsOutOfBound()` to disable causal masking gives diff=0.0 (identical output)
- This confirms the difference is in the causal masking code path, not elsewhere

### IsMasking-Dependent Code Paths

| Location | Function | IsMasking=true | IsMasking=false |
|----------|----------|----------------|-----------------|
| `block_masking.hpp:113` | `GetTileRangeAlongX()` | Causal-aware range | `(0, x_total)` |
| `block_masking.hpp:267` | `IsEdgeTile()` | Causal edge check | Padding-only check |
| `block_masking.hpp:214` | `IsOutOfBound()` | Causal mask | Padding-only |
| `pipeline.hpp:1517` | Final normalization | Safe divide: `l==0 ? 0 : 1/l` | Direct: `1/l` |

**Conclusion:** The bug must be in one of these IsMasking-dependent paths. Case 4 shows that aligning `IsOutOfBound()` produces identical output, but we need to find what's **incorrect** about the original causal logic.

### v3 Mask Type (Verified from JIT-generated code)

v3 (`QRKSVS_ASYNC_TRLOAD_V3`) uses `FmhaMasks::CausalMask` = `GenericAttentionMask<true, false>`:

```cpp
// IsOutOfBound for IsMasking=true:
index_t x_end = min(i_y + x, x_total);
return i_x >= x_end || i_y >= y_total;
```

---

## TODO

- [x] Verify scale_p value at runtime → 448 (correct)
- [x] Verify mask formula matches reference → correct
- [x] Confirm `set_tile_if` works for IsMasking=false path
- [x] Modify IsMasking=true path to only check padding → still differs by 0.02-0.04
- [x] Case 3 & 4 comparison → `IsOutOfBound()` change produces identical output
- [ ] **Find the actual bug in causal masking logic**
- [ ] Print FULL intermediate values (32+ elements) when debugging

---

## Best Practices

1. **Print more values:** When comparing kernel registers with reference, print 32+ elements, not just first 4-8. Matching on few elements doesn't mean all match.

2. **Batch kernel changes:** Debug prints require expensive recompilation. Plan all changes at once rather than incrementally.

3. **Only record verified facts:** Do not write conclusions or hypotheses in this document until they are verified by experiments. Record experiment results and observations, not speculation.

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
- Masking: `3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/block_masking.hpp`
- Block GEMM: `3rdparty/composable_kernel/include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp`
- Test: `op_tests/test_mha_fp8.py`
- Debug Script: `op_tests/debug_mha_fp8.py`

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
