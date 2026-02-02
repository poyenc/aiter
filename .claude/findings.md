# FP8 FMHA v3 Investigation Findings

This document contains detailed investigation notes, experiments, and findings for the FP8 FMHA v3 pipeline issues.

For current status and open issues, see [session.md](session.md).

---

## Issue #2 Investigation (2026-02-02)

### Ruled Out Hypotheses

#### 1. FP8 scale_p Mismatch - RULED OUT

**Initial Hypothesis:** `scale_p` was incorrectly computed as 126 instead of 448.

**Investigation:**
```
[SCALE_DEBUG] scale_p=448.0000 (CORRECT!)
[SCALE_DEBUG] hw_builtin(0x7e)=448.0
```

**Conclusion:** scale_p = 448 is correct. The `(float)numeric<fp8_t>::max()` returning 126.0 only affects debug print formatting, not the actual computation path which uses `type_convert<float>()`.

#### 2. Causal Mask Off-by-One - RULED OUT

**Initial Hypothesis:** The mask formula in block_masking.hpp had an off-by-one error.

**Investigation:**
- Kernel mask: `x = 1 + right_size + x_tmp`, `x_end = min(i_y + x, x_total)`, masked if `col >= x_end`
- Reference mask: `mask = col > row + seqlen_k - seqlen_q + window_size[1]`
- For causal with `window_size[1] = 0`: both formulas give same result

**Verified via debug prints:**
```
[MASK LANE4] tile_idx=(196,0) row=196 col=0 should_mask=0
[MASK LANE4] tile_idx=(196,1) row=196 col=1 should_mask=0
[MASK LANE4] tile_idx=(196,2) row=196 col=2 should_mask=0
[MASK LANE4] tile_idx=(196,3) row=196 col=3 should_mask=0
[MASK LANE36] tile_idx=(196,4) row=196 col=4 should_mask=0
[MASK LANE36] tile_idx=(196,5) row=196 col=5 should_mask=1
```

Row 196 with seqlen_q=256, seqlen_k=64: valid cols = {0,1,2,3,4} (5 positions) - matches reference.

**Conclusion:** Kernel mask is correct. No off-by-one bug.

#### 3. m (row max) and l (row sum) - VERIFIED CORRECT

Previously verified via debug prints that m and l values match reference.

### Current Test Failure

```bash
python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 64 -d 128 -dv 128 -c
```

**Output:**
```
Output max diff (kernel vs bf16 ref): 0.287109375
Output max diff (kernel vs online ref): 0.271484375
Output max diff (bf16 ref vs online ref): 0.03515625
```

The two references agree (diff=0.035), but kernel differs by ~0.27-0.29.

### GEMM Config Comparison: v3 vs async_trload (2026-02-02)

Compared `GetQKBlockGemm` and `GetPVBlockGemm` between:
- v3: `block_fmha_fwd_v3_pipeline_default_policy.hpp`
- async_trload: `block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp`

**GetQKBlockGemm (GEMM0: Q × K):**
| Aspect | v3 Pipeline | async_trload Pipeline |
|--------|-------------|----------------------|
| WarpGemm | Explicit: `WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<>{}` | Dispatcher: `WarpGemmDispatcher<..., true>` |
| GemmLoopOrder | `MNK` | `MNK` |

Both use `MNK` - no difference for GEMM0.

**GetPVBlockGemm (GEMM1: P × V):**
| Aspect | v3 Pipeline | async_trload Pipeline |
|--------|-------------|----------------------|
| WGAttrNumAccessEnum | Always `Double` | Conditional: `Double` only for (16×32) or (32×16), else `Single` |
| GemmLoopOrder | **`MNK`** | **`KMN`** |

**Key finding:** v3 uses `GemmLoopOrder::MNK` while async_trload uses `GemmLoopOrder::KMN` for GEMM1.

**GemmLoopOrder semantics (from block_gemm_areg_breg_creg_v2.hpp):**
- `KMN`: Outer loop order K → M → N (loops over K first, accumulation-oriented)
- `MNK`: Outer loop order M → N → K (loops over M, N first, row/col-oriented)

**Potential impact:** Different loop order affects how partial products accumulate. For P×V with causal masking where some P rows are partially or fully zeroed, the accumulation order may produce different intermediate rounding behavior.

**Next step:** Try changing v3 GEMM1 to use `GemmLoopOrder::KMN` and test if this fixes the causal + large seqlen bug.

### Remaining Investigation Areas

1. **GemmLoopOrder mismatch** (NEW) - v3 uses `MNK`, async_trload uses `KMN` for GEMM1
2. **P×V GEMM (gemm_1)** - P values appear correct, but o_acc differs from reference
3. **Online softmax rescaling** - `o_acc *= exp2(scale_s * (m_old - m_new))` when some lanes are all-masked
4. **Final O normalization** - `O = o_acc / l * scale_o`

---

# Historical Investigation Notes

## Original Problem Description
FP8 v3 kernel has an 8-lane offset bug in `load_tile_transpose()` causing V data to be read into wrong lanes.

**Update (2025-01-31):** The actual root cause was the K tile half-stride bug, not V tile transpose load. The V tile investigation was a red herring, but documented below for reference.

## Investigation Progress (2024-01-30)

### Ruled Out (verified via static_assert)
| Property | test_block_gemm (WORKS) | FMHA v3 (BUG) |
|----------|-------------------------|---------------|
| V window dim0 | 64 | 64 |
| V window dim1 | 128 | 128 |
| TransposedDstrEncode | identical | identical |
| BlockGemm type | identical | identical |

### Narrowed Down: LDS Layout Difference
The key difference between test_block_gemm (works) and FMHA v3 (bug):

| Aspect | test_block_gemm (WORKS) | FMHA v3 (BUG) |
|--------|-------------------------|---------------|
| LDS Store | `b_lds[idx] = b_global[idx]` | `async_load_tile_raw()` swizzled |
| LDS View | `make_naive_tensor_view()` row-major | `MakeVLdsLoadBlockDescriptor()` 5D→2D |
| Strides | `(128, 1)` row-major | Swizzled with dimension reordering |

### Hypothesis
The complex swizzled LDS layout from `async_load_tile_raw()` may not be compatible with how `load_tile_transpose()` expects data to be laid out.

### Next Step: Test with async_load_tile()
Replace `async_load_tile_raw()` with `async_load_tile()` to test if simpler LDS layout fixes the bug:
1. `async_load_tile()` uses tile distribution for LDS offset calculation
2. Allows using simple row-major LDS descriptors like test_block_gemm
3. Eliminates complex swizzled store/load descriptor pairing

**Files to modify:**
- `block_fmha_fwd_v3_pipeline_default_policy.hpp`: Simplify LDS descriptors
- `block_fmha_fwd_v3_pipeline.hpp`: Replace `async_load_tile_raw()` with `async_load_tile()`

---

## Debug Test Pattern for test_mha_fp8.py

**Symptom:**
- V[16] data appears in kernel output row 8
- V[8] data appears in kernel output row 16
- Values are correct, just positioned wrong (8-lane offset pattern)

**Test configuration:** Only V[16] has data, all other rows are zeros

To recreate the debug test that detects the 8-lane offset bug, modify `test_flash_attn_output()` in `op_tests/test_mha_fp8.py`:

### Data Initialization Pattern

```python
# Debug pattern: identity-like Q/K, only V[16] has data
target_row = 16

# Q: identity-like pattern - Q[i, i % d] = 10.0
q = torch.zeros(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
for i in range(seqlen_q):
    q[:, i, :, i % d] = 10.0

# K: identity-like pattern - K[j, j % d] = 10.0, rest = -10.0
k = torch.full(
    size=(batch_size, seqlen_k, nheads_k, d),
    fill_value=-10.0,
    device="cuda",
    dtype=dtype,
)
for j in range(seqlen_k):
    k[:, j, :, j % d] = 10.0

# V: only target_row (16) has data, all other rows are zeros
v = torch.rand(batch_size, seqlen_k, nheads_k, d_v, device="cuda", dtype=dtype)
v[:, :target_row, :, :] = 0
v[:, target_row+1:, :, :] = 0
```

### V[8]/V[16] Check (add after run_ck calls)

```python
# Debug: Check V[8] vs V[16] lane offset issue
out_2d = out.squeeze(0).squeeze(1)  # [seqlen_q, d_v]
out_ref_2d = out_ref.squeeze(0).squeeze(1)

if seqlen_q > 16 and seqlen_k > 16:
    row8_sum = out_2d[8].abs().sum().item()
    row16_sum = out_2d[16].abs().sum().item()
    row8_ref_sum = out_ref_2d[8].abs().sum().item()
    row16_ref_sum = out_ref_2d[16].abs().sum().item()
    print(f"[POYENC] V[8]/V[16] check (target_row={target_row}):")
    print(f"  GPU output row 8 sum: {row8_sum:.6f}, row 16 sum: {row16_sum:.6f}")
    print(f"  Reference row 8 sum: {row8_ref_sum:.6f}, row 16 sum: {row16_ref_sum:.6f}")
    if row8_sum > 0.1 and row16_ref_sum > 0.1 and row16_sum < 0.1:
        print(f"  WARNING: 8-lane offset bug detected! V[16] data appeared in row 8")

    # Print first few values of rows 8 and 16
    print(f"  GPU row 8 first 8 vals: {out_2d[8, :8].tolist()}")
    print(f"  GPU row 16 first 8 vals: {out_2d[16, :8].tolist()}")
    print(f"  Ref row 8 first 8 vals: {out_ref_2d[8, :8].tolist()}")
    print(f"  Ref row 16 first 8 vals: {out_ref_2d[16, :8].tolist()}")
```

### Expected Bug Output

When the 8-lane offset bug is present:
```
[POYENC] V[8]/V[16] check (target_row=16):
  GPU output row 8 sum: 64.500000, row 16 sum: 0.000000
  Reference row 8 sum: 0.000000, row 16 sum: 65.000000
  WARNING: 8-lane offset bug detected! V[16] data appeared in row 8
  GPU row 8 first 8 vals: [0.314453125, 0.2275390625, ...]  <- WRONG: has V[16] data
  GPU row 16 first 8 vals: [0.0, 0.0, 0.0, ...]             <- WRONG: should have data
  Ref row 8 first 8 vals: [0.0, 0.0, 0.0, ...]              <- Correct
  Ref row 16 first 8 vals: [0.30078125, 0.2275390625, ...]  <- Correct
Output max diff: 0.98046875
```

### Why This Pattern Works

1. **Identity-like Q/K**: Forces attention to focus on diagonal elements
   - Q[i] has high value at position `i % d`
   - K[j] has high value at position `j % d`, low elsewhere
   - Result: Q[i] · K[j] is maximized when `i % d == j % d`

2. **Single V row with data**: Makes offset bugs visible
   - Only V[16] has non-zero values
   - If output row 16 should have V[16] data (based on attention pattern)
   - But row 8 has the data instead → 8-lane offset bug confirmed

3. **Causal mask**: Required to trigger the v3 kernel path with the bug

---

## Bug Confirmed

Debug prints showed:
- Lane 8: coord=(4,0) instead of expected (8,?)
- Lane 16: coord=(0,16) instead of expected (16,?)
- LDS descriptor is correct: row 8 → offset 128, row 16 → offset 256
- **Bug location:** Tile distribution produces wrong `bottom_tensor_thread_coord`

---

## Investigation: Step 1 - Compile-Time Type Inspection

**Method:** Added `ShowType<T>` template to trigger compile errors showing actual constexpr values

**File:** `3rdparty/composable_kernel/include/ck_tile/core/tensor/load_tile_transpose.hpp:298-303`

**Extracted values (for FP8 v3 with ReverseDirection=true):**

```cpp
outer_hs_lengthss = tuple<sequence<4, 1>, sequence<2, 2, 2>>
reversed_outer_hs_lengthss = tuple<sequence<2, 2, 2>, sequence<4, 1>>  // ✓ Correctly swapped

quad_idx_offset = tuple<constant<3>, constant<2>>
  // quad_idx_offset[0] = 3 (from reversed dim0 size)
  // quad_idx_offset[1] = 2 (from reversed dim1 size)

quad_output_ps_to_rhss_major0 = sequence<2, 1, 2>  // Dimension indices

quad_output_ps_minor_offset = sequence<2, 3, 2>  // Current (WRONG)
  // Calculation:
  //   x=2: quad_idx_offset[2-1] = quad_idx_offset[1] = 2
  //   x=1: quad_idx_offset[1-1] = quad_idx_offset[0] = 3
  //   x=2: quad_idx_offset[2-1] = quad_idx_offset[1] = 2
```

**Expected:** `sequence<3, 2, 3>` (values swapped)

---

## Investigation: Step 2 - Manual Trace

**Root Cause Identified:**

When `ReverseDirection=true`:
1. `QuadInputEncoding` and `QuadOutputEncoding` are **SWAPPED** (line 215-222):
   ```cpp
   using QuadOutputEncoding = std::conditional_t<
       ReverseDirection,
       typename Policy::template QuadInputEncoding<LaneGroupSize>,  // ← Uses INPUT when reverse
       typename Policy::template QuadOutputEncoding<LaneGroupSize>>;
   ```

2. `quad_output_ps_to_rhss_major0` comes from `QuadOutputEncoding::ps_to_rhss_major_[0]`
   - When `ReverseDirection=true`, this is actually `QuadInputEncoding`
   - Contains dimension indices in **ORIGINAL coordinate space** (before transpose)

3. `quad_idx_offset` is computed from `reversed_outer_hs_lengthss`
   - Contains sizes in **REVERSED coordinate space** (after swap)

4. **Mismatch:** Original-space indices used to lookup reversed-space sizes
   - Index 1 (original dim0) → looks up quad_idx_offset[0] = 3 (reversed dim0, was original dim1) ✗
   - Index 2 (original dim1) → looks up quad_idx_offset[1] = 2 (reversed dim1, was original dim0) ✗

**Correct mapping should be:**
- Index 1 (original dim0) → needs reversed dim1 size = 2
- Index 2 (original dim1) → needs reversed dim0 size = 3

---

## Fix Attempts (All Failed)

All attempts to modify `quad_output_ps_minor_offset` calculation failed with:
```
candidate template ignored: requirement 'TransposeTileDistrChecker<...>::distr_encoding_valid' was not satisfied
```

### Attempt 1: if constexpr with index swap
```cpp
static constexpr auto quad_output_ps_minor_offset = to_sequence(generate_tuple_for(
    [](auto x) {
        if constexpr(ReverseDirection)
        {
            constexpr auto swapped_x = (x == 1) ? 2 : (x == 2) ? 1 : x;
            return quad_idx_offset[number<swapped_x - 1>{}];
        }
        else
        {
            return quad_idx_offset[number<x - 1>{}];
        }
    },
    quad_output_ps_to_rhss_major0));
```
**Result:** ❌ Validation failed

### Attempt 2: Use outer_hs_lengthss instead of reversed
```cpp
static constexpr auto quad_idx_offset =
    transform_tuples([](auto x) { return number<x.size()>{}; }, outer_hs_lengthss);
```
**Result:** ❌ Constexpr initialization errors in `tile_distribution_encoding.hpp`

### Attempt 3: tuple_reverse(quad_idx_offset)
```cpp
static constexpr auto reversed_quad_idx_offset = tuple_reverse(quad_idx_offset);
static constexpr auto quad_output_ps_minor_offset = to_sequence(generate_tuple_for(
    [](auto x) { return reversed_quad_idx_offset[number<x - 1>{}]; },
    quad_output_ps_to_rhss_major0));
```
**Result:** ❌ Validation failed

### Attempt 4-8: Various other approaches
- Lambda with external ndim variable → Validation failed
- Conditional quad_idx_offset source selection → Constexpr errors
- Transform input with swap_one_and_two → Validation failed
- All variations → Same validation failure

### Attempt 9: Transform quad_output_ps_to_rhss_major0
```cpp
static constexpr auto adjusted_quad_output_ps_to_rhss_major0 =
    std::conditional_t<ReverseDirection,
                      decltype(quad_output_ps_to_rhss_major0.transform(swap_one_and_two)),
                      decltype(quad_output_ps_to_rhss_major0)>{};

static constexpr auto quad_output_ps_minor_offset = to_sequence(generate_tuple_for(
    [](auto x) { return quad_idx_offset[number<x - 1>{}]; },
    adjusted_quad_output_ps_to_rhss_major0));
```
**Result:** ❌ Validation failed - produces same invalid encoding `sequence<2, 3, 2, 4>`

---

## Validation Logic Analysis

**File:** `3rdparty/composable_kernel/include/ck_tile/core/tensor/load_tile_transpose.hpp:51-189`

### Validation Checks (ValidationTraitsImpl:116-174)

For a tile distribution to be valid, it must pass ALL of:

1. **2D tensor check:** `InDstrEncode::NDimX == 2`

2. **Suffix check:** Quad hs_lengthss must be suffix of input hs_lengthss (both dimensions)
   ```cpp
   util::is_sequence_suffix_v<quad_hs[0], input_hs[0]>
   util::is_sequence_suffix_v<quad_hs[1], input_hs[1]>
   ```

3. **PS→RHS mapping check:** Quad PS mapping must be suffix of input PS mapping (with offset adjustment)
   ```cpp
   // Check quad_ps_major0 is suffix of input_ps_major_last
   // Check shifted_quad_ps_minor0 is suffix of input_ps_minor_last
   //   where shifted = quad_ps_minor0[i] + psys_offset[quad_ps_major0[i] - 1]
   ```

4. **YS→RHS mapping check:** Must match expected pattern
   ```cpp
   input_ys_major.back() == 2
   input_ys_minor.back() == input_hs[1].size() - 1
   ```

### Quad Encodings (Quad8 for FP8, sizeof(DataType)==1)

```cpp
InputEncoding:
  hs_lengthss: tuple<sequence<8>, sequence<LaneGroupSize/16, 2, 8>>
  ps_to_rhss_major: tuple<sequence<2, 1, 2>>
  ps_to_rhss_minor: tuple<sequence<0, 0, 1>>

OutputEncoding:
  hs_lengthss: tuple<sequence<LaneGroupSize>, sequence<8>>
  ps_to_rhss_major: tuple<sequence<1>>
  ps_to_rhss_minor: tuple<sequence<0>>
```

**Note:** LaneGroupSize is validated as 64, 32, or 16 (not 8). The validation tries all three and uses whichever passes.

### Why Our Fixes Fail

When we modify `quad_output_ps_minor_offset` (which affects the generated tile distribution encoding), the result likely fails one of:
- Suffix checks (quad pattern no longer matches as suffix of input)
- PS→RHS mapping suffix check (the shifted minor offsets don't match)

**Hypothesis:** The swapped values break the suffix relationship or PS mapping suffix property.

---

## V Tile Transpose Load Bug Status

**Root Cause Confirmed:**
- Dimension index mismatch when ReverseDirection=true
- `quad_output_ps_to_rhss_major0` uses original coordinate space (from swapped QuadInputEncoding)
- `quad_idx_offset` uses reversed coordinate space (from reversed_outer_hs_lengthss)
- Results in: Current `sequence<2, 3, 2>`, Expected `sequence<3, 2, 3>`

**Fix Attempts Summary:**
- 9 different approaches to modify `quad_output_ps_minor_offset` calculation
- All produce the corrected sequence `<3, 2, 3>` → encoding `ps_to_rhss_minor[1] = <2, 3, 2, 4>`
- All fail with same validation error: `TransposeTileDistrChecker<...>::distr_encoding_valid` not satisfied

**Key Insight:**
- The fixed encoding has all valid indices into HsLengthss
- Validation fails on STRUCTURAL pattern matching, not index validity
- The pattern `<2, 3, 2, 4>` is not a suffix of the quad encoding pattern
- This suggests the fix may need to be at a different level:
  - Quad encoding definition for transpose loads
  - QuadInput/QuadOutputEncoding selection logic
  - Validation logic itself

---

## Investigation: K Tile Half-Stride Bug (2024-01-30)

### Test Pattern

Same alternating pattern as Q test applied to K:
```python
# K[row, 0:16] = row_base, K[row, 16:32] = row_base+0.1, etc.
for j in range(seqlen_k):
    row_base = (j + 1) * 1.0
    for chunk in range(d // 16):
        k[:, j, :, chunk*16:(chunk+1)*16] = row_base + chunk * 0.1
```

### Host K Values (seqlen_k=64):

| Row | First byte (hex) |
|-----|------------------|
| 0 | 0x4e |
| 4 | 0x61 |
| 8 | 0x68 |
| 16 | 0x6f |

### Kernel K Tile Thread Buffer:

| Lane | First 32 bytes | Expected row | Actual row |
|------|----------------|--------------|------------|
| Lane 0 | 0x4e, 0x50... | row 0 | row 0 ✓ |
| Lane 32 | 0x4f, 0x51... | row 0 (odd chunks) | row 0 ✓ |
| Lane 8 | 0x61 all | row 8 | row 4 ✗ |
| Lane 40 | 0x61 all | row 8 (odd chunks) | row 4 ✗ |
| Lane 16 | 0x68 all | row 16 | row 8 ✗ |
| Lane 48 | 0x68 all | row 16 (odd chunks) | row 8 ✗ |

### Bug Identified: K Tile Half-Stride

**Lane N gets K row N/2 instead of row N!**

- Lane 0 → row 0 (0/2=0) ✓
- Lane 8 → row 4 (8/2=4) ✗ should be row 8
- Lane 16 → row 8 (16/2=8) ✗ should be row 16

The K tile distribution has a stride that is half of what it should be.

### Root Cause

The v3 pipeline used `WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution` for FP8 QK GEMM, which has a SwizzleB encoding with half-stride in BWarpDstrEncoding. This caused the K tile loading from LDS to have incorrect row mapping.

### Fix Applied (2024-01-30)

**Two changes made to `block_fmha_fwd_v3_pipeline_default_policy.hpp`:**

1. **Changed FP8 warp GEMM** (line ~267): Replace SwizzleB version with non-SwizzleB version
   ```cpp
   // OLD (buggy):
   return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<swizzle_factor>{};

   // NEW (fixed):
   return WarpGemmMfma_f32_32x32x32_fp8_fp8_CTransposed<>{};
   ```

2. **Updated MakeKRegTileDistribution()** (line ~165): Match async_trload pattern for proper block-level distribution encoding.

### Result

- K tile half-stride bug: **FIXED**
- Lane 8 now correctly gets K row 8 (was getting row 4)
- Lane 16 now correctly gets K row 16 (was getting row 8)
- 8-lane offset bug in output: **FIXED**
- Test passes with max diff 0.035 < threshold 0.055

---

## Investigation: Causal + Large Seqlen Failure (2025-01-31)

### Experiment: Full Pytest Run with K Tile Fix

After applying the K tile half-stride fix, ran full pytest suite:

```bash
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python -m pytest op_tests/test_mha_fp8.py -v --tb=short"
```

### Results

**48 failed, 128 passed**

All failures follow a pattern:
- **causal=True** (all causal=False pass)
- **seqlen ≥ 256** (all seqlen < 256 pass)

| Sequence Length | causal=False | causal=True |
|-----------------|--------------|-------------|
| 32, 108, 113, 128 | ✓ PASS | ✓ PASS |
| 256, 512 | ✓ PASS | ❌ FAIL |
| 1023, 1024 | ✓ PASS | ❌ FAIL |
| 2048, 4096 | ✓ PASS | ❌ FAIL |

### Analysis

The K tile half-stride fix resolved the lane mapping issue for small sequences. However, a separate bug exists when:
1. Causal masking is enabled
2. Sequence length requires multiple tile iterations (seqlen ≥ 256, tile size = 64)

This suggests the bug is in one of:
- **Multi-tile K/V loop:** How tiles are iterated when seqlen > tile size
- **Causal mask at tile boundaries:** Mask calculation across tile boundaries
- **Online softmax rescaling:** `m` and `l` accumulator updates between tiles
- **V tile transpose load:** The bypassed `load_tile_transpose()` issue may have runtime effects

### Verification: Causal=True/False with Small Seqlen

```bash
# Both pass with small seqlen
docker exec poyenc-ck bash -c "... python op_tests/test_mha_fp8.py -b 1 -n 1 -q 32 -k 32 -d 128 -dv 128"      # PASS
docker exec poyenc-ck bash -c "... python op_tests/test_mha_fp8.py -b 1 -n 1 -q 32 -k 32 -d 128 -dv 128 -c"   # PASS
```

Output for both:
```
GPU output row 8 sum: 0.000000, row 16 sum: 64.500000
Reference row 8 sum: ~0, row 16 sum: 65.000000
Output max diff: 0.035
```

---

## Key Files Reference

| Component | File Path |
|-----------|-----------|
| v3 Pipeline Policy | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp` |
| v3 Pipeline | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp` |
| v3 Kernel | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp` |
| Masking | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/block_masking.hpp` |
| async_trload Policy | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload_policy.hpp` |
| Transpose Load | `3rdparty/composable_kernel/include/ck_tile/core/tensor/load_tile_transpose.hpp` |
| Warp GEMM Dispatcher | `3rdparty/composable_kernel/include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp` |
| Test | `op_tests/test_mha_fp8.py` |
