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

---

## V3 Kernel Assembly Analysis (FP8 vs BF16)

### Kernel Metadata Comparison

| Property | FP8 (fp8bf16) | BF16 |
|----------|---------------|------|
| Tile size | 256×64×128 (M×N×K) | 256×32×128 |
| VGPRs | 256 | 256 |
| SGPRs | 91 | ~90 |
| LDS | 51,200 bytes | ~51,200 bytes |
| Occupancy | 2 waves/CU | 2 waves/CU |
| MFMA count | 176 | 88 |
| MFMA instruction | `v_mfma_f32_32x32x16_fp8_fp8` | `v_mfma_f32_32x32x16_bf16` |

### Phase Structure (4 Phases per Iteration)

Both FP8 and BF16 V3 kernels use the same phase structure with warp group specialization (Wave0-3 and Wave4-7):

| Phase | Purpose | Key Operations |
|-------|---------|----------------|
| Phase 0 | S = Q × K^T + softmax | MFMA (QK), v_exp_f32, v_add_f32, v_permlane32_swap |
| Phase 1 | Memory load (next K/V) | buffer_load to LDS, ds_read_b64_tr (transpose read) |
| Phase 2 | O += P × V | MFMA (PV), v_max3_f32, v_fma_f32 |
| Phase 3 | Memory load (prep) | buffer_load, ds_read for next iteration |

### P Conversion Location Difference (Potential Optimization Issue)

**BF16 V3:** P→BF16 conversion happens **in Phase 0** after exp():
```asm
; Phase 0 (line ~773-795)
v_exp_f32_e32 v143, v165
v_add_f32_e32 v147, v145, v143
...
v_cvt_pk_bf16_f32 v118, v143, v145   ; 8× conversions in Phase 0
v_cvt_pk_bf16_f32 v119, v146, v138
; Phase 1 starts
```

**FP8 V3:** P→FP8 conversion happens **between Phase 1 and Phase 2** (outside phase markers):
```asm
; Phase 1 (line 921)
buffer_load_dwordx4 ... lds
ds_read_b64_tr_b8 ...              ; 16× LDS reads
; Boundary check (v_cmp, v_cndmask)
; .LBB0_24:
v_mul_f32_e32 v217, v150, v217     ; Scale by p_scale
v_med3_f32 v252, v217, s28, v251   ; Clamp to FP8 range
v_cmp_nlg_f32_e64 vcc, |v217|, s21 ; NaN/Inf check
v_cndmask_b32_e32 v217, v252, v217
v_cvt_pk_fp8_f32 v252, v217, v217  ; 32× conversions between phases
...
; Phase 2 (line 1319)
```

### FP8 Quantization Extra Operations

FP8 P conversion requires additional operations vs BF16:

| Step | FP8 | BF16 |
|------|-----|------|
| Scale | `v_mul_f32` (apply p_scale) | None |
| Saturate | `v_med3_f32` (clamp to FP8 range) | None |
| NaN check | `v_cmp_nlg_f32` + `v_cndmask_b32` | None |
| Convert | `v_cvt_pk_fp8_f32` | `v_cvt_pk_bf16_f32` |
| Count | 32 conversions | 8 conversions |

### Phase 2 Data Flow (P×V MFMA)

Phase 2 uses the P conversion results:

```asm
; Phase 2: Pack FP8 P values for MFMA
v_lshlrev_b16_e32 v191, 8, v210      ; Use v210 from P conversion
v_bitop3_b16 v191, v211, v191, s12   ; Pack bytes
v_or_b32_sdwa v209, v191, v202       ; Create MFMA input pair

; V values from Phase 1's ds_read (transposed)
v_perm_b32 v142, v142, v142, s29     ; Transpose V

; MFMA: O += P × V
v_mfma_f32_32x32x16_fp8_fp8 v[34:49], v[144:145], v[202:203], v[34:49]
;                                     ^^^^^^^^^ V   ^^^^^^^^^ P (FP8)
```

### Root Cause: Compiler Code Sinking

The compiler optimization called "code sinking" moves data preparation code closer to where the data is consumed. This hurts GPU kernel performance because we want to overlap computation with memory latency.

**Source code comment (block_fmha_fwd_v3_pipeline.hpp:843-846):**
```cpp
/// Note: The compiler keeps sinking the conversion instructions because the
/// result 'p' is only consumed later. To anchor them here, we rewrite
/// the cast_tile() call as inline assembly, forcing the conversions to be
/// emitted at this point.
```

**BF16/FP16: Uses inline asm to prevent sinking**
```cpp
// Lines 857-862: BF16 uses asm volatile wrapper
auto casted = detail::cvt_pk_bf16_f32(x, y);  // asm volatile inside

// The wrapper function (lines 232-239):
CK_TILE_DEVICE bf16x2_t cvt_pk_bf16_f32(float a, float b)
{
    bf16x2_t result;
    asm volatile("v_cvt_pk_bf16_f32 %[result], %[a], %[b]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b));
    return result;
}
```

**FP8: Uses regular type_convert (NO inline asm protection)**
```cpp
// Lines 863-867: FP8 uses regular conversion
else if constexpr(std::is_same_v<PDataType, fp8_t>)
{
    sp(sp_reg_idx).p.thread_buf_[idx]     = type_convert<PDataType>(x);
    sp(sp_reg_idx).p.thread_buf_[idx + 1] = type_convert<PDataType>(y);
}
```

### Why FP8 P Conversion Gets Sunk

1. `type_convert<fp8_t>()` is NOT wrapped in `asm volatile`
2. Compiler sees P is only consumed in Phase 2's MFMA
3. Compiler sinks the conversion to be close to Phase 2
4. Result: P→FP8 conversion ends up between Phase 1 and Phase 2

### Fix: Add Inline ASM Wrapper for FP8

To prevent sinking, FP8 needs an `asm volatile` wrapper similar to BF16:
```cpp
CK_TILE_DEVICE fp8x2_t cvt_pk_fp8_f32(float a, float b)
{
    fp8x2_t result;
    asm volatile("v_cvt_pk_fp8_f32 %[result], %[a], %[b]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b));
    return result;
}
```

Note: FP8 also requires additional operations (scale, saturate, NaN check) before conversion, which would also need inline asm protection.

### Assembly File Locations

```
aiter/jit/build/mha_fwd_fp8bf16_*/build/*v3*gfx950.s   # FP8 kernel
aiter/jit/build/mha_fwd_bf16_*/build/*v3*gfx950.s      # BF16 kernel
```

Enable `--save-temps` in optCompilerConfig.json to generate `.s` files.

---

## V3 Pipeline Instruction Scheduler Design

The FMHA V3 pipeline uses AMD's `__builtin_amdgcn_sched_group_barrier()` intrinsic for **explicit instruction scheduling control**. This maximizes GPU utilization by precisely overlapping computation with memory operations.

**Source:** `3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp`

### The `sched_group_barrier` Intrinsic

```cpp
__builtin_amdgcn_sched_group_barrier(uint32_t mask, uint32_t count, uint32_t flags);
```

| Parameter | Description |
|-----------|-------------|
| `mask` | Bitmask selecting instruction group (MFMA, VMEM, LDS, VALU, SALU) |
| `count` | Number of instructions of that type to schedule before proceeding |
| `flags` | Reserved (always 0) |

**Scheduler Group Masks** (from `include/ck_tile/core/arch/arch.hpp`):

| Mask | Name | Description |
|------|------|-------------|
| `0x002` | VALU | Vector ALU operations |
| `0x004` | SALU | Scalar ALU operations |
| `0x008` | MFMA | Matrix FMA (MFMA) instructions |
| `0x020` | VMEM_READ | Global memory reads |
| `0x040` | VMEM_WRITE | Global memory writes |
| `0x100` | DS_READ | LDS read operations (including `ds_read_b64_tr_b8` transpose reads) |
| `0x200` | DS_WRITE | LDS write operations |
| `0x400` | TRANS | Transcendental unit (`v_exp_f32`, `v_log_f32`, `v_rcp_f32`, etc.) |

**Note:** `ds_read_b64_tr_b8` (LDS transpose read) is DS_READ (`0x100`), NOT TRANS (`0x400`). TRANS is the transcendental/special-function unit.

### How `sched_group_barrier` Works (Placement Rules)

**Critical:** `sched_group_barrier` directives are placed **AFTER** the code whose instructions they reorder. They operate on the scheduling region defined by the preceding `sched_barrier(0)` fence(s).

**Execution model:**
1. `sched_barrier(0)` opens a scheduling region (a fence — no instructions cross it)
2. Instructions (MFMA, VALU, TRANS, etc.) are emitted into the region
3. `sched_group_barrier(mask, count, 0)` directives define the desired interleaving pattern
4. The LLVM scheduler reorders the instructions in the region to satisfy the pattern
5. The next `sched_barrier(0)` closes the region

**Example — Phase 2 code structure:**
```cpp
__builtin_amdgcn_sched_barrier(0);           // Open region
// ... code with MFMAs, v_exp_f32, v_fma, v_max3 ...
cl_calc(sp_reg_idx, gemm1);                  // Emits MFMAs + fmha_alu0
Scheduler::schedule(cl_p, number<2>{});      // Emits sched_group_barriers AFTER the code
__builtin_amdgcn_sched_barrier(0);           // Close region
```

The barriers at `Scheduler::schedule()` tell the scheduler: "take all the instructions above (between the fences) and reorder them so that between consecutive MFMAs there are at least N TRANS and M VALU instructions."

**In assembly output**, `sched_group_barrier` comments appear after the instructions they schedule — this is correct and expected, not a bug:

```asm
; sched_barrier mask(0x00000000)              ; <-- region start
v_mfma_f32_32x32x16_fp8_fp8 ...              ; instructions in the region
v_exp_f32_e32 ...
v_fma_f32 ...
v_mfma_f32_32x32x16_fp8_fp8 ...
v_exp_f32_e32 ...
; sched_group_barrier mask(0x00000008) size(1) ; <-- MFMA:1 (applied to above)
; sched_group_barrier mask(0x00000400) size(2) ; <-- TRANS:2
; sched_group_barrier mask(0x00000002) size(4) ; <-- VALU:4
; ...repeating pattern...
; sched_barrier mask(0x00000000)              ; <-- region end
```

**`asm volatile` and sched_group_barrier:** Instructions inside `asm volatile` are invisible to the scheduler — they cannot be counted toward any `sched_group_barrier` quota and cannot be reordered. They stay in source-code order. Only non-`asm volatile` instructions participate in the scheduling.

### Two Warp Group Architecture

The V3 pipeline divides the thread block (512 threads = 8 waves) into **two warp groups**:

```cpp
const index_t warp_group_id = get_warp_id() / 4;  // line 480
```

| Warp Group | Waves | Role |
|------------|-------|------|
| 0 | 0-3 | Executes `core_loop(number<0>{})` |
| 1 | 4-7 | Executes `core_loop(number<1>{})` |

They run different code paths with **asymmetric scheduling** (lines 1256-1272):

```cpp
if(warp_group_id == 0) {
    __builtin_amdgcn_s_setprio(0);  // Lower priority
    while(core_loop(number<0>{}));
}
if(warp_group_id != 0) {
    __builtin_amdgcn_s_setprio(1);  // Higher priority
    while(core_loop(number<1>{}));
}
```

### Four-Phase Execution Model

Each iteration consists of 4 phases. The `CoreLoopScheduler` template (lines 40-189) defines the scheduling pattern for each (WarpGroup, Phase) combination:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Phase Timeline (One Iteration)                            │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────┤
│          │ Phase 0  │ Phase 1  │ Phase 2  │ Phase 3  │ Phase 0  │    ...    │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────────┤
│WarpGrp 0 │ GEMM0    │ Load K   │ GEMM1    │ Load V   │ GEMM0    │    ...    │
│          │ +softmax │ +mask    │ +scale   │          │ +softmax │           │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────────┤
│WarpGrp 1 │ Load V   │ GEMM0    │ Load K   │ GEMM1    │ Load V   │    ...    │
│          │          │ +softmax │ +mask    │ +scale   │          │           │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────────┘
```

This **ping-pong pattern** ensures:
- While WarpGroup 0 computes GEMM0 (Q×K), WarpGroup 1 loads data
- While WarpGroup 1 computes GEMM1 (P×V), WarpGroup 0 loads data

### Scheduler Patterns per Phase

**Default Scheduling (bf16/fp16):**

| Effective Phase | Barrier Sequence | Purpose |
|-----------------|------------------|---------|
| GEMM0 compute | `8 × (MFMA:1, TRANS:2, VALU:2)` | 8 MFMA + 16 transpose + 16 VALU |
| Load | `VALU:2, SALU:4` | Light work during K/V load |
| GEMM1 compute | `VALU:4` (if packed), `8 × (MFMA:1, VALU:4)` | 8 MFMA + VALU |

**FP8 Scheduling (asymmetric, exp2 in Phase 0):**

| Effective Phase | Barrier Sequence | Purpose |
|-----------------|------------------|---------|
| GEMM0 compute (Phase 0) | K iter 0: `8 × (MFMA:1, TRANS:4, VALU:4)` | exp2(sp_delta) + rowsum + permlane |
| | K iter 1: `8 × (MFMA:1, VALU:6)` | VALU-heavy: P scale + cvt_pk_fp8 + o_acc rescale |
| Load | `VALU:2, SALU:4` | Light work during K/V load (same as default) |
| GEMM1 compute (Phase 2) | `VALU:4` (if packed), first half: `8 × (MFMA:1, VALU:4)` | v_perm + v_max3 + permlane chain |
| | second half: `8 × (MFMA:1, VALU:3)` | Looser constraint for data-dep limited v_fma |

**Current GEMM1 scheduling gap:** MFMAs 9-12 are back-to-back due to data dependency (serial max3→permlane→max→mul→fma chain). 29 `v_pk_mul_f32` + 1 `v_exp_f32` trail after the last MFMA with zero interleaving.

**Explored and reverted — moving exp2 to Phase 2:** Moving 32 `v_exp_f32` from Phase 0 to Phase 2 would provide TRANS for GEMM1 interleaving, but reducing Phase 0 instruction density caused the compiler to sink FP8 byte extraction (`packed & 0xFF`) into Phase 1. Anchoring with `asm volatile` fixed the sinking but introduced 32 extra `v_lshlrev_b16`/`v_bitop3_b16` per GEMM1 (compiler needs to repack individual fp8 bytes into 32-bit registers for MFMA). The root cause: `bit_cast<fp8_t>(uint8_t)` is zero-cost, but `asm volatile` forces bytes into separate VGPRs, preventing the compiler from keeping the packed representation.

**Phase-to-effective mapping:** WG0: Phase N = effective N; WG1: Phase N = effective (N+3)%4

### The MFMA Pattern

The core compute phases use scheduling groups to interleave MFMA with VALU.

**GEMM0 (QK matmul):** `kMfmaPerWarpGemm0` groups, each: MFMA:1, TRANS:2, VALU:2

**GEMM1 (PV matmul):** `kMfmaPerWarpGemm1` groups, each: MFMA:1, VALU:4

**kMfmaPerWarpGemm formula (unified for all dtypes):**
```cpp
kMfmaPerWarpGemm = MIterPerWarp * NIterPerWarp * KIterPerWarp *
                   (WarpGemm::kK / WarpGemm::WarpGemmAttribute::Impl::kK);
```

The last factor (`WarpGemm::kK / Impl::kK`) is the internal K iteration count (kKIter) — the number of hardware MFMA instructions per warp gemm call:
- **bf16/fp16:** kKIter=1 (K=16 warp gemm, K=16 base MFMA) → kMfmaPerWarpGemm = 8
- **fp8:** kKIter=2 (K=32 warp gemm wrapping 2× K=16 `v_mfma_f32_32x32x16_fp8_fp8`) → kMfmaPerWarpGemm = 16

No dtype-specific override needed — the formula derives the correct count for all dtypes.

### CoreLoopScheduler Static Dispatch Pitfall

`CoreLoopSchedulerDefaultBase` defines `schedule()` which calls `schedule_gemm0_compute()` and `schedule_gemm1_compute()`. Since these are **static methods** (not virtual), `schedule()` always calls the **base class** versions. To override any phase helper in a derived specialization, you must also override `schedule()` to call the derived version.

**Current state:** The fp8 specialization overrides `schedule_gemm0_compute()`, `schedule_gemm1_compute()`, and `schedule()` with asymmetric sched_group_barrier patterns. bf16 and fp16 use the default base.

### Phase Boundary Markers

```cpp
// lines 12-20
#define ASM_MARKER(marker)               \
    __builtin_amdgcn_sched_barrier(0);   \
    asm volatile("; [POYENC] " #marker); \
    __builtin_amdgcn_sched_barrier(0);
```

These serve two purposes:
1. **Prevent code sinking**: Compiler cannot move instructions across `asm volatile`
2. **Debug markers**: Comments appear in disassembly for profiling

### Phase Implementation in Core Loop

**Phase 0 (WarpGroup 0) - GEMM0 + Softmax:**
```cpp
__builtin_amdgcn_sched_barrier(0);
s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();  // Wait for LDS operations
__builtin_amdgcn_sched_barrier(0);
cl_calc(xdl_SP_p01_reg_idx, gemm0);  // GEMM0: Q × K^T
fmha_alu1(xdl_SP_p23_reg_idx);       // Softmax exp + rowsum
fmha_logits_trans(xdl_SP_p01_reg_idx);

Scheduler::schedule(cl_p, number<0>{});  // Emit scheduling barriers
__builtin_amdgcn_sched_barrier(0);       // Force barrier boundary
```

**Phase 1 (WarpGroup 0) - Load K:**
```cpp
s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();  // Wait for DMA
__builtin_amdgcn_s_barrier();                        // Workgroup sync
cl_load(memK, K_w0_lds_wr_idx, V_w0_lds_rd_idx);     // Async K load + V LDS read
Scheduler::schedule(cl_p, number<1>{});
fmha_mask(xdl_SP_p01_reg_idx);                       // Apply attention mask
```

**Phase 2 (WarpGroup 0) - GEMM1:**
```cpp
s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();  // Wait for LDS
__builtin_amdgcn_s_barrier();        // Sync
cl_calc(xdl_SP_p23_reg_idx, gemm1);  // GEMM1: P × V

Scheduler::schedule(cl_p, number<2>{});
fmha_alu_D_upd();                    // Rescale O accumulator
```

**Phase 3 (WarpGroup 0) - Load V:**
```cpp
s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
__builtin_amdgcn_s_barrier();
cl_load(memV, V_w0_lds_wr_idx, K_w0_lds_rd_idx);  // Async V load + K LDS read

Scheduler::schedule(cl_p, number<3>{});
kv_token_start += kN0;  // Move to next token
```

**Note:** All `s_waitcnt` calls use CK core's architecture-aware API from `arch.hpp` (supports GFX9/GFX11/GFX12 via layout structs with `static_assert` validation). Parameter order is `<vmcnt, expcnt, lgkmcnt>`, with defaults at max (no-wait).

### Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Iteration N                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─── Phase 0 ───┐  ┌─── Phase 1 ───┐  ┌─── Phase 2 ───┐  ┌─── Phase 3 ───┐ │
│  │               │  │               │  │               │  │               │ │
│  │   WG0: GEMM0  │  │   WG0: Load K │  │   WG0: GEMM1  │  │   WG0: Load V │ │
│  │   8×MFMA      │  │   async_load  │  │   8×MFMA      │  │   async_load  │ │
│  │   softmax     │  │   s_barrier   │  │   rescale O   │  │   s_barrier   │ │
│  │               │  │   fmha_mask   │  │               │  │               │ │
│  ├───────────────┤  ├───────────────┤  ├───────────────┤  ├───────────────┤ │
│  │               │  │               │  │               │  │               │ │
│  │   WG1: Load V │  │   WG1: GEMM0  │  │   WG1: Load K │  │   WG1: GEMM1  │ │
│  │   async_load  │  │   8×MFMA      │  │   async_load  │  │   8×MFMA      │ │
│  │               │  │   softmax     │  │   fmha_mask   │  │   rescale O   │ │
│  │               │  │               │  │               │  │               │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘ │
│                                                                              │
│  ◄───sched_barrier───►◄───sched_barrier───►◄───sched_barrier───►           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Packed FP32 Support

When using packed FP32 operations (`v_pk_mul_f32`), additional VALU slots are needed:

```cpp
// lines 69-71, 105-107
#if !CK_TILE_DISABLE_PACKED_FP32
    __builtin_amdgcn_sched_group_barrier(0x002, 4, 0); // Extra 4 VALU for v_pk_mul_f32
#endif
```

### Inline Assembly Wrappers

The file defines custom asm wrappers (lines 192-261) to prevent code motion:

| Function | Purpose |
|----------|---------|
| `fma_impl_vsv()` | FMA with scalar operand |
| `cvt_pk_fp8_f32()` | FP8 packing (prevents sinking) |
| `cvt_pk_bf16_f32()` | BF16 packing |
| `pk_mul_f32()` | Packed FP32 multiply |

### asm volatile vs Instruction Scheduler

**`asm volatile` is a black box to the LLVM instruction scheduler.** The scheduler cannot resolve the instruction type (VALU, MFMA, TRANS, etc.) inside an `asm volatile` block. This has two consequences:

1. **`sched_group_barrier` cannot count `asm volatile` instructions.** If `sched_group_barrier(VALU, 4, 0)` requests 4 VALU instructions, any VALU wrapped in `asm volatile` is invisible — the scheduler cannot use it to satisfy the quota.

2. **The scheduler cannot reorder `asm volatile` relative to other instructions.** This is the intended purpose (preventing code sinking), but it also prevents beneficial reordering like interleaving v_fma with MFMAs.

**Tradeoff:** `asm volatile` is needed to anchor instructions in a specific phase (e.g., `cvt_pk_fp8_f32` in Phase 0 to prevent sinking to Phase 2). But for instructions where the compiler already has the right placement and we just want better interleaving, replacing `asm volatile` with a compiler intrinsic (e.g., `__builtin_fmaf()`) makes the instruction visible to the scheduler.

**Example (tried and reverted — worse performance):** In fp8 GEMM1 (Phase 2), `fma_impl_vsv()` wraps `v_fma_f32` in `asm volatile`. Replacing with `__builtin_fmaf()` for fp8 made v_fma visible to the scheduler, which interleaved them with MFMAs #8-12. However, profiling showed **worse performance** due to:

1. **Matrix core contention from `v_pk_fma_f32`**: The compiler merged some FMA pairs into packed `v_pk_fma_f32`, which execute on the matrix core (same unit as `v_mfma`). The compiler has a workaround that detects this contention and selectively unpacks some back to scalar `v_fma_f32`, but 3 out of 16 pairs remained packed — adding 3 extra matrix core instructions competing with 16 MFMAs.

2. **VALU quota displacement**: `sched_group_barrier(VALU, N)` counts v_fma toward its quota, displacing critical-path VALU work (v_perm for byte packing, v_max3 for row reduction, v_permlane) to worse positions. These prepare MFMA input operands and feed the next iteration — delaying them hurts more than the v_fma interleaving helps.

3. **`asm volatile` v_fma still fills gaps via source ordering**: Even though `asm volatile` is invisible to the scheduler, the hardware still executes v_fma in the gaps between MFMAs based on source code placement (after the max chain completes). The scheduling was already reasonable; making it "smarter" introduced worse tradeoffs.

**Lesson:** Making instructions visible to the scheduler is not always beneficial. If the scheduler's VALU budget gets consumed by non-critical-path work (v_fma for sp_delta) at the expense of critical-path work (v_perm/v_max3 for MFMA operand preparation), overall throughput decreases.

**Packed FP32 instructions and matrix core:** `v_pk_fma_f32` and `v_pk_mul_f32` execute on the **matrix core**, the same unit as `v_mfma`. They cannot run simultaneously with MFMA instructions. The compiler has a workaround that detects potential contention and selectively unpacks some packed FP32 back to scalar instructions, but it doesn't catch all cases.

### Basic Block Boundaries and sched_group_barrier

**`sched_group_barrier` operates within a single basic block.** A basic block is a straight-line code sequence with no branches. Conditional branches (e.g., `s_cbranch_scc1` in fmha_mask) create basic block boundaries that the scheduler cannot cross.

**Impact on fmha_mask:** The per-pixel mask check in fmha_mask uses `s_cbranch_scc1`, which splits the code into separate basic blocks. Even though fmha_mask has ~96 VALU instructions (v_add, v_cmp, v_cndmask ×32), these are isolated in their own basic block and **cannot be interleaved with MFMAs** by `sched_group_barrier`.

**Consequence for VALU budget:** When computing `sched_group_barrier(VALU, N)` counts for GEMM1 Phase 2, only the ~30 VALU in the same basic block as the MFMAs are available (v_perm, v_max3 chain, permlane/max/mul). Requesting more VALU than available (e.g., VALU:14 assuming fmha_mask VALU are available) creates an unsatisfiable constraint, causing the scheduler to give up and issue MFMAs back-to-back.

### Performance Optimization Rationale

1. **Hide MFMA Latency**: MFMA takes ~32 cycles; interleaving with TRANS/VALU keeps matrix cores busy

2. **Overlap Memory and Compute**: While one warp group computes, the other loads next tiles

3. **Explicit Scheduling**: Without `sched_group_barrier`, the compiler might:
   - Issue all MFMAs together (causing pipeline stalls)
   - Move loads too late (causing memory latency exposure)
   - Reorder softmax ops (breaking dependencies)

4. **Priority Control**: `s_setprio(1)` for WarpGroup 1 ensures its compute phases aren't starved by WarpGroup 0's memory operations

### CoreLoopScheduler Template

After refactoring (2026-02-07), the scheduler uses dtype-aware dispatch:

```cpp
template <typename PipelineProblem>
struct CoreLoopScheduler
    : CoreLoopSchedulerImpl<PipelineProblem, QDataType, KDataType, VDataType>
{};
```

- `CoreLoopSchedulingParams<Problem>` — auto-derives MFMA counts from tile/gemm config
- `CoreLoopSchedulerDefaultBase<Problem>` — reusable phase helpers using `LLVMSchedGroupMask` enum
- `CoreLoopSchedulerImpl<Problem, Q, K, V>` — dtype-specialized dispatch (fp8 has asymmetric scheduling; bf16/fp16 use default base)

The old `kIsMasking` template parameter was removed — masking is handled separately by `fmha_mask()`, and both mask/nomask variants have identical scheduling patterns (verified via assembly comparison).
