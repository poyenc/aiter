# FP8 FMHA v3 Debug Session

**Last Updated:** 2025-01-31

## Current Focus

Investigating **Issue #2: Causal + Large Seqlen Bug** - tests fail when causal=True AND seqlen ≥ 256.

See [issues.md](issues.md) for full issue tracking and test results.

---

## Progress

### 2025-01-31

1. **Fixed Issue #1 (K Tile Half-Stride Bug)**
   - Changed `GetQKBlockGemm()` to use non-SwizzleB warp GEMM
   - Verified fix is equivalent to async_trload policy for FP8/FP16/BF16
   - Small seqlen tests now pass (32, 108, 113, 128)

2. **Discovered Issue #2 (Causal + Large Seqlen Bug)**
   - Ran full pytest suite: 48 failed, 128 passed
   - All failures are causal=True + seqlen≥256
   - All causal=False tests pass regardless of seqlen
   - All small seqlen tests pass regardless of causal

3. **Documented findings**
   - Created `.claude/issues.md` for issue tracking
   - Created `.claude/findings.md` for investigation details
   - Updated CLAUDE.md with workflow instructions

### 2025-01-30

1. Investigated 8-lane offset bug (V[16] data in output row 8)
2. Traced through V tile transpose load - found `quad_output_ps_minor_offset` mismatch
3. 9 fix attempts all failed validation
4. Pivoted to investigate K tile loading
5. Found K tile half-stride bug (lane N gets row N/2)
6. Applied fix to GetQKBlockGemm()

---

## Hypothesis

For Issue #2 (causal + large seqlen), the bug likely involves:

1. **Multi-tile iteration** (HIGH)
   - seqlen=256 requires 4 tile iterations (tile size = 64)
   - Bug may be in K/V tile loop when seqlen > tile size

2. **Causal mask at tile boundaries** (HIGH)
   - Mask calculation may be wrong when attention spans multiple tiles
   - First tile iteration vs subsequent may differ

3. **Online softmax state management** (MEDIUM)
   - `m` (max) and `l` (sum) accumulators updated across tiles
   - Rescaling factor may be computed incorrectly

4. **V tile transpose load** (LOW)
   - Issue #3 is bypassed, not fixed
   - May have runtime effects in multi-tile scenarios

---

## TODO

- [ ] Debug seqlen=256, causal=True case
  - [ ] Add debug prints for tile iteration index
  - [ ] Print attention scores (S) for each tile
  - [ ] Print softmax output (P) for each tile
  - [ ] Print output accumulator (O) after each tile

- [ ] Check causal mask logic
  - [ ] Verify mask coordinates for tile boundaries
  - [ ] Compare v3 vs async_trload mask handling

- [ ] Check online softmax
  - [ ] Print m/l values across tile iterations
  - [ ] Verify rescaling factor calculation

- [ ] Compare pipelines
  - [ ] Run same config with async_trload (if possible)
  - [ ] Diff the pipeline implementations

---

## Next Steps

1. **Immediate:** Add debug prints to v3 pipeline for seqlen=256, causal=True
   - File: `block_fmha_fwd_v3_pipeline.hpp`
   - Print tile iteration index, S, P, O values for specific lanes

2. **Short-term:** Identify which tile iteration introduces the error
   - First tile vs subsequent tiles
   - Tile boundary vs tile interior

3. **Investigation:** Compare causal mask handling
   - v3: `block_fmha_fwd_v3_pipeline.hpp`
   - async_trload: `block_fmha_pipeline_qr_ks_vs_async.hpp`

---

## Quick Reference

### Test Commands

> Replace `<CONTAINER>` and `<WORKSPACE>` with values from `.claude/user.md`

```bash
# Failing test (Issue #2)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 8 -q 256 -k 256 -d 128 -dv 128 -c"

# Passing test (same config, no causal)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 8 -q 256 -k 256 -d 128 -dv 128"
```

### Key Files
- v3 Pipeline: `3rdparty/composable_kernel/.../block_fmha_fwd_v3_pipeline.hpp`
- v3 Policy: `3rdparty/composable_kernel/.../block_fmha_fwd_v3_pipeline_default_policy.hpp`
- Test: `op_tests/test_mha_fp8.py`

### Related Docs
- [issues.md](issues.md) - Issue status and reproduction
- [findings.md](findings.md) - Detailed investigation notes

---

## Notice

If encounter error 3 times and cannot make progress, ask user for help and let user decide what to do next.
