# FP8 FMHA v3 Debug Session

**Last Updated:** 2026-02-03

---

## Current Status

**ROOT CAUSE IDENTIFIED** - P and V tile distribution mismatch in PV GEMM.

See [knowledge.md](knowledge.md) for technical details on the root cause.

---

## Root Cause Summary

P[K=4] is in lane 32 but V[K=4] is in lane 0, causing `P[K=4] × V[K=4] = 124 × 0 = 0`.

| Tile | Distribution Source | K=4 Location |
|------|---------------------|--------------|
| P | `MakePRegTileDistribution()` | Lane 32 |
| V | `MakeVRegTileDistribution()` | Lane 0 |

**Why seqlen_k <= 4 works:** All data in lane 0 for both P and V.
**Why seqlen_k >= 5 fails:** Lane 32 has P[K=4] but V is zeros.

---

## Next Steps

1. Fix `MakeVRegTileDistribution()` to match P's distribution after transpose
2. Or fix `MakePRegTileDistribution()` to match V's distribution
3. Verify fix with full pytest suite

---

## Verified Correct (Ruled Out)

| Component | Status | Notes |
|-----------|--------|-------|
| scale_p = 448 | CORRECT | Debug print confirmed |
| Causal mask formula | CORRECT | Matches reference |
| m (row max) | CORRECT | All lanes match |
| l (row sum) | CORRECT | All lanes match |
| P values (FP8) | CORRECT | All rows match |
| V tile values | CORRECT | Verified with seqlen_k=64 |
| QK GEMM | CORRECT | sp_compute matches reference |

---

## TODO

- [x] Verify scale_p → 448 (correct)
- [x] Verify mask formula → correct
- [x] Verify V tile lane mapping → correct
- [x] Identify root cause → P/V distribution mismatch
- [ ] Fix distribution mismatch in v3 policy
- [ ] Run full pytest suite to verify fix

---

## Test Commands

```bash
# Full pytest suite (source of truth)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python -m pytest op_tests/test_mha_fp8.py -v"

# Single failing test (causal)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 64 -d 128 -dv 128 -c"

# Single passing test (non-causal)
docker exec <CONTAINER> bash -c "cd <WORKSPACE> && rm -f aiter/jit/*.so && python op_tests/test_mha_fp8.py -b 1 -n 1 -q 256 -k 64 -d 128 -dv 128"
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
