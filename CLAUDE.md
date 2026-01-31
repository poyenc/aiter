# AIter Project Notes

## Workflow

1. **Before starting**:
   - Read [session.md](.claude/session.md) to understand current progress and next steps
   - Read [issues.md](.claude/issues.md) for issue status and reproduction steps

2. **During work**:
   - Update `session.md` with progress, hypothesis, and TODO changes
   - Update `issues.md` when issue status or test results change
   - Add detailed experiments to `findings.md`

3. **Before ending** (before compacting is triggered OR user exits conversation):
   - Update `session.md` with latest progress and next steps
   - Commit `.claude/*.md` files with a summary of progress

## Technical Documentation

- [Session Notes](.claude/session.md) - Progress, hypothesis, TODO, and next steps
- [Issue Tracker](.claude/issues.md) - Issue status, reproduction steps, and test results
- [Investigation Findings](.claude/findings.md) - Detailed experiments, root cause analysis, and historical notes
- [FP8 Attention Notes](op_tests/FP8_ATTENTION_NOTES.md) - FP8 attention kernel behavior and reference implementation details

## Project Structure

- `aiter/` - Python library source
- `csrc/` - C++/CUDA kernel implementations
- `op_tests/` - Operator tests
- `3rdparty/composable_kernel/` - AMD Composable Kernel library (submodule)

## Key Files

### FP8 Flash Attention
- `csrc/ck_fmha_fwd/src/fmha_fwd_kernel.hpp` - Kernel entry point, scale definitions
- `3rdparty/composable_kernel/.../block_fmha_pipeline_qr_ks_vs_async.hpp` - Online softmax implementation
- `op_tests/test_mha_fp8.py` - FP8 attention tests and reference implementation
