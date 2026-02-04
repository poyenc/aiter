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

## Build and Run Environment

User-specific settings are in `.claude/user.md` (see `.claude/user.md.example` for template).

### Docker Container
```bash
# Use container name from .claude/user.md or user.md.example
docker exec -it <CONTAINER> bash
```

### Build & Run
Run any Python script that invokes a `@compile_ops` annotated function.
This triggers JIT HIP kernel compilation and loads the compiled `.so` under `aiter/jit/`.

### Clean Rebuild
If kernel code changed, remove related `.so` before running:
```bash
rm aiter/jit/*.so
```

### Example
```bash
# Inside container
rm -f aiter/jit/*.so && python -m pytest op_tests/test_mha_fp8.py -v
```

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

## Best Practices

- **Print all values:** When comparing kernel registers with reference, print all thread_buffer values, not just the first few. Matching on a few elements doesn't mean all match. Do this by hardcoding the lane ID and repeating the cycle (update lane ID → run test → collect output) until you have enough data to compare. 
- **Verify completely before moving on:** When debugging intermediate values (sp_compute, m, l, P, V, o_acc), verify ALL rows/cols before concluding correctness. Ask the user to confirm findings before proceeding to the next stage.
- **Ask user for direction:** Before moving to the next debugging step, present options to the user and let them decide the direction. Do not assume what to check next.
- **Batch kernel changes:** Debug prints require expensive recompilation. Plan all changes at once rather than incrementally.
- **Only record verified facts:** Do not write conclusions or hypotheses in this document until they are verified by experiments. Record experiment results and observations, not speculation.

## Key Files

### FP8 Flash Attention
- `csrc/ck_fmha_fwd/src/fmha_fwd_kernel.hpp` - Kernel entry point, scale definitions
- `3rdparty/composable_kernel/.../block_fmha_pipeline_qr_ks_vs_async.hpp` - Online softmax implementation
- `op_tests/test_mha_fp8.py` - FP8 attention tests and reference implementation
