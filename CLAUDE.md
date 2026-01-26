# AIter Project Notes

## Technical Documentation

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
