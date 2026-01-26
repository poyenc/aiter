# FP8 Attention Reference Implementation Notes

## Problem

`attention_fp8_ref()` in `test_mha_fp8.py` had high numerical error (0.07~0.20) compared to BF16 kernel output, especially with:
- seqlen_q >= 512
- seqlen_k >= 256
- causal = True

## Root Cause

The reference implementation was normalizing P (softmax output) **before** FP8 quantization, but the kernel uses **online softmax** which quantizes **unnormalized** P.

### Kernel Behavior (block_fmha_pipeline_qr_ks_vs_async.hpp)

```cpp
// Line 651: Compute unnormalized P (NOT divided by sum yet)
p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);

// Line 771: Quantize unnormalized P to FP8
// p_compute_element_func = scales<>{scale_p} where scale_p = 448
cast_tile<PDataType>(tile_elementwise_in(p_compute_element_func, p_compute));

// Lines 861-868: Normalize output at the END
o_acc(i_idx, j_idx) = o_acc(i_idx, j_idx) * (1.0f / l[i_idx]);
```

### Why This Matters

FP8 E4M3 has limited precision. When you quantize:
- **Normalized P** (values 0~1): Small values lose precision
- **Unnormalized P** (values 0~448 after scaling): Better utilizes FP8 dynamic range

The error patterns are different, so comparing reference (normalized before quant) vs kernel (unnormalized before quant) produces systematic error.

## Solution

Modified `attention_fp8_ref()` to match kernel behavior:

```python
# BEFORE (wrong):
p_normalized = softmax(scores)  # normalized
p_fp8 = (p_normalized * 448).to(fp8)
output = p_fp8 @ v

# AFTER (correct):
p_compute = exp2(scale_s * (scores - row_max))  # unnormalized
p_fp8 = (p_compute * 448).to(fp8)
output = p_fp8 @ v
p_sum = p_compute.sum(dim=-1, keepdim=True)
output = output / p_sum  # normalize at the end
```

### Shape Mismatch Fix

Also fixed a tensor shape issue:
- scores shape: `[b, h, t, s]`
- p_sum shape: `[b, h, t, 1]`
- output shape: `[b, t, h, d]`

Need to permute p_sum before division:
```python
p_sum_for_div = p_sum.permute(0, 2, 1, 3)  # [b, h, t, 1] -> [b, t, h, 1]
output = output / p_sum_for_div
```

## Key Constants (gfx950)

- `fp8_max = 448.0` (FP8 E4M3 max representable value)
- `scale_s = (1/sqrt(d)) * log2(e) * q_descale * k_descale`
- `scale_p = fp8_max = 448.0`
- `scale_o = v_descale / scale_p`

## Files Modified

- `op_tests/test_mha_fp8.py`: Fixed `attention_fp8_ref()` function

## Related Kernel Files

- `csrc/ck_fmha_fwd/src/fmha_fwd_kernel.hpp`: Kernel entry point, scale definitions
- `3rdparty/composable_kernel/.../block_fmha_pipeline_qr_ks_vs_async.hpp`: Online softmax implementation

## Verification

All 176 pytest cases passed after the fix.
