#include "mha_common.h"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include <string>
#include <type_traits>
#include <utility>

#include "fmha_fwd_v3.hpp"
#include "mask.hpp"

namespace aiter {
namespace torch_itfs {

std::vector<at::Tensor> fmha_v3_varlen_fwd_ck(const at::Tensor& q,            // [total_q, hq, d]
                                              const at::Tensor& k,            // [total_k, hk, d]
                                              const at::Tensor& v,            // [total_k, hk, d]
                                              const at::Tensor& cu_seqlens_q, // [b+1]
                                              const at::Tensor& cu_seqlens_k, // [b+1]
                                              int max_seqlen_q,
                                              int max_seqlen_k,
                                              float softmax_scale,
                                              float logits_soft_cap,
                                              bool is_causal,
                                              std::optional<const at::Tensor> q_descale, // [1]
                                              std::optional<const at::Tensor> k_descale, // [1]
                                              std::optional<const at::Tensor> v_descale  // [1]
)
{
    auto q_dtype = q.scalar_type();
    bool is_qkv_fp8 =
        q_dtype == at::ScalarType::Float8_e4m3fn || q_dtype == at::ScalarType::Float8_e4m3fnuz;

    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16 ||
                    is_qkv_fp8,
                "FlashAttention only support fp16, bf16 and fp8_e4m3 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    std::string dtype_str = torchDTypeToStr(c10::scalarTypeToTypeMeta(q_dtype));
    if(is_qkv_fp8)
    {
        dtype_str = "fp8bf16"; // only support bf16 out for fp8
    }

    quant_scale_enum qscale_type =
        q_descale.has_value() ? quant_scale_enum::pertensor : quant_scale_enum::no_scale;

    TORCH_CHECK(q_descale.has_value() == k_descale.has_value() &&
                    k_descale.has_value() == v_descale.has_value(),
                "q_descale, k_descale, v_descale must be all provided or all not provided");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size  = cu_seqlens_q.numel() - 1;
    int num_heads         = sizes[1];
    const int head_size_q = q.size(-1);
    const int head_size_v = v.size(-1);
    const int num_heads_k = k.size(-2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_q <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_v <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_q % 8 == 0,
                "query, key, value, and out_ must have a head_size_q that is a multiple of 8");
    TORCH_CHECK(head_size_v % 8 == 0,
                "query, key, value, and out_ must have a head_size_q that is a multiple of 8");
    TORCH_CHECK(
        num_heads % num_heads_k == 0,
        "ck_tile::number of heads in key/value must divide ck_tile::number of heads in query");

    const int total_q = q.size(0);
    CHECK_SHAPE(q, total_q, num_heads, head_size_q);
    const int total_k = k.size(0);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_q);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);

    mask_info mask;
    if(is_causal)
    {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        std::string mask_identify = "b:-1,0";
        mask = mask_info::decode(mask_identify, max_seqlen_q, max_seqlen_k); // casual
    }
    else
    {
        mask = mask_info::decode("0", max_seqlen_q, max_seqlen_k); // no mask
    }

    fmha_fwd_traits traits{head_size_q,
                           head_size_q,
                           dtype_str,
                           true,
                           true,
                           0.f < logits_soft_cap,
                           mask.type,
                           bias_enum::no_bias,
                           false,
                           false,
                           qscale_type};

    fmha_fwd_args args;

    args.batch        = batch_size;
    args.max_seqlen_q = max_seqlen_q;
    args.hdim_q       = head_size_q;
    args.hdim_v       = head_size_v;
    args.nhead_q      = num_heads;
    args.nhead_k      = num_heads_k;

    args.scale_s         = softmax_scale;
    args.logits_soft_cap = logits_soft_cap;

    args.seqstart_q_ptr  = cu_seqlens_q.data_ptr();
    args.seqstart_k_ptr  = cu_seqlens_k.data_ptr();
    args.seqlen_q_ptr    = nullptr;
    args.seqlen_k_ptr    = nullptr;
    args.cu_seqlen_q_ptr = nullptr;
    args.cu_seqlen_k_ptr = nullptr;

    args.window_size_left  = mask.left;
    args.window_size_right = mask.right;
    args.mask_type         = static_cast<ck_tile::index_t>(mask.type);

    args.q_ptr          = q.data_ptr();
    args.stride_q       = q.stride(0);
    args.nhead_stride_q = q.stride(1);

    args.k_ptr          = k.data_ptr();
    args.stride_k       = k.stride(0);
    args.nhead_stride_k = k.stride(1);

    args.v_ptr          = v.data_ptr();
    args.stride_v       = v.stride(0);
    args.nhead_stride_v = v.stride(1);

    args.q_descale_ptr = q_descale.has_value() ? q_descale.value().data_ptr() : nullptr;
    args.k_descale_ptr = k_descale.has_value() ? k_descale.value().data_ptr() : nullptr;
    args.v_descale_ptr = v_descale.has_value() ? v_descale.value().data_ptr() : nullptr;

    auto opts           = q.options();
    auto out_type       = dtype_str == "fp8bf16" ? at::ScalarType::BFloat16 : q_dtype;
    at::Tensor out      = torch::empty({total_q, num_heads, head_size_v}, opts.dtype(out_type));
    args.o_ptr          = out.data_ptr();
    args.stride_o       = out.stride(0);
    args.nhead_stride_o = out.stride(1);

    auto stream = at::cuda::getCurrentHIPStream().stream();
    ck_tile::stream_config stream_config{stream};

    fmha_fwd_v3(traits, args, stream_config);

    return {out};
}

} // namespace torch_itfs
} // namespace aiter