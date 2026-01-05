#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor> fmha_v3_fwd_ck(const at::Tensor& q, // [b, sq, hq, d]
                                       const at::Tensor& k, // [b, sk, hk, d]
                                       const at::Tensor& v, // [b, sk, hk, d_v]
                                       float softmax_scale,
                                       float logits_soft_cap,
                                       bool is_causal,
                                       std::optional<const at::Tensor> q_descale, // [1]
                                       std::optional<const at::Tensor> k_descale, // [1]
                                       std::optional<const at::Tensor> v_descale  // [1]
);
} // namespace torch_itfs
} // namespace aiter