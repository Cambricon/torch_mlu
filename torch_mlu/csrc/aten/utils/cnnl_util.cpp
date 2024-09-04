/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/pytorch/pytorch/graphs/contributors Redistribution and use in
source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <algorithm>
#include "aten/utils/cnnl_util.h"
#include "aten/utils/utils.h"
#include "framework/core/tensor_impl.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/cnnl_kernel.h"

namespace torch_mlu {

cnnlTensorLayout_t suggest_cnnl_layout(const at::Tensor& input) {
  auto suggest_memory_format = input.suggest_memory_format();
  cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
  switch (input.dim()) {
    case 4:
      layout = (suggest_memory_format == at::MemoryFormat::ChannelsLast)
          ? CNNL_LAYOUT_NHWC
          : CNNL_LAYOUT_NCHW;
      break;
    case 5:
      layout = (suggest_memory_format == at::MemoryFormat::ChannelsLast3d)
          ? CNNL_LAYOUT_NDHWC
          : CNNL_LAYOUT_NCDHW;
      break;
    default:
      layout = CNNL_LAYOUT_ARRAY;
  }
  return layout;
}

at::MemoryFormat get_channels_last_memory_format(int64_t dim) {
  TORCH_MLU_CHECK(
      (dim > 3) && (dim < 6),
      "at::MemoryFormat only support rank 4 or 5 tensor with channels_last memory format.");
  at::MemoryFormat memory_format;
  switch (dim) {
    case 4:
      memory_format = at::MemoryFormat::ChannelsLast;
      break;
    case 5:
      memory_format = at::MemoryFormat::ChannelsLast3d;
      break;
  }
  return memory_format;
}

bool pair_first_down(
    std::pair<int64_t, int64_t> pair1,
    std::pair<int64_t, int64_t> pair2) {
  return pair1.first > pair2.first;
}

std::vector<int64_t> get_permute_back_order(const at::Tensor& input) {
  auto input_sizes = input.sizes().vec();
  auto input_strides = input.strides().vec();
  auto ndim = input.dim();
  std::vector<std::pair<int64_t, int64_t>> strides_sizes(
      ndim, std::pair<int64_t, int64_t>(1, 1));
  for (int64_t i = 0; i < ndim; ++i) {
    strides_sizes[i] = std::pair<int64_t, int64_t>(
        (static_cast<int64_t>(input_strides[i])),
        (static_cast<int64_t>(input_sizes[i])));
  }
  sort(strides_sizes.begin(), strides_sizes.end(), pair_first_down);
  std::vector<int64_t> permute_back_order(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    auto pair = strides_sizes[i];
    for (int64_t j = 0; j < ndim; ++j) {
      if ((pair.first == input_strides[j]) && (pair.second == input_sizes[j])) {
        permute_back_order[i] = j;
        input_strides[j] = -1;
        input_sizes[j] = -1;
        break;
      }
    }
  }
  return permute_back_order;
}

inline bool get_overflow_check_env() {
  const char* enable_c_str = std::getenv("ENABLE_MLU_OVERFLOW_CHECK");
  // disable by default
  if (!enable_c_str) {
    return false;
  }
  std::string enable(enable_c_str);
  if (enable == "1" || enable == "ON" || enable == "on") {
    return true;
  }
  return false;
}

// if input's dtype is long, cast input to int and return
at::Tensor cast_long_to_int_if_needed(
    const at::Tensor& input,
    const bool need_check) {
  auto input_tensor = input;
  if (input_tensor.scalar_type() == at::kLong) {
    TORCH_WARN_ONCE("Casting input of dtype int64 to int32, maybe overflow!");
    static const bool enable_overflow_check = get_overflow_check_env();
    if (enable_overflow_check && need_check) {
      int64_t max_value = std::numeric_limits<int>::max();
      int64_t min_value = std::numeric_limits<int>::min();
      bool overflow_compare_result =
          at::any(at::gt(input, max_value)).item().to<bool>();
      bool underflow_compare_result =
          at::any(at::lt(input, min_value)).item().to<bool>();
      bool result = overflow_compare_result || underflow_compare_result;
      if (result) {
        throw std::runtime_error(
            "Overflow occurred! there is a tensor that does cast from int64 value to int32.");
      }
    }
    input_tensor = input_tensor.to(at::kInt);
  }
  return input_tensor;
}

// if output's dtype is long, create an int tenor and return
at::Tensor create_int_tensor_if_needed(const at::Tensor& output) {
  auto output_tensor = output;
  if (output_tensor.scalar_type() == at::kLong) {
    output_tensor =
        at::empty_like(output_tensor, output_tensor.options().dtype(at::kInt));
  }
  return output_tensor;
}

// cast int output to long out
void cast_int_to_long_if_needed(
    const at::Tensor& output,
    const at::Tensor& out) {
  if (out.scalar_type() == at::kLong) {
    out.copy_(output);
  }
}

// 1. input Size([1,1,1,1]), numel = 1, output Size([]), numel = 1.
// 2. input and output have same memory format.
bool is_same_format_tensor(const at::TensorList& tensors) {
  TORCH_MLU_CHECK(
      tensors.size() > 0, "Input tensor num need be greater than 0.");
  bool only_one_element = true;
  for (const auto& t : tensors) {
    if (t.numel() != 1) {
      only_one_element = false;
      break;
    }
  }
  if (only_one_element == true)
    return true;
  const auto& size = tensors[0].sizes();
  const auto& stride = tensors[0].strides();
  for (int i = 1; i < tensors.size(); i++) {
    TORCH_MLU_CHECK(tensors[i].defined(), "Input tensor is not defined.");
  }
  for (const auto& t : tensors) {
    if (t.is_non_overlapping_and_dense() == false) {
      return false;
    }
    if (t.sizes() != size || !is_same_strides(size, stride, t.strides())) {
      return false;
    }
  }
  return true;
}

std::tuple<at::DimVector, at::DimVector> inferSqueezeGeometry(
    const at::Tensor& tensor) {
  at::DimVector sizes;
  at::DimVector strides;

  for (const auto d : c10::irange(tensor.dim())) {
    if (tensor.sizes()[d] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }

  return std::make_tuple(std::move(sizes), std::move(strides));
}

at::Tensor svd_backward(
    const std::vector<torch::autograd::Variable>& grads,
    const at::Tensor& self,
    bool some,
    bool compute_uv,
    const at::Tensor& raw_u,
    const at::Tensor& sigma,
    const at::Tensor& raw_v) {
  TORCH_CHECK(
      compute_uv,
      "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
      "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");

  auto m = self.size(-2);
  auto n = self.size(-1);
  auto k = sigma.size(-1);
  auto gsigma = grads[1];

  auto u = raw_u;
  auto v = raw_v;
  auto gu = grads[0];
  auto gv = grads[2];

  if (!some) {
    // We ignore the free subspace here because possible base vectors cancel
    // each other, e.g., both -v and +v are valid base for a dimension.
    // Don't assume behavior of any particular implementation of svd.
    u = raw_u.narrow(-1, 0, k);
    v = raw_v.narrow(-1, 0, k);
    if (gu.defined()) {
      gu = gu.narrow(-1, 0, k);
    }
    if (gv.defined()) {
      gv = gv.narrow(-1, 0, k);
    }
  }
  auto vh = v.conj().transpose(-2, -1);

  at::Tensor sigma_term;
  if (gsigma.defined()) {
    gsigma = gsigma.to(self.dtype());
    // computes u @ diag(gsigma) @ vh
    sigma_term = at::matmul(u * gsigma.unsqueeze(-2), vh);
  } else {
    sigma_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  // in case that there are no gu and gv, we can avoid the series of kernel
  // calls below
  if (!gv.defined() && !gu.defined()) {
    return sigma_term;
  }

  auto uh = u.conj().transpose(-2, -1);
  auto sigma_inv = sigma.pow(-1);
  auto sigma_sq = sigma.pow(2);
  auto F = sigma_sq.unsqueeze(-2) - sigma_sq.unsqueeze(-1);
  // The following two lines invert values of F, and fills the diagonal with 0s.
  // Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
  // first to prevent nan from appearing in backward of this function.
  F.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(INFINITY);
  F = F.pow(-1);

  at::Tensor u_term, v_term;

  if (gu.defined()) {
    auto guh = gu.conj().transpose(-2, -1);
    u_term = at::matmul(
        u,
        F.mul(at::matmul(uh, gu) - at::matmul(guh, u)) * sigma.unsqueeze(-2));
    if (m > k) {
      // projection operator onto subspace orthogonal to span(U) defined as I -
      // UU^H
      auto proj_on_ortho_u = -at::matmul(u, uh);
      proj_on_ortho_u.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).add_(1);
      u_term = u_term + proj_on_ortho_u.matmul(gu * sigma_inv.unsqueeze(-2));
    }
    u_term = at::matmul(u_term, vh);
  } else {
    u_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (gv.defined()) {
    auto gvh = gv.conj().transpose(-2, -1);
    v_term = sigma.unsqueeze(-1) *
        at::matmul(F.mul(at::matmul(vh, gv) - at::matmul(gvh, v)), vh);
    if (n > k) {
      // projection operator onto subspace orthogonal to span(V) defined as I -
      // VV^H
      auto proj_on_v_ortho = -at::matmul(v, vh);
      proj_on_v_ortho.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).add_(1);
      v_term =
          v_term + sigma_inv.unsqueeze(-1) * at::matmul(gvh, proj_on_v_ortho);
    }
    v_term = at::matmul(u, v_term);
  } else {
    v_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (self.is_complex() && gu.defined()) {
    at::Tensor L = at::matmul(uh, gu).diagonal(0, -2, -1);
    at::real(L).zero_();
    at::imag(L).mul_(sigma_inv);
    at::Tensor imag_term = at::matmul(u * L.unsqueeze(-2), vh);
    return u_term + sigma_term + v_term + imag_term;
  }

  return u_term + sigma_term + v_term;
}

at::Tensor unsqueeze_multiple(
    const at::Tensor& t,
    at::IntArrayRef dim,
    size_t n_dims) {
  auto dims_to_unsqueeze = at::dim_list_to_bitset(dim, n_dims);
  at::Tensor res = t;
  for (size_t i = 0; i < n_dims; i++) {
    if (dims_to_unsqueeze[i]) {
      res = res.unsqueeze(i);
    }
  }
  return res;
}

c10::MemoryFormat switch_tensors_suggest_memory_format(
    const std::vector<at::Tensor>& tensor_list) {
  if (tensor_list.size() == 0) {
    return c10::MemoryFormat::Contiguous;
  }
  std::vector<c10::MemoryFormat> tensors_memory_format;
  auto ndim = tensor_list[0].dim();
  bool all_ndim_same = true;
  for (auto& tensor : tensor_list) {
    tensors_memory_format.push_back(tensor.suggest_memory_format());
    all_ndim_same &= (tensor.dim() == ndim);
  }
  /* 1. If the ndim of all input tensors is same
   *       If tensors_memory_format contains ChannelsLast, return ChannelsLast,
   *       otherwise return Contiguous
   *  2. If the ndim of all input tensors is not same, return Contiguous
   */
  if (all_ndim_same) {
    auto channel_last_3d_it = std::find(
        tensors_memory_format.begin(),
        tensors_memory_format.end(),
        c10::MemoryFormat::ChannelsLast3d);
    if (channel_last_3d_it != tensors_memory_format.end() && ndim == 5) {
      return c10::MemoryFormat::ChannelsLast3d;
    }
    auto channel_last_2d_it = std::find(
        tensors_memory_format.begin(),
        tensors_memory_format.end(),
        c10::MemoryFormat::ChannelsLast);
    if (channel_last_2d_it != tensors_memory_format.end() && ndim == 4) {
      return c10::MemoryFormat::ChannelsLast;
    }
    return c10::MemoryFormat::Contiguous;
  }
  return c10::MemoryFormat::Contiguous;
}
} // namespace torch_mlu
