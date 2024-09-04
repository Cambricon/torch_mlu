/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2023, the respective contributors
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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include <ATen/native/Resize.h>

namespace torch_mlu {
namespace ops {

// Currently, double max and min which are greater than
// torch.finfo(torch.float).max are not supported due to the limitation of
// cnnlHistc. Similarly, Long max and min which are greater than
// torch.iinfo(torch.int).max or less than torch.iinfo(torch.int).min are not
// supported as well.

at::Tensor cnnl_histc(
    const at::Tensor& self,
    int64_t bins,
    const at::Scalar& min,
    const at::Scalar& max) {
  if (self.scalar_type() == at::ScalarType::Half) {
    AT_ERROR("HalfTensor is not supported");
  }
  auto self_contiguous = cnnl_contiguous(self);
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "cnnl_histc", [&] {
    using bounds_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
    // param check
    if (bins <= 0) {
      TORCH_CHECK(false, "bins must be > 0");
    }
    at::Tensor output = at::native::zeros(
        {bins},
        c10::nullopt,
        self_contiguous.scalar_type(),
        c10::nullopt /* layout */,
        at::DeviceType::PrivateUse1,
        c10::nullopt /* pin_memory */);
    scalar_t minvalue = min.to<bounds_t>();
    scalar_t maxvalue = max.to<bounds_t>();
    if (minvalue == maxvalue && self_contiguous.numel() > 0) {
      minvalue = *self_contiguous.min().cpu().data_ptr<scalar_t>();
      maxvalue = *self_contiguous.max().cpu().data_ptr<scalar_t>();
    }
    if (minvalue == maxvalue) {
      minvalue = minvalue - 1;
      maxvalue = maxvalue + 1;
    }
    TORCH_CHECK(
        !(std::isinf(minvalue) || std::isinf(maxvalue) ||
          std::isnan(minvalue) || std::isnan(maxvalue)),
        "range of [",
        minvalue,
        ", ",
        maxvalue,
        "] is not finite");
    TORCH_CHECK(minvalue < maxvalue, "max must be larger than min");
    auto out = create_int_tensor_if_needed(output);
    self_contiguous = cast_long_to_int_if_needed(self_contiguous);
    cnnl_histc_internal(self_contiguous, bins, minvalue, maxvalue, out);
    cast_int_to_long_if_needed(out, output);
    return output;
  });
}

at::Tensor& cnnl_histc_out(
    const at::Tensor& self,
    int64_t bins,
    const at::Scalar& min,
    const at::Scalar& max,
    at::Tensor& result) {
  TORCH_CHECK(
      self.dtype() == result.dtype(),
      "torch.histogram: input tensor and hist tensor should",
      " have the same dtype, but got input ",
      self.dtype(),
      " and hist ",
      result.dtype());
  auto ret = cnnl_histc(self, bins, min, max);
  at::native::resize_output(result, ret.sizes());
  result.copy_(ret);
  return result;
}
} // namespace ops
} // namespace torch_mlu
