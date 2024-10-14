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

#include <algorithm>
#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

template <typename T, cnrtDataType_V2_t U = cnrtHalf>
T get_bound(at::optional<at::Scalar> input) {
  if (input.has_value())
    return input->to<T>();
  return 0;
}

template <>
uint16_t get_bound<uint16_t, cnrtHalf>(at::optional<at::Scalar> input) {
  if (input.has_value()) {
    auto temp = input->to<float>();
    uint16_t result = 0;
    TORCH_CNRT_CHECK(cnrtCastDataType_V2(
        static_cast<void*>(&temp),
        cnrtFloat,
        static_cast<void*>(&result),
        cnrtHalf,
        1,
        nullptr,
        cnrtRounding_rm));
    return result;
  }
  return 0;
}

template <>
uint16_t get_bound<uint16_t, cnrtBfloat>(at::optional<at::Scalar> input) {
  if (input.has_value()) {
    auto temp = input->to<float>();
    uint16_t result = 0;
    TORCH_CNRT_CHECK(cnrtCastDataType_V2(
        static_cast<void*>(&temp),
        cnrtFloat,
        static_cast<void*>(&result),
        cnrtBfloat,
        1,
        nullptr,
        cnrtRounding_rm));
    return result;
  }
  return 0;
}

template <typename T, cnrtDataType_V2_t U = cnrtHalf>
void clip(
    at::Tensor& output,
    const at::Tensor& self,
    at::optional<at::Scalar> min,
    at::optional<at::Scalar> max) {
  // get bound
  T min_bound = get_bound<T, U>(min);
  T max_bound = get_bound<T, U>(max);
  // get tensor impl
  auto self_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  // create the desc
  auto desc_self = getTensorDesc(self_impl, CNNL_LAYOUT_ARRAY);
  auto desc_output = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  // get current handle
  auto handle = getCurrentHandle();

  // get the mlu ptr
  auto self_ptr = mlu_data_ptr(self_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // compute ops
  cnnlPointerMode_t pointer_mode = CNNL_POINTER_MODE_HOST;
  TORCH_CNNL_CHECK(cnnlClip_v2(
      handle,
      pointer_mode,
      desc_self.get(),
      self_ptr,
      min.has_value() ? static_cast<void*>(&min_bound) : nullptr,
      max.has_value() ? static_cast<void*>(&max_bound) : nullptr,
      desc_output.get(),
      output_ptr));
}

void cnnl_clamp_internal(
    at::Tensor& output,
    const at::Tensor& self,
    at::optional<at::Scalar> min,
    at::optional<at::Scalar> max) {
  TORCH_CHECK(!self.is_complex(), "clamp is not supported for complex types");
  // now mlu tensor only support Layout::Strided
  // python exception test alose can't test, "Unsupported device type for sparse
  // layout: mlu"
  TORCH_CHECK(
      self.layout() == c10::Layout::Strided,
      "clamp only supports strided layout, got: ",
      self.layout());

  std::unordered_map<std::string, int> dtype = {
      {"float", 1},
      {"double", 1},
      {"int", 2},
      {"long int", 6},
      {"half", 3},
      {"c10::Half", 4},
      {"c10::BFloat16", 5}};
  switch (dtype[std::string(self.dtype().name())]) {
    case 1:
      clip<float>(output, self, min, max);
      break;
    case 2:
      clip<int>(output, self, min, max);
      break;
    case 3:
      clip<uint16_t>(output, self, min, max);
      break;
    case 4:
      clip<uint16_t>(output, self, min, max);
      break;
    case 5:
      clip<uint16_t, cnrtBfloat>(output, self, min, max);
      break;
    case 6:
      clip<long>(output, self, min, max);
      break;
    default:
      auto self_cast = self.to(at::kFloat);
      auto output_cast = at::empty_like(self_cast);
      clip<float>(output_cast, self_cast, min, max);
      cnnl_cast_internal(output_cast, output);
  }
}

void cnnl_clamp_tensor_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& min,
    const at::Tensor& max) {
  TORCH_CHECK(!self.is_complex(), "clamp is not supported for complex types");
  // now mlu tensor only support Layout::Strided
  // python exception test alose can't test, "Unsupported device type for sparse
  // layout: mlu"
  TORCH_CHECK(
      self.layout() == c10::Layout::Strided,
      "clamp only supports strided layout, got: ",
      self.layout());
  TORCH_CHECK(
      min.layout() == c10::Layout::Strided,
      "clamp only supports strided layout, got: ",
      min.layout());
  TORCH_CHECK(
      max.layout() == c10::Layout::Strided,
      "clamp only supports strided layout, got: ",
      max.layout());

  // get tensor impl
  auto self_impl = getMluTensorImpl(self);
  auto desc_self = getTensorDesc(self_impl, CNNL_LAYOUT_ARRAY);
  auto self_ptr = mlu_data_ptr(self_impl);

  auto min_impl = getMluTensorImpl(min);
  auto desc_min = getTensorDesc(min_impl, CNNL_LAYOUT_ARRAY);
  auto min_ptr = mlu_data_ptr(min_impl);

  auto max_impl = getMluTensorImpl(max);
  auto desc_max = getTensorDesc(max_impl, CNNL_LAYOUT_ARRAY);
  auto max_ptr = mlu_data_ptr(max_impl);

  auto output_impl = getMluTensorImpl(output);
  auto desc_output = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // get the workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetClipWorkspaceSize(
      handle,
      desc_self.get(),
      desc_min.get(),
      desc_max.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // compute ops
  TORCH_CNNL_CHECK(cnnlClip_v3(
      handle,
      desc_self.get(),
      self_ptr,
      desc_min.get(),
      min_ptr,
      desc_max.get(),
      max_ptr,
      workspace_ptr.get(),
      workspace_size,
      desc_output.get(),
      output_ptr));
}

} // namespace ops
} // namespace torch_mlu
