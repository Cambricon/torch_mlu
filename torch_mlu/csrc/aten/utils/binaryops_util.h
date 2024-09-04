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

#pragma once

#include <ATen/native/TypeProperties.h>
#include <algorithm>
#include "ATen/ScalarOps.h"

namespace torch_mlu {

using ScalarTypeVec = std::vector<at::ScalarType>;

extern std::vector<at::ScalarType> type_vec;

inline at::Tensor convertTensorType(
    const at::Tensor& input,
    at::ScalarType type) {
  return type == input.scalar_type() ? input : input.to(type);
}

// Create mlu scalar tensor based on cpu scalar tensor and specified type.
at::Tensor scalar_to_tensor_with_dtype(
    const at::Tensor& tensor,
    at::ScalarType ct_type = at::ScalarType::Undefined);

inline bool isTransformComputeType(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::kFloat ||
      tensor.scalar_type() == at::kHalf;
}

inline bool isCpuScalar(const at::Tensor& tensor) {
  return tensor.numel() == 1 && !tensor.device().is_privateuseone();
}

// wrap cpu scalar to tensor. copy from pytorch .cpp code.
at::Tensor wrapped_scalar_tensor(const at::Scalar& scalar);

} // namespace torch_mlu
