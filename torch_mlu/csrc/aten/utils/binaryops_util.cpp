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
#include "aten/utils/binaryops_util.h"
#include "aten/utils/dispatch.h"

namespace torch_mlu {

std::vector<at::ScalarType> type_vec = {at::kHalf, at::kFloat};

at::Tensor scalar_to_tensor_with_dtype(
    const at::Tensor& tensor,
    at::ScalarType ct_type) {
  if (tensor.numel() != 1 || tensor.device().is_privateuseone()) {
    return tensor;
  }

  if (ct_type == at::ScalarType::Undefined) {
    ct_type = tensor.scalar_type();
  }
  at::Tensor result;
  at::Scalar scalar = tensor.item();
  AT_DISPATCH_MLU_TENSOR_SCLAER_TYPES(ct_type, "create_mlu_scalar_tensor", [&] {
    result = at::full(
        {1},
        scalar.to<scalar_t>(),
        tensor.options().dtype(ct_type).device(at::kPrivateUse1));
  });
  return result;
}

at::Tensor wrapped_scalar_tensor(const at::Scalar& scalar) {
  auto tensor = c10::scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

} // namespace torch_mlu
