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

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {
void cnnl_erfc_internal(at::Tensor& output, const at::Tensor& self) {
  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = self_impl->mlu_data_ptr();
  CnnlTensorDescriptor desc_self;
  desc_self.set(self, CNNL_LAYOUT_ARRAY);

  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output, CNNL_LAYOUT_ARRAY);

  cnnlActivationMode_t mode = CNNL_ACTIVATION_ERFC;
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  cnnlNanPropagation_t nan_prop = CNNL_PROPAGATE_NAN;

  // set activation info.
  CnnlActivationDescriptor desc_activation;
  desc_activation.set(
      /*cnnlActivationMode_t       */ mode,
      /*cnnlActivationPreference_t */ prefer,
      /*cnnlNanPropagation_t       */ nan_prop,
      /*coef                       */ 1.0,
      /*sliced_dim                 */ -1,
      /*gamma                      */ 1.0,
      /*scale                      */ 1.0,
      /*is_result                  */ false,
      /*approximate                */ false);

  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlActivationForward(
      /*handle                     */ handle,
      /*cnnlActivationDescriptor_t */ desc_activation.desc(),
      /*alpha                      */ NULL,
      /*cnnlTensorDescriptor_t     */ desc_self.desc(),
      /*x_ptr                      */ self_ptr,
      /*beta                       */ NULL,
      /*cnnlTensorDescriptor_t     */ output_desc.desc(),
      /*y_ptr                      */ output_ptr));
}
} // namespace ops
} // namespace torch_mlu
