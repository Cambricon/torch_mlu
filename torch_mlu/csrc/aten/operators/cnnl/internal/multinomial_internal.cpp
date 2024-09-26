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
#include "aten/operators/cnnl/internal/philox_utils.h"
#include "framework/generator/generator_impl.h"

namespace torch_mlu {
namespace ops {

void cnnl_multinomial_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    std::optional<at::Generator> gen) {
  // get tensor impl
  auto* input_impl = getMluTensorImpl(self);
  auto desc_input = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto* output_impl = getMluTensorImpl(output);
  auto desc_output = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetRandGenerateMultinomialWorkspaceSize(
      handle, desc_input.get(), &workspace_size));

  // get mlu ptr
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(
      gen, getDefaultMLUGenerator());
  const int64_t nelem = output.numel();
  cnnlRandRngType_t rng_type = CNNL_RAND_RNG_PHILOX;
  PhiloxMLUState rng_engine_inputs;
  int thread_num = 0;
  TORCH_CNNL_CHECK(
      cnnlGetRandSimulateThreadNum_v2(handle, rng_type, &thread_num));
  auto counter_offset = calc_counter_offset(nelem, (int64_t)thread_num);
  {
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    rng_engine_inputs = gen_impl->philox_mlu_state(counter_offset);
  }

  // The rows of input do not need to sum to one (in which case we use the
  // values as weights), but must be non-negative, finite and have a non-zero
  // sum. So set is_logits false.
  bool is_logits = false;

  if (rng_engine_inputs.captured_) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "MLU multinomial",
        [&] {
          TORCH_CNNL_CHECK(cnnlGenerateRandMultinomial(
              handle,
              rng_type,
              desc_input.get(),
              input_ptr,
              true,
              0,
              0,
              rng_engine_inputs.seed_.ptr,
              rng_engine_inputs.offset_.ptr,
              rng_engine_inputs.offset_intragraph_,
              replacement,
              is_logits,
              workspace_ptr.get(),
              workspace_size,
              desc_output.get(),
              output_ptr));
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "MLU multinomial",
        [&] {
          TORCH_CNNL_CHECK(cnnlGenerateRandMultinomial(
              handle,
              rng_type,
              desc_input.get(),
              input_ptr,
              false,
              rng_engine_inputs.seed_.val,
              rng_engine_inputs.offset_.val,
              nullptr,
              nullptr,
              0,
              replacement,
              is_logits,
              workspace_ptr.get(),
              workspace_size,
              desc_output.get(),
              output_ptr));
        });
  }
}

} // namespace ops
} // namespace torch_mlu
