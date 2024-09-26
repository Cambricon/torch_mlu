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
#include "aten/utils/dispatch.h"
#include "framework/generator/generator_impl.h"

namespace torch_mlu {
namespace ops {

void cnnl_exponential_internal(
    at::Tensor& output,
    double lambda,
    c10::optional<at::Generator> generator) {
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);
  auto* output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();

  // get current handle
  auto handle = getCurrentHandle();

  // The type of random numbers generation.
  // The series of MLU200 use hardware to generate the real random numbers,
  // while the series of MLU300 use the algorithm to generate pseudo random
  // numbers.
  auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(
      generator, getDefaultMLUGenerator());
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

  if (rng_engine_inputs.captured_) {
    // TODO(PYTORCH-9647): support bf16 and use accumulation type like cuda
    AT_DISPATCH_MLU_FLOATING_TYPES_HALF_AND_BFLOAT16(
        output.scalar_type(), "MLU exponential", [&] {
          auto cnnl_lambda = static_cast<scalar_t>(lambda);
          TORCH_CNNL_CHECK(cnnlGenerateRandExponential(
              handle,
              rng_type,
              &cnnl_lambda,
              true,
              0,
              0,
              rng_engine_inputs.seed_.ptr,
              rng_engine_inputs.offset_.ptr,
              rng_engine_inputs.offset_intragraph_,
              output_desc.desc(),
              output_ptr));
        });
  } else {
    AT_DISPATCH_MLU_FLOATING_TYPES_HALF_AND_BFLOAT16(
        output.scalar_type(), "MLU exponential", [&] {
          auto cnnl_lambda = static_cast<scalar_t>(lambda);
          TORCH_CNNL_CHECK(cnnlGenerateRandExponential(
              handle,
              rng_type,
              &cnnl_lambda,
              false,
              rng_engine_inputs.seed_.val,
              rng_engine_inputs.offset_.val,
              nullptr,
              nullptr,
              0,
              output_desc.desc(),
              output_ptr));
        });
  }
}

} // namespace ops
} // namespace torch_mlu
