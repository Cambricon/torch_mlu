/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2024, the respective contributors
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

#include "ATen/Generator.h"
#include "ATen/Utils.h"

namespace torch_mlu {
namespace ops {

void cnnl_rrelu_with_noise_internal(
    at::Tensor& output,
    const at::Tensor& noise,
    const at::Tensor& self,
    std::optional<at::Generator> gen,
    const float lower,
    const float upper) {
  if (self.numel() == 0)
    return;

  auto self_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);
  auto noise_impl = getMluTensorImpl(noise);

  auto self_ptr = mlu_data_ptr(self_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  auto noise_ptr = mlu_data_ptr(noise_impl);

  auto memory_format = self.suggest_memory_format();
  auto layout = suggestCnnlLayout(memory_format);
  auto self_desc = getTensorDesc(self_impl, layout);
  auto output_desc = getTensorDesc(output_impl, layout);
  auto noise_desc = getTensorDesc(noise_impl, layout);

  // get current handle
  auto handle = getCurrentHandle();

  auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(
      gen, getDefaultMLUGenerator());
  const int64_t nelem = self.numel();
  PhiloxMLUState rng_engine_inputs;
  int thread_num = 0;
  TORCH_CNNL_CHECK(cnnlRandGetSimulateThreadNum(handle, &thread_num));
  auto counter_offset = calc_counter_offset(nelem, (int64_t)thread_num);
  {
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    rng_engine_inputs = gen_impl->philox_mlu_state(counter_offset);
  }

  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetRreluWithNoiseWorkspaceSize(
      handle,
      self_desc.get(),
      output_desc.get(),
      noise_desc.get(),
      &space_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);

  if (rng_engine_inputs.captured_) {
    TORCH_CNNL_CHECK(cnnlRreluWithNoise(
        handle,
        self_desc.get(),
        self_ptr,
        true,
        0,
        0,
        rng_engine_inputs.seed_.ptr,
        rng_engine_inputs.offset_.ptr,
        rng_engine_inputs.offset_intragraph_,
        lower,
        upper,
        workspace_ptr.get(),
        space_size,
        output_desc.get(),
        output_ptr,
        noise_desc.get(),
        noise_ptr));
  } else {
    TORCH_CNNL_CHECK(cnnlRreluWithNoise(
        handle,
        self_desc.get(),
        self_ptr,
        false,
        rng_engine_inputs.seed_.val,
        rng_engine_inputs.offset_.val,
        nullptr,
        nullptr,
        0,
        lower,
        upper,
        workspace_ptr.get(),
        space_size,
        output_desc.get(),
        output_ptr,
        noise_desc.get(),
        noise_ptr));
  }
}

} // namespace ops
} // namespace torch_mlu
