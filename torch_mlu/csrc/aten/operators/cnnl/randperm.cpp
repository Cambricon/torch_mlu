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
#include <ATen/core/Generator.h>

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_randperm_out(
    int64_t n,
    c10::optional<at::Generator> generator,
    at::Tensor& out) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  at::native::check_supported_max_int_with_precision(n, out);

  // cnnl is not good at do this kind of operation,
  // So we use cpu to do this.
  // TODO(liangyuefeng): use at::empty_pinned()
  auto out_cpu = at::empty({n}, out.options().device(at::Device::Type::CPU));
  // TODO(PYTORCH-9634): after this jira completed, replace with cnnl kernel.
  auto generator_cpu = at::make_generator<at::CPUGeneratorImpl>();
  at::randperm_out(out_cpu, n, generator_cpu);

  return out.resize_as_(out_cpu).copy_(out_cpu);
}

} // namespace ops
} // namespace torch_mlu
