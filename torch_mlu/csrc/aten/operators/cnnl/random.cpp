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

#include "ATen/native/DistributionTemplates.h"
#include "ATen/native/UnaryOps.h"

#include "aten/DispatchStub.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/dispatch.h"
namespace torch_mlu {
namespace ops {

using at::native::random_from_to_stub;
using at::native::random_full_64_bits_range_stub;
using at::native::random_stub;

at::Tensor& cnnl_random_(at::Tensor& self, c10::optional<at::Generator> gen) {
  return at::native::random_(self, gen);
}

at::Tensor& cnnl_random_(
    at::Tensor& self,
    int64_t from,
    c10::optional<int64_t> to,
    c10::optional<at::Generator> gen) {
  return at::native::random_(self, from, to, gen);
}

at::Tensor& cnnl_random_(
    at::Tensor& self,
    int64_t to,
    c10::optional<at::Generator> gen) {
  return at::native::random_(self, to, gen);
}

void modify_random_range(int64_t& from_value, int64_t& to_value) {
  TORCH_CHECK(
      to_value >= from_value,
      "to_value should be not less than from_value, but to_value is ",
      to_value,
      " from_value is ",
      from_value);

  // range should be [-2^47, 2^47-1] INT48 limit
  if (from_value < -140737488355328) {
    from_value = -140737488355328;
  } else if (from_value >= 140737488355328) {
    from_value = 140737488355328 - 1;
  }

  if (to_value >= 140737488355328) {
    to_value = 140737488355328 - 1;
  } else if (to_value < -140737488355328) {
    to_value = -140737488355328;
  }
}

/**
 * Samples a discrete uniform distribution in the range [base, base+range) of
 * type T
 */
void random_from_to_kernel(
    at::TensorIteratorBase& iter,
    uint64_t range,
    int64_t base,
    c10::optional<at::Generator> gen) {
  auto self = iter.output(0);
  int64_t to_value = base + static_cast<int64_t>(range);
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  if (self.scalar_type() == c10::ScalarType::Long) {
    modify_random_range(base, to_value);
  }
  AT_DISPATCH_MLU_FLOAT_HALF_INT_BOOL_AND_BFLOAT16(
      self.scalar_type(), "random_", [&] {
        cnnl_random_internal(self_contiguous, to_value - base, base, gen);
      });
  // output is not contiguous
  if (!self.is_same(self_contiguous)) {
    self.copy_(self_contiguous);
  }
}

/**
 * Samples a discrete uniform distribution in the range [min_value(int64_t),
 * max_value(int64_t)]
 */
void random_full_64_bits_range_kernel(
    at::TensorIteratorBase& iter,
    c10::optional<at::Generator> gen) {
  auto self = iter.output(0);
  int64_t from_value = std::numeric_limits<int64_t>::min();
  int64_t to_value = std::numeric_limits<int64_t>::max();
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  if (self.scalar_type() == c10::ScalarType::Long) {
    modify_random_range(from_value, to_value);
  }
  AT_DISPATCH_MLU_FLOAT_HALF_INT_BOOL_AND_BFLOAT16(
      self.scalar_type(), "random_", [&] {
        cnnl_random_internal(
            self_contiguous, to_value - from_value, from_value, gen);
      });
  // output is not contiguous
  if (!self.is_same(self_contiguous)) {
    self.copy_(self_contiguous);
  }
}

/**
 * Samples a discrete uniform distribution in the range [0, max_value(T)] for
 * integral types and [0, 2^mantissa] for floating-point types.
 */
void random_kernel(
    at::TensorIteratorBase& iter,
    c10::optional<at::Generator> gen) {
  auto self = iter.output(0);
  int64_t from_value = 0;
  int64_t to_value = 0;
  if (c10::isFloatingType(self.scalar_type())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "random_calc",
        [&] {
          constexpr int64_t scalar_t_max = static_cast<int64_t>(1)
              << std::numeric_limits<scalar_t>::digits;
          to_value = scalar_t_max > std::numeric_limits<int64_t>::max()
              ? std::numeric_limits<int64_t>::max()
              : static_cast<int64_t>(scalar_t_max);
        });
  } else if (c10::isIntegralType(self.scalar_type(), true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(
        at::ScalarType::Bool, self.scalar_type(), "random_calc", [&] {
          if (std::is_same<scalar_t, bool>::value) {
            to_value = static_cast<int64_t>(true);
          } else {
            to_value =
                static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
          }
        });
  } else {
    TORCH_MLU_CHECK(
        false,
        "random_from_to_impl handles only integral, "
        "floating-point and boolean types");
  }

  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  if (self.scalar_type() == c10::ScalarType::Long) {
    modify_random_range(from_value, to_value);
  }
  AT_DISPATCH_MLU_FLOAT_HALF_INT_BOOL_AND_BFLOAT16(
      self.scalar_type(), "random_", [&] {
        cnnl_random_internal(
            self_contiguous, to_value - from_value, from_value, gen);
      });
  // output is not contiguous
  if (!self.is_same(self_contiguous)) {
    self.copy_(self_contiguous);
  }
}

REGISTER_PRIVATEUSE1_DISPATCH(random_from_to_stub, &random_from_to_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(random_stub, &random_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    random_full_64_bits_range_stub,
    &random_full_64_bits_range_kernel);

} // namespace ops
} // namespace torch_mlu
