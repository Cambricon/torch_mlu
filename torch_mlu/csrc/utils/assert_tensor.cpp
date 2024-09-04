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

#include "utils/assert_tensor.h"

void assertEqual(float diff, float prec) {
  if (diff != prec) {
    CNLOG(INFO) << std::to_string(diff) << " not equal to "
                << std::to_string(prec);
    TORCH_CHECK(false, "detect error, diff and prec are not equal");
  }
  return;
}

void assertLessEqual(float diff, float prec) {
  if (diff > prec) {
    CNLOG(INFO) << std::to_string(diff) << " not less than or equal to "
                << std::to_string(prec);
    TORCH_CHECK(
        false, "detect error, cpu and mlu results are bigger than threshold");
  }
  return;
}

void assertTensorsEqual(
    const at::Tensor& tensor_one,
    const at::Tensor& tensor_two,
    float prec = 0.0,
    bool use_MSE = false,
    bool use_RAE = false,
    bool use_RMA = false) {
  auto tensor_first = tensor_one.cpu();
  auto tensor_second = tensor_two.cpu();
  if (tensor_first.sizes() != tensor_second.sizes()) {
    TORCH_CHECK(false, "detect error, two tensors sizes are not equal");
    return;
  }
  // in C++, tensors value can't be inf or nan, which exists in Python,
  // so here we don't deal with these two cases.
  double diff_return = 0.0;
  auto diff = tensor_first.sub(tensor_second);
  auto diff_abs = at::abs(diff);
  const float epsilon = 1.0 / 16384;
  if (use_MSE) {
    auto diff_pow = at::pow(diff_abs, 2);
    auto diff_sum = diff_pow.sum();
    auto tensor_pow_sum = at::sum(at::pow(tensor_first, 2));
    if (diff_sum.item().toFloat() <= 4 * epsilon * epsilon) {
      return;
    }
    if (tensor_pow_sum.item().toFloat() <= epsilon) {
      tensor_pow_sum += epsilon;
    }
    diff_return = diff_sum.item().toFloat() / tensor_pow_sum.item().toFloat();
    assertLessEqual(std::sqrt(diff_return), prec);
  } else if (use_RAE) {
    auto diff_sum = at::sum(diff_abs);
    auto tensor_abs_sum = at::sum(at::abs(tensor_first));
    if (tensor_abs_sum.item().toFloat() == 0.0) {
      assertEqual(diff_sum.item().toFloat(), tensor_abs_sum.item().toFloat());
    } else {
      diff_return = diff_sum.item().toFloat() / tensor_abs_sum.item().toFloat();
      assertLessEqual(diff_return, prec);
    }
  } else if (use_RMA) {
    auto a_mean = at::mean(at::abs(tensor_first));
    auto b_mean = at::mean(at::abs(tensor_second));
    if (a_mean.item().toFloat() == 0) {
      assertEqual(a_mean.item().toFloat(), b_mean.item().toFloat());
    } else {
      diff_return = (a_mean.sub(b_mean)).div(a_mean).item().toFloat();
      assertLessEqual(diff_return, prec);
    }
  } else {
    auto max_err = at::max(diff_abs).item().toFloat();
    assertLessEqual(max_err, prec);
  }
  return;
}
