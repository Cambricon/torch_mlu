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

#include "c10/core/ScalarType.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/internal_util.h"
#include "framework/core/memory_allocator.h"

namespace torch_mlu::ops {

template <
    int inputNum,
    int outputNum = 1,
    bool isInplace = false,
    typename math_t = float,
    std::enable_if_t<!std::is_same_v<math_t, void>, int> = 1>
class ForeachOPTensorScalarHandle {
 public:
  ForeachOPTensorScalarHandle(
      const std::vector<at::TensorList>& input_output,
      const at::ArrayRef<at::Scalar>& scalar_list)
      : tensor_num_(0) {
    constexpr int total_num = inputNum + outputNum;
    TORCH_CHECK(
        total_num == input_output.size(),
        "TensorList size is not equal to input and output sum num.");
    const int total_tensor_num = input_output[0].size();
    // init desc container
    this->tensor_desc_.reserve(total_tensor_num);
    this->tensor_desc_unique_ptr_.reserve(total_tensor_num);
    // init scalar list container by using self nums.
    const int scalar_list_num = scalar_list.size();
    const bool include_scalar_list = scalar_list_num == 0 ? false : true;
    if (include_scalar_list == true) {
      this->scalar_list_.reserve(scalar_list_num);
    }

// init input output ptr container
#pragma unroll
    for (int i = 0; i < total_num; ++i) {
      if (isInplace == true && i == inputNum)
        continue;
      input_output_ptr_[i].reserve(input_output[i].size());
    }
    auto scalar_type = input_output[0][0].scalar_type();
    auto cnnl_dtype = getCnnlDataType(scalar_type);
    // auto cnnl_dtype = getCnnlDataType(input_output[0][0].scalar_type());
    // store real data in each container.
    for (int i = 0; i < total_tensor_num; ++i) {
      const auto& tensor = input_output[0][i];
      // Ignore empty tensor.
      if (tensor.numel() == 0)
        continue;
      auto tensor_impl = getMluTensorImpl(tensor);
      this->tensor_desc_unique_ptr_.emplace_back(getTensorDesc(
          {tensor_impl->numel()}, {1}, cnnl_dtype, CNNL_LAYOUT_ARRAY));
      this->tensor_desc_.emplace_back(
          tensor_desc_unique_ptr_[tensor_num_].get());
#pragma unroll
      for (int j = 0; j < total_num; ++j) {
        // Using first input tensor info as inplace output tensor info.
        if (isInplace == true && j == inputNum)
          continue;
        // Ignore empty inputs tensor list.
        if (input_output[j].size() == 0)
          continue;
        this->input_output_ptr_[j].emplace_back(
            mlu_data_ptr(getMluTensorImpl(input_output[j][i])));
      }
      if (include_scalar_list == true) {
        this->scalar_list_.emplace_back(scalar_list[i].to<math_t>());
      }
      ++this->tensor_num_;
    }
  }

  inline int64_t get_tensor_num() {
    return this->tensor_num_;
  }

  template <int index>
  std::pair<cnnlTensorDescriptor_t*, void**> get_input_tensor_desc_and_ptr() {
    static_assert(
        index < inputNum,
        "Input index is greater than InputNume in ForeachOPTensorScalarHandle");
    if (this->input_output_ptr_[index].size() == 0) {
      return {nullptr, nullptr};
    }
    return {this->tensor_desc_.data(), this->input_output_ptr_[index].data()};
  }

  template <int index>
  std::pair<cnnlTensorDescriptor_t*, void**> get_output_tensor_desc_and_ptr() {
    static_assert(
        index < outputNum,
        "Output index is greater than outputNum in ForeachOPTensorScalarHandle");
    if constexpr (isInplace == true && index == 0) {
      return {this->tensor_desc_.data(), this->input_output_ptr_[0].data()};
    }
    return {
        this->tensor_desc_.data(),
        this->input_output_ptr_[index + inputNum].data()};
  }

  inline math_t* get_scalar_list_ptr() {
    return this->scalar_list_.size() == 0 ? nullptr : this->scalar_list_.data();
  }

 private:
  // This is used for keep desc ptr alive.
  std::vector<tensorDescPtr_t> tensor_desc_unique_ptr_;
  std::vector<cnnlTensorDescriptor_t> tensor_desc_;
  std::array<std::vector<void*>, inputNum + outputNum> input_output_ptr_;
  c10::SmallVector<math_t, 10> scalar_list_;
  int64_t tensor_num_ = 0;
};

} // namespace torch_mlu::ops
