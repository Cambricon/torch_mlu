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

#pragma once

#include <array>
#include <vector>
#include <type_traits>
#include <sys/time.h>
#include "utils/cnlog.h"
#include "aten/operators/bang/internal/block_info_container.h"

#define BANGC_PARAM_CHECK(api, condition, ...)                                \
  if (condition == false) {                                                   \
    CNLOG(FATAL) << #api << " check failed, and error message: " __VA_ARGS__; \
  }

class Timer {
 public:
  __mlu_func__ Timer(const char* name) {
    name_ = name;
    start_ = getTimerTick();
  }

  __mlu_func__ void print_time(const int size) {
    long end_ = getTimerTick();
    const long duration = end_ - start_;
    printf(
        "\n name: %s, taskId: %d, handle size: %d, time: %ld us.",
        name_,
        taskId,
        size,
        duration);
  }
  __mlu_func__ void print_time_with_multi(const int size) {
    long end_ = getTimerTick();
    const double duration = (end_ - start_) * 1.0 / size;
    printf(
        "\n name: %s, taskId: %d, multi times: %d, time: %f us.",
        name_,
        taskId,
        size,
        duration);
  }

 private:
  __mlu_func__ long getTimerTick() {
    __asm__ volatile("sync.all;\n\t");
    gettimeofday(&tv_, NULL);
    return tv_.tv_sec * 1000000 + tv_.tv_usec;
  }

 private:
  long start_;
  struct timeval tv_;
  const char* name_;
};

/**
 * Note [CNRTTypeValueToBangcCppType]
 *
 * CNRTTypeValueToBangcCppType is to convert cnrt type value to bangc type.
 * And now only support float, half and bfloat16_t.
 *
 * MLUOpMathType:
 * @brief get cnrt type value and return bangc type
 * @param data_type: cnrt type value
 * @return bangc cpp type
 *
 * Usage:
 * using new_type = CNRTTypeValueToBangcCppType_t<ori_type>;
 *
 */

template <cnrtDataType_V2_t value>
struct CNRTTypeValueToBangcCppType {};

template <>
struct CNRTTypeValueToBangcCppType<cnrtDataType_V2_t::cnrtFloat> {
  using type = float;
};

template <>
struct CNRTTypeValueToBangcCppType<cnrtDataType_V2_t::cnrtBfloat> {
  using type = bfloat16_t;
};

template <>
struct CNRTTypeValueToBangcCppType<cnrtDataType_V2_t::cnrtHalf> {
  using type = half;
};

template <cnrtDataType_V2_t value>
using CNRTTypeValueToBangcCppType_t =
    typename CNRTTypeValueToBangcCppType<value>::type;

namespace torch_mlu::bangcommon {

/**
 * Note [multi tensor data apply]
 * Apply kernel function to data stored in BlockInfoContainer, and those
 * data is splited from input tensorlists.
 * Based on block_size, compute each tensor repeat num and remain num.
 *
 * BlockInfoContainer details: Note [BlockInfoContainer]
 *
 */
template <
    int maxBlockNum,
    int depth,
    typename tupleTypeList,
    template <typename, int, int>
    typename Functor,
    typename... ARGS>
void multi_tensor_apply(
    const std::vector<std::array<void*, depth>>& data_ptr_list,
    const std::vector<int64_t>& sizes,
    const int& block_size,
    cnrtQueue_t stream,
    cnrtFunctionType_t k_type,
    cnrtDim3_t k_dim,
    Functor<tupleTypeList, maxBlockNum, depth>&& func,
    ARGS&&... args) {
  const int tensors_num = data_ptr_list.size();
  BANGC_PARAM_CHECK(
      "multi_tensor_apply",
      tensors_num == sizes.size(),
      "TensorList and sizes num need be equal.");
  // More details in [BlockInfoContainer]
  BlockInfoContainer<maxBlockNum, depth> container;
  container.block_size = block_size;
  int block_index = 0;
  for (int i = 0; i < tensors_num; ++i) {
    const int num_elements = sizes[i];
    const int repeat_num = (num_elements + block_size - 1) / block_size;
    const int remain_element_num = num_elements % block_size;
    container.repeat_block_num[block_index] = block_index == 0
        ? repeat_num
        : repeat_num + container.repeat_block_num[block_index - 1];
    container.remainder_num[block_index] = remain_element_num;
    for (int j = 0; j < depth; ++j) {
      container.address_array[block_index][j] = data_ptr_list[i][j];
    }
    ++block_index;
    if ((block_index == maxBlockNum) || (i == (tensors_num - 1))) {
      container.total_tensor_num = block_index;
      func.call<<<k_dim, k_type, stream>>>(
          container, std::forward<ARGS>(args)...);
      block_index = 0;
    }
  }
}

// Add scalar tensor list for multi_tensor_apply function.
// Adam / Adamw each grad have different step tensor, so need
// scalar tensor list to take step tensor scalar.
template <
    int maxBlockNum,
    int depth,
    typename tupleTypeList,
    typename Functor,
    typename... ARGS>
void multi_tensor_apply_with_scalar_tensor(
    const std::vector<std::array<void*, depth>>& data_ptr_list,
    const std::vector<int64_t>& sizes,
    const std::vector<void*>& steps_ptr_list,
    const int& block_size,
    cnrtQueue_t stream,
    cnrtFunctionType_t k_type,
    cnrtDim3_t k_dim,
    Functor&& func,
    ARGS&&... args) {
  const int tensors_num = data_ptr_list.size();
  BANGC_PARAM_CHECK(
      "multi_tensor_apply",
      tensors_num == sizes.size(),
      "TensorList and sizes num need be equal.");
  BANGC_PARAM_CHECK(
      "multi_tensor_apply",
      tensors_num == steps_ptr_list.size(),
      "TensorList and step tensorlist num need be equal.");
  // More details in [BlockInfoContainer]
  BlockInfoWithTensorScalarList<maxBlockNum, depth> container;
  BlockInfoContainer<maxBlockNum, depth>& inner_container =
      container.block_info_container;
  inner_container.block_size = block_size;
  int block_index = 0;
  for (int i = 0; i < tensors_num; ++i) {
    const int num_elements = sizes[i];
    const int repeat_num = (num_elements + block_size - 1) / block_size;
    const int remain_element_num = num_elements % block_size;
    inner_container.repeat_block_num[block_index] = block_index == 0
        ? repeat_num
        : repeat_num + inner_container.repeat_block_num[block_index - 1];
    inner_container.remainder_num[block_index] = remain_element_num;
    for (int j = 0; j < depth; ++j) {
      inner_container.address_array[block_index][j] = data_ptr_list[i][j];
    }
    container.scalar_tensor_list[block_index] = steps_ptr_list[i];
    ++block_index;
    if ((block_index == maxBlockNum) || (i == tensors_num - 1)) {
      inner_container.total_tensor_num = block_index;
      func.call<<<k_dim, k_type, stream>>>(
          container, std::forward<ARGS>(args)...);
      block_index = 0;
    }
  }
}

// Add tensor index list for multi_tensor_apply function.
// Some ops need compute result for each tensor in tensor list,
// so need to take tensor index to locate output position.
template <
    int maxBlockNum,
    int depth,
    typename tupleTypeList,
    typename Functor,
    typename... ARGS>
void multi_tensor_apply_with_tensor_index(
    const std::vector<std::array<void*, depth>>& data_ptr_list,
    const std::vector<int64_t>& tensor_size_list,
    const std::vector<int>& tensor_index_list,
    const int& block_size,
    cnrtQueue_t stream,
    cnrtFunctionType_t k_type,
    cnrtDim3_t k_dim,
    Functor&& func,
    ARGS&&... args) {
  const int tensors_num = data_ptr_list.size();
  BANGC_PARAM_CHECK(
      "multi_tensor_apply_with_tensor_index",
      tensors_num == tensor_size_list.size(),
      "TensorList and sizes num need be equal.");
  BANGC_PARAM_CHECK(
      "multi_tensor_apply_with_tensor_index",
      tensors_num == tensor_index_list.size(),
      "TensorList and tensor index num need be equal.");
  // More details in [BlockInfoContainer]
  torch_mlu::bangcommon::BlockInfoWithIndexList<maxBlockNum, depth> container;
  auto& inner_container = container.block_info_container;
  inner_container.block_size = block_size;
  int block_index = 0;
  for (int i = 0; i < tensors_num; ++i) {
    const int num_elements = tensor_size_list[i];
    const int repeat_num = (num_elements + block_size - 1) / block_size;
    const int remain_element_num = num_elements % block_size;
    inner_container.repeat_block_num[block_index] = block_index == 0
        ? repeat_num
        : repeat_num + inner_container.repeat_block_num[block_index - 1];
    inner_container.remainder_num[block_index] = remain_element_num;
    for (int j = 0; j < depth; ++j) {
      inner_container.address_array[block_index][j] = data_ptr_list[i][j];
    }
    container.index_list[block_index] = tensor_index_list[i];
    ++block_index;
    if ((block_index == maxBlockNum) || (i == (tensors_num - 1))) {
      inner_container.total_tensor_num = block_index;
      func.call<<<k_dim, k_type, stream>>>(
          container, std::forward<ARGS>(args)...);
      block_index = 0;
    }
  }
}

} // namespace torch_mlu::bangcommon
