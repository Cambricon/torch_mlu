/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2023, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
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

# pragma once

#include <iostream>
#include <array>
#include <vector>
#include <tuple>
#include <type_traits>
#include <sys/time.h>
#include "utils/cnlog.h"

#define BANGC_PARAM_CHECK(api, condition, ...)                   \
  if (condition == false) {       \
    CNLOG(FATAL) << #api << " check failed, and error message: " \
                 __VA_ARGS__ ;                                   \
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
      printf("\n name: %s, taskId: %d, handle size: %d, time: %ld us.",
              name_, taskId, size, duration);
    }
    __mlu_func__ void print_time_with_multi(const int size) {
      long end_ = getTimerTick();
      const double duration = (end_ - start_) * 1.0 / size;
      printf("\n name: %s, taskId: %d, multi times: %d, time: %f us.",
              name_, taskId, size, duration);
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

template<cnrtDataType_V2_t value>
struct CNRTTypeValueToBangcCppType {};

template<>
struct CNRTTypeValueToBangcCppType<cnrtDataType_V2_t::cnrtFloat> {
  using type = float;
};

template<>
struct CNRTTypeValueToBangcCppType<cnrtDataType_V2_t::cnrtBfloat> {
  using type = bfloat16_t;
};

template<>
struct CNRTTypeValueToBangcCppType<cnrtDataType_V2_t::cnrtHalf> {
  using type = half;
};

template<cnrtDataType_V2_t value>
using CNRTTypeValueToBangcCppType_t = typename CNRTTypeValueToBangcCppType<value>::type;

namespace torch_mlu::bangcommon {

// Kernel launch support host data size is 24 * 1024 Bytes, now
// just using 8KB for host data when kernel launch.
// (depth * sizeof(BlockInfo) + sizeof(int)) * 2 <= 8 * 1024
// Each number needs to be divisible by 2 without remainder.
static constexpr int depth_to_max_blockinfo[5] = {510, 320, 252, 200, 168};

/**
 * Note [BlockInfoContainer]
 * BlockInfoContainer is used to store the block info of tensors, which is used in
 * for kernel launch and compute.
 * Each kernel can use block info to load, compute and store data.
 * address_array is used to store the address of tensors. First half part is to stored
 * multi nram block part, and the second half part is to stored the left part of tensors.
 *
 * block_repeat_num is used to record the accumulate repeat times of blocks.
 * each_segment_element_num is to stored the segment block, and which is just store the
 * segment size without accumulate with other segment.
 * block_repeat_num and each_segment_element_num are combined to
 * block_repeat_and_segment_element_num, which first part is block_repeat_num, other part
 * is each_segment_element_num.
 *
 * block_num is mean the part of tensors which size is multi-time of nram block;
 * segment_num is to record the left part of tensors which size is not multi-time
 * of nram block.
 *
 * each_block_element_size is using to record the element num of each time compute.
 */
template<int maxBlockNum, int depth>
struct BlockInfoContainer {
  static_assert(
    maxBlockNum * (sizeof(void*) * depth + sizeof(int))
    + 3 * sizeof(int) < 8 * 1024, "Host launch data size is over 8 * 1024.");
  void* address_array[maxBlockNum][depth];
  // Using repeat num to avoid using int64 on device side.
  int block_repeat_and_segment_element_num[maxBlockNum];
  int block_num = 0;
  int segment_num = 0;
  int each_block_element_size = 0;
};

template<int maxBlockNum, int depth>
struct BlockInfoWithTensorScalarList {
  static_assert(sizeof(BlockInfoContainer<maxBlockNum, depth>)
    + sizeof(void*) * maxBlockNum < 9 * 1024, "Host launch data size is over 9 * 1024.");
  BlockInfoContainer<maxBlockNum, depth> block_info_container;
  void* scalar_tensor_list[maxBlockNum];
};

// host print for BlockInfoContainer
template<int maxBlockNum, int depth>
std::ostream& operator<<(std::ostream& os, const BlockInfoContainer<maxBlockNum, depth>& other) {
  os << "\nmaxBlockNum: " << maxBlockNum << " depth: " << depth;
  os << "\nBlockInfoContainer block_num: " << other.block_num << " ";
  for (int i = 0; i < other.block_num; ++i) {
    const int repeat_num = i == 0 ? other.block_repeat_and_segment_element_num[i]
      : other.block_repeat_and_segment_element_num[i] - other.block_repeat_and_segment_element_num[i - 1];
    const int block_size = repeat_num * other.each_block_element_size;
    os << "\nNum size: " << block_size << " current block_repeat_num: "
       << other.block_repeat_and_segment_element_num[i] << " block_repeat_num: " << repeat_num
       << " each_block_element_size: " << other.each_block_element_size << " Address info: ";
    for (int j = 0; j < depth; ++j) {
      os << other.address_array[i][j] << " ";
    }
  }
  os << "\nBlockInfoContainer segment_num: " << other.segment_num
     << " BlockInfoContainer segment repeat num: " << other.segment_num << " ";
  for (int i = 0; i < other.segment_num; ++i) {
    const int index = i + maxBlockNum / 2;
    os << "\nNum size: " << other.block_repeat_and_segment_element_num[index] << " Address info: ";
    for (int j = 0; j < depth; ++j) {
      os << other.address_array[index][j] << " ";
    }
  }
  return os;
}

template<int maxBlockNum, int depth>
std::ostream& operator<<(std::ostream& os, const BlockInfoWithTensorScalarList<maxBlockNum, depth>& other) {
  os << other.block_info_container;
  os << "\nScalar tensor list of block part address info: ";
  for (int i = 0; i < other.block_info_container.block_num; ++i) {
    os << other.scalar_tensor_list[i] << " ";
  }
  os << "\nScalar tensor list of segment part address info: ";
  for (int i = 0; i < other.block_info_container.segment_num; ++i) {
    const int index = i + maxBlockNum / 2;
    os << other.scalar_tensor_list[index] << " ";
  }
  return os;
}

// device print for BlockInfoContainer
template<int maxBlockNum, int depth>
__mlu_func__ void print_block_info_container(const BlockInfoContainer<maxBlockNum, depth>& other) {
  printf("\nTaskId: %d", taskId);
  printf("\nmaxBlockNum: %d depth: %d.", maxBlockNum, depth);
  printf("\nBlockInfoContainer block_num: %d ", other.block_num);
  for (int i = 0; i < other.block_num; ++i) {
    const int repeat_num = i == 0 ? other.block_repeat_and_segment_element_num[i]
      : other.block_repeat_and_segment_element_num[i] - other.block_repeat_and_segment_element_num[i - 1];
    const int block_size = repeat_num * other.each_block_element_size;
    printf("\nNum size: %d, current repeat num: %d, block_repeat_num: %d, each_block_element_size: %d ",
      block_size, other.block_repeat_and_segment_element_num[i], repeat_num, other.each_block_element_size);
    printf("Address info: ");
    for (int j = 0; j < depth; ++j) {
      printf("%p ", other.address_array[i][j]);
    }
  }
  printf("\nBlockInfoContainer segment_num: %d BlockInfoContainer segment repeat num: %d ",
         other.segment_num, other.segment_num);
  for (int i = 0; i < other.segment_num; ++i) {
    const int index = i + maxBlockNum / 2;
    printf("\nNum size: %d Address info: ", other.block_repeat_and_segment_element_num[index]);
    for (int j = 0; j < depth; ++j) {
      printf("%p ", other.address_array[index][j]);
    }
  }
}

template<template<int, typename UtupleTypeList> typename className,
         typename tupleTypeList, int end, int current = 0>
struct static_unrool {
  template<typename... ARGS>
  __mlu_func__ __mlu_host__ static inline void with_args(ARGS&&... args) {
    className<current, tupleTypeList>::apply(std::forward<ARGS>(args)...);
    static_unrool<className, tupleTypeList, end, current + 1>::with_args(args...);
  }
};

template<template<int, typename UtupleTypeList> typename className,
         typename tupleTypeList, int end>
struct static_unrool<className, tupleTypeList, end, end> {
  template<typename... ARGS>
  __mlu_func__ __mlu_host__ static inline void with_args(ARGS&&... args) {}
};

template<int index, typename tupleTypeList>
struct CaculateDataPtrOffset {
  template<typename array_t, typename container_t>
  __mlu_func__ __mlu_host__ static inline void apply(array_t& array,
                                                     const container_t& data_ptr,
                                                     const int64_t& offset) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    array[index] = reinterpret_cast<T*>(data_ptr[index]) + offset;
  }

  // init data_ptr with offset and index.
  template<typename array_t>
  __mlu_func__ static inline void apply(array_t& array,
                                        char* start_ptr,
                                        const int& char_element_num) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    array[index] = reinterpret_cast<T*>(start_ptr + index * char_element_num);
  }

  template<typename array_t>
  __mlu_func__ static inline void apply(array_t& array,
                                        void* const data_ptr[],
                                        const int& repeat_time,
                                        const int& each_time_element_num) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    array[index] = reinterpret_cast<T*>(data_ptr[index]) + repeat_time * each_time_element_num;
  }

  template<typename array_t>
  __mlu_func__ static inline void apply(array_t& array,
                                        void* const data_ptr[]) {
    array[index] = data_ptr[index];
  }
};

// Note [multi tensor data structure]
// write and read contiguous memory between different cores.
// Assume each taskID is a cuda threadIdx.x, read, compute and write a block data.
// Assume taskDim is a cuda blockDim.x, and only have one block.
// Data load explain:
// Each block data contains element_num is greater or equal to SIZE_PER_REGION_ADAM / sizeof(float);
// |      block1       |       block2      | ... |      blockN       | segment1 | segment2 | segmentN |
// | taskid0 | taskid1 | taskid2 | taskid3 | ... | taskid0 | taskid1 |  taskid2 |    ...   | taskidN  |
// Each block part maybe contains multiple repeat_num, each repeat_num contains element_num data. And
// can be computed in parallel by several taskID.
// Each segment part is a remainder part, and can be computed by one taskID.
// maxBlockNum is used for segment dptr index.
template<int maxBlockNum, int depth, typename tupleTypeList,
         template<typename, int, int> typename Functor, typename... ARGS>
void multi_tensor_apply(const std::vector<std::array<void*, depth>>& data_ptr_list,
                        const std::vector<int64_t>& sizes,
                        const int& block_size,
                        cnrtQueue_t queue,
                        cnrtFunctionType_t k_type,
                        cnrtDim3_t k_dim,
                        Functor<tupleTypeList, maxBlockNum, depth>&& func,
                        ARGS&& ...args) {
  const int tensors_num = data_ptr_list.size();
  BANGC_PARAM_CHECK("multi_tensor_apply", tensors_num == sizes.size(),
                    "TensorList and sizes num need be equal.");
  BANGC_PARAM_CHECK("multi_tensor_apply", maxBlockNum % 2 == 0,
                    "maxBlockNum need be multiply of 2.");
  // TODO(shangang): It's not efficiently to store block part and segement part in half part
  // of BlockInfoContainer. Maybe more efficiently way is to stored block part from begin of
  // BlockInfoContainer, and store segment part from the end of BlockInfoContainer.
  constexpr int half_max_block_num = maxBlockNum / 2;
  // More details in [BlockInfoContainer]
  BlockInfoContainer<maxBlockNum, depth> container;
  container.each_block_element_size = block_size;
  int& block_index = container.block_num;
  int& segment_index = container.segment_num;
  for (int i = 0; i < tensors_num; ++i) {
    const int num_elements = sizes[i];
    const int local_repeat_num = num_elements / block_size;
    const int remain_element_num = num_elements % block_size;
    if (local_repeat_num != 0) {
      container.block_repeat_and_segment_element_num[block_index]
        = block_index == 0 ? local_repeat_num
            : container.block_repeat_and_segment_element_num[block_index - 1] + local_repeat_num;
      for (int j = 0; j < depth; ++j) {
        container.address_array[block_index][j] = data_ptr_list[i][j];
      }
      block_index++;
    }
    if (remain_element_num != 0) {
      const int address_array_index = half_max_block_num + segment_index;
      container.block_repeat_and_segment_element_num[address_array_index] = remain_element_num;
      const int64_t offset = local_repeat_num * block_size;
      static_unrool<CaculateDataPtrOffset, tupleTypeList, depth>::with_args(
        container.address_array[address_array_index], data_ptr_list[i], offset);
      segment_index++;
    }
    if ((block_index == half_max_block_num) || (segment_index == half_max_block_num)
        || (i == tensors_num - 1)) {
      func.call<<<k_dim, k_type, queue>>>(container, std::forward<ARGS>(args)...);
      block_index = 0;
      segment_index = 0;
    }
  }
}

template<int maxBlockNum, int depth, typename tupleTypeList,
         typename Functor, typename... ARGS>
void multi_tensor_apply_with_scalar_tensor(const std::vector<std::array<void*, depth>>& data_ptr_list,
                                           const std::vector<int64_t>& sizes,
                                           const std::vector<void*>& steps_ptr_list,
                                           const int& block_size,
                                           cnrtQueue_t stream,
                                           cnrtFunctionType_t k_type,
                                           cnrtDim3_t k_dim,
                                           Functor&& func,
                                           ARGS&& ...args) {
  const int tensors_num = data_ptr_list.size();
  BANGC_PARAM_CHECK("multi_tensor_apply", tensors_num == sizes.size(),
                    "TensorList and sizes num need be equal.");
  BANGC_PARAM_CHECK("multi_tensor_apply", tensors_num == steps_ptr_list.size(),
                    "TensorList and step tensorlist num need be equal."); 
  BANGC_PARAM_CHECK("multi_tensor_apply", maxBlockNum % 2 == 0,
                    "maxBlockNum need be multiply of 2.");
  constexpr int half_max_block_num = maxBlockNum / 2;
  // More details in [BlockInfoContainer]
  BlockInfoWithTensorScalarList<maxBlockNum, depth> container;
  BlockInfoContainer<maxBlockNum, depth>& inner_container = container.block_info_container;
  inner_container.each_block_element_size = block_size;
  int& block_index = inner_container.block_num;
  int& segment_index = inner_container.segment_num;
  for (int i = 0; i < tensors_num; ++i) {
    const int num_elements = sizes[i];
    const int local_repeat_num = num_elements / block_size;
    const int remain_element_num = num_elements % block_size;
    if (local_repeat_num != 0) {
      inner_container.block_repeat_and_segment_element_num[block_index]
        = block_index == 0 ? local_repeat_num
            : inner_container.block_repeat_and_segment_element_num[block_index - 1] + local_repeat_num;
      for (int j = 0; j < depth; ++j) {
        inner_container.address_array[block_index][j] = data_ptr_list[i][j];
      }
      container.scalar_tensor_list[block_index] = steps_ptr_list[i];
      block_index++;
    }
    if (remain_element_num != 0) {
      const int address_array_index = half_max_block_num + segment_index;
      inner_container.block_repeat_and_segment_element_num[address_array_index] = remain_element_num;
      const int64_t offset = local_repeat_num * block_size;
      static_unrool<CaculateDataPtrOffset, tupleTypeList, depth>::with_args(
        inner_container.address_array[address_array_index], data_ptr_list[i], offset);
      container.scalar_tensor_list[address_array_index] = steps_ptr_list[i];
      segment_index++;
    }
    if ((block_index == half_max_block_num) || (segment_index == half_max_block_num)
        || (i == tensors_num - 1)) {
      func.call<<<k_dim, k_type, stream>>>(container, std::forward<ARGS>(args)...);
      block_index = 0;
      segment_index = 0;
    }
  }
}

// Load multi gdram data to nram, and tupleTypeList is for using each data type.
template<int index, typename tupleTypeList>
struct LoadMultiDatas {
  template<typename array_t, typename container_t>
  __mlu_func__ static inline void apply(const container_t& data_ptr,
                                        const array_t& array,
                                        const int& copy_size) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    __memcpy_async(array[index], data_ptr[index], copy_size * sizeof(T), mluMemcpyDirection_t::GDRAM2NRAM);
  }
  
  // apply with offset.
  template<typename array_t, typename container_t>
  __mlu_func__ static inline void apply(const container_t& data_ptr,
                                        const array_t& array,
                                        const int& copy_size,
                                        const int& offset) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    __memcpy_async(reinterpret_cast<T*>(array[index]) + offset, data_ptr[index],
                   copy_size * sizeof(T), mluMemcpyDirection_t::GDRAM2NRAM);
  }
};

// Store multi nram data to gdram, and tupleTypeList is for using each data type.
template<int index, typename tupleTypeList>
struct StoreMultiDatas {
  template<typename array_t, typename container_t>
  __mlu_func__ static inline void apply(const container_t& data_ptr,
                                        const array_t& array,
                                        const int& copy_size) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    if (array[index] == nullptr) return;
    __memcpy_async(data_ptr[index], array[index], copy_size * sizeof(T), mluMemcpyDirection_t::NRAM2GDRAM);
  }
};

// See Note [BlockOffset]
static constexpr int MAX_STORE_CIRCLE_NUM = 8;
static_assert(MAX_STORE_CIRCLE_NUM>=4, "MAX_STORE_CIRCLE_NUM need to greater than 4.");

/**
 * Note [MemoryPolicy]
 * write and read contiguous memory between different cores.
 * Assume each taskID is a cuda threadIdx.x, read, compute and write a block data.
 * Assume taskDim is a cuda blockDim.x, and only have one block.
 * Data load explain:
 * Each block data contains element_num is greater or equal to SIZE_PER_REGION_ADAM / sizeof(float);
 * |      block1       |       block2      | ... |      blockN       | segment1 | segment2 | segmentN |
 * | taskid0 | taskid1 | taskid2 | taskid3 | ... | taskid0 | taskid1 |  taskid2 |    ...   | taskidN  |
 * Each block part maybe contains multiple repeat_num, each repeat_num contains element_num data. And
 * can be computed in parallel by several taskID.
 * Each segment part is a remainder part, and can be computed by one taskID.
 *
 * [Load and Store]
 * Load and store data is used to load and store data between gdram and nram. In mlu side, io and
 * compute pipeline is always used to overlap compute and io time.
 * This like: load data -> compute -> store data -> load data -> compute -> store data.
 * load data0     load data1     store data0     store data1     store data2
 *              compute data0   compute data1   compute data2
 *                                load data2
 * Load ping-pong data first, then compute and store ping data. Store data index is alway
 * later than load data index. This is very different with gpu side, so need to carefully.
*/
template<typename tupleTypeList, int maxBlockNum, int depth>
class MemoryPolicy {
 public:
  __mlu_func__ MemoryPolicy(const BlockInfoContainer<maxBlockNum, depth>& block_info_container)
                            : container_(block_info_container),
                              with_step_(false) {
    this->block_total_repeat_num_ = this->container_.block_num == 0 ? 0
      : this->container_.block_repeat_and_segment_element_num[this->container_.block_num - 1];
    const int total_repeat_num = this->block_total_repeat_num_ + this->container_.segment_num;
    const int repeat = total_repeat_num / taskDim;
    const int remain = total_repeat_num % taskDim;
    // each core repeat time and no remain part.
    this->circle_ = taskId < remain ? repeat + 1 : repeat;
    // segment_block_offset is used to find segment data index.
    // repeat_times * taskDim + taskId - this->block_total_repeat_num_ 
    // + maxBlockNum / 2.
    this->segment_block_offset_ = maxBlockNum / 2 - this->block_total_repeat_num_;
    // get circle index to block index.
    this->init_block_offset_info();
  }

  __mlu_func__ MemoryPolicy(const MemoryPolicy&) = delete;
  __mlu_func__ MemoryPolicy& operator=(const MemoryPolicy&) = delete;
  __mlu_func__ MemoryPolicy(MemoryPolicy&&) = delete;
  __mlu_func__ MemoryPolicy& operator=(MemoryPolicy&&) = delete;

  __mlu_func__ inline const int get_circle_num() {
    return this->circle_;
  }

  __mlu_func__ inline const int get_each_block_element_num() {
    return this->container_.each_block_element_size;
  }

  __mlu_func__ inline const int get_block_index(const int& index) {
    const int offset_index = index % MAX_STORE_CIRCLE_NUM;
    return *(this->data_index_list_ + offset_index);
  }

  template<typename T, int depth_index>
  __mlu_func__ inline void load(const int& index, T* nram_ptr, int& copy_size) {
    const int offset_index = index % MAX_STORE_CIRCLE_NUM;
    copy_size = this->block_instance_.copy_size_[offset_index];
    __memcpy_async(nram_ptr, this->block_instance_.gdram_ptr_[offset_index][depth_index],
                   copy_size * sizeof(T), mluMemcpyDirection_t::GDRAM2NRAM);
    // circle compute index.
    if (offset_index == this->block_instance_.start_index_) {
      this->circle_rewrite_block_info();
    }
  }

  template<typename T, int depth_index>
  __mlu_func__ inline void load_with_offset(const int& index,
                                            T* nram_ptr,
                                            int& copy_size,
                                            const int& offset) {
    const int offset_index = index % MAX_STORE_CIRCLE_NUM;
    copy_size = this->block_instance_.copy_size_[offset_index];
    __memcpy_async(nram_ptr + offset, this->block_instance_.gdram_ptr_[offset_index][depth_index],
                   copy_size * sizeof(T), mluMemcpyDirection_t::GDRAM2NRAM);
    // circle compute index.
    if (offset_index == this->block_instance_.start_index_) {
      this->circle_rewrite_block_info();
    }
  }

  template<typename T, int depth_index>
  __mlu_func__ inline void store(const int& index, T* nram_ptr, const int& copy_size) {
    const int offset_index = index % MAX_STORE_CIRCLE_NUM;
    __memcpy_async(this->block_instance_.gdram_ptr_[offset_index][depth_index], nram_ptr,
                   copy_size * sizeof(T), mluMemcpyDirection_t::NRAM2GDRAM);
  }

  __mlu_func__ inline void multi_data_load_same_size(const int& index,
                                                     void* nram_ptr[],
                                                     int& copy_size) {
    const int offset_index = index % MAX_STORE_CIRCLE_NUM;
    copy_size = this->block_instance_.copy_size_[offset_index];
    static_unrool<LoadMultiDatas, tupleTypeList, depth>::with_args(
        this->block_instance_.gdram_ptr_[offset_index], nram_ptr, copy_size);
    // circle compute index.
    if (offset_index == this->block_instance_.start_index_) {
      this->circle_rewrite_block_info();
    }
  }

  __mlu_func__ inline void multi_data_load_same_size_with_offset(const int& index,
                                                                 void* nram_ptr[],
                                                                 int& copy_size,
                                                                 const int& offset) {
    const int offset_index = index % MAX_STORE_CIRCLE_NUM;
    copy_size = this->block_instance_.copy_size_[offset_index];
    static_unrool<LoadMultiDatas, tupleTypeList, depth>::with_args(
        this->block_instance_.gdram_ptr_[offset_index], nram_ptr, copy_size, offset);
    // circle compute index.
    if (offset_index == this->block_instance_.start_index_) {
      this->circle_rewrite_block_info();
    }
  }

  __mlu_func__ inline void multi_data_store_same_size(const int& index,
                                                      void* nram_ptr[],
                                                      const int& copy_size) {
    const int offset_index = index % MAX_STORE_CIRCLE_NUM;
    static_unrool<StoreMultiDatas, tupleTypeList, depth>::with_args(
        this->block_instance_.gdram_ptr_[offset_index], nram_ptr, copy_size);
  }

 private:
  __mlu_func__ void init_block_offset_info() {
    for (int i = 0; i < MAX_STORE_CIRCLE_NUM; ++i) {
      if (i >= this->circle_) {
        break;
      }
      this->calculate_block_info(i, i * taskDim);
    }
    this->block_instance_.repeated_times_ = 1;
    this->block_instance_.start_index_ = MAX_STORE_CIRCLE_NUM - 1;
  }

  __mlu_func__ void circle_rewrite_block_info() {
    constexpr int max_rewrite_num = MAX_STORE_CIRCLE_NUM - 3;
    const int already_repeat_num = (this->block_instance_.repeated_times_ - 1) * max_rewrite_num
                                   + MAX_STORE_CIRCLE_NUM;
    // redirector_index is used to find right array index.
    int redirector_index = 0;
    for (int i = 0; i < max_rewrite_num; ++i) {
      int index = already_repeat_num + i;
      if (index >= this->circle_) {
        break;
      }
      // Note [BlockOffset]
      // assume MAX_STORE_CIRCLE_NUM is 8, and need to reserve 3 spaces for last part to store.
      // So each time only update 8 - 3 spaces.
      // Simple index accumulate example:
      // | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23|
      // To redirector index is:
      // | 0, 1, 2, 3, 4, 5, 6, 7, 0, 1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7|
      // and start index is:
      // | -, -, -, -, -, -, -, 7, -, -,  -,  -,  4,  -,  -,  -,  -,  1,  -,  -,  -,  -,  6,  -|
      // write the next index of BlockOffset array.
      redirector_index = (this->block_instance_.start_index_ + i + 1) % MAX_STORE_CIRCLE_NUM;
      this->calculate_block_info(redirector_index, index * taskDim);
    }
    this->block_instance_.start_index_ = redirector_index;
    ++this->block_instance_.repeated_times_;
  }

  __mlu_func__ inline void calculate_block_info(const int& index,
                                                const int& grid_offset) {
    // Each core repeat index, and using offset taskDim to next data block.
    // index * taskDim is mean grid offset. taskId is mean thread offset.
    // alread_data_offset is mean already handled circle data offset.
    const int core_current_repeat_num = grid_offset + taskId;
    // data block index
    int block_index = 0;
    if (core_current_repeat_num < this->block_total_repeat_num_) {
      // Can't using this function in each compute loop, cause stream_load_ge
      // break compute pipeline. So using block, offset and copy size arrays to store all
      // block index, offset and copy size, when core init stack data.
      // After this, only need to read block index and offset when compute pipeline.
      for (int i = 0; i < this->container_.block_num; ++i) {
        if (core_current_repeat_num < container_.block_repeat_and_segment_element_num[i]) {
          block_index = i;
          break;
        }
      }
      const int repeat_num = block_index == 0 ? core_current_repeat_num : core_current_repeat_num -
          this->container_.block_repeat_and_segment_element_num[block_index - 1];
      static_unrool<CaculateDataPtrOffset, tupleTypeList, depth>::with_args(
          this->block_instance_.gdram_ptr_[index], this->container_.address_array[block_index],
          repeat_num, this->container_.each_block_element_size);
      this->block_instance_.copy_size_[index] = this->container_.each_block_element_size;
    } else {
      // segment data part
      block_index  = core_current_repeat_num + this->segment_block_offset_;
      static_unrool<CaculateDataPtrOffset, tupleTypeList, depth>::with_args(
          this->block_instance_.gdram_ptr_[index],
          this->container_.address_array[block_index]);
      this->block_instance_.copy_size_[index] =
        this->container_.block_repeat_and_segment_element_num[block_index];
    }
    this->data_index_list_[index] = block_index;
  }

 private:
   template<int NUM>
   struct BlockOffset {
     void* gdram_ptr_[NUM][depth];
     int copy_size_[NUM];
     int start_index_ = 0;
     int repeated_times_ = 0;
   };
   const BlockInfoContainer<maxBlockNum, depth>& container_;
   struct BlockOffset<MAX_STORE_CIRCLE_NUM> block_instance_;
   int data_index_list_[MAX_STORE_CIRCLE_NUM];
   int block_total_repeat_num_ = 0;
   int segment_block_offset_ = 0;
   int circle_ = 0;
   bool with_step_ = false;
};

// function invoke

// Now only just support same num inputs and outputs, and it's easy to support
// different num inputs and outputs.
template<typename tupleTypeList, typename FUNC, typename Array_t,
         typename... ARGS, std::size_t... I>
__mlu_func__ constexpr void invoke_impl(FUNC&& func,
                                        Array_t&& nram_array,
                                        std::index_sequence<I...>,
                                        ARGS&&... args) {
  std::forward<FUNC>(func)((reinterpret_cast<std::tuple_element_t<I, tupleTypeList>*>(nram_array[I]))...,
                            std::forward<ARGS>(args)...);
}

template<int depth, typename tupleTypeList, typename FUNC, typename Array_t, typename... ARGS>
__mlu_func__ constexpr void invoke(FUNC&& func, Array_t&& nram_array, ARGS&&... args) {
  invoke_impl<tupleTypeList>(func, nram_array, std::make_index_sequence<depth>(), args...);
}

}  // namespace torch_mlu::bangcommon