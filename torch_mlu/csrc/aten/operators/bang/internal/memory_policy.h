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

#include <type_traits>
#include "aten/operators/bang/internal/load_and_store.h"
#include "aten/operators/bang/internal/block_info_container.h"

namespace torch_mlu::bangcommon {

// See Note [BlockOffset]
static constexpr int MAX_STORE_CIRCLE_NUM = 8;
static_assert(
    MAX_STORE_CIRCLE_NUM >= 4,
    "MAX_STORE_CIRCLE_NUM need to greater than 4.");

/**
 * Note [MemoryPolicy]
 * MemoryPolicy provides a layer of abstraction for on-chip and off-chip address
 * information. There are a small data struct BlockOffset to record
 * corresponding address information between on-chip and off-chip. More details
 * in Note [BlockOffset].
 *
 * MemoryPolicy using reading continuous data between cores, each core handle
 * part of a tensor, and part size is less or equal to block size.
 *
 * Example:
 * part1 of tensor1 means the first block size of tensor1;
 * t1_p1_addr means the first block addr;
 * core1_nram_ping means the nram addr of first block of tensor1.
 * core1_gdram_store means the compute result of first block of tensor1.
 *
 * |                    ping                 |                    pong | | part1
 * of tensor1 |  part2 of tensor1 | ... | part1 of tensor2 | part2 of tensor2 |
 * |    t1_p1_addr    |    t1_p2_addr     | ... |     t2_p1_addr   | t2_p2_addr
 * | |      core1       |      core2        | ... |      core1       | core2 |
 * | core1_nram_ping  | core2_nram_ping   | ... | core1_nram_pong  |
 * core2_nram_pong  | |       x          |       x           | ...
 * |core1_gdram_store |core2_gdram_store |
 *
 * 1) Each core read tensor info from BlockInfoContainer and core calculates its
 * own required gdram data block. Then store those info in BlockOffset; 2) Each
 * repeat circle of each core, only simply inputting the address array of the
 * nram, the corresponding data loading can be completed by multi-load or
 * multi-store;
 *
 * Note [Load and Store]
 * Load and store data means copy data between gdram and nram/wram.
 * In mlu side, ping-pong pipeline is always used to overlap compute and io
 * time.
 * This pipeline like:
 *   ping step       pong step           ping step           pong step
 * load data0_0    load data1_0        store data0_0        store data1_0
 *                compute data0_0     compute data1_0      data1_1 compute
 *                                     load data1_1         load data2_0
 * note1:
 * data{index}_{data_index}, index means ping or pong step, now 0 means ping
 * step, 1 means pong step. data_index means data segment index, 0 means first
 * segment of whole data block, 1 means second and so on. As we know from this
 * pipeline. A segment data index will load in first step and store result in
 * next step. This is very different with GPU side(GPU without explicit pipeline
 * like this).
 *
 */
template <typename tupleTypeList, int maxBlockNum, int depth>
class MemoryPolicy {
 public:
  __mlu_func__ MemoryPolicy(
      const BlockInfoContainer<maxBlockNum, depth>& block_info_container)
      : container_(block_info_container) {
    this->total_repeat_num_ = this->container_.total_tensor_num == 0
        ? 0
        : this->container_
              .repeat_block_num[this->container_.total_tensor_num - 1];
    const int repeat = this->total_repeat_num_ / taskDim;
    const int remain = this->total_repeat_num_ % taskDim;
    // each core repeat time and no remain part.
    this->circle_ = taskId < remain ? repeat + 1 : repeat;
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
    return this->container_.block_size;
  }

  __mlu_func__ inline const int get_block_index(const int& repeat_num) {
    return this->tensor_index_list_[repeat_num % MAX_STORE_CIRCLE_NUM];
  }

  __mlu_func__ inline void multi_data_load_same_size(
      const int& repeat_num,
      void* nram_ptr[],
      int& copy_size) {
    const int offset_index = repeat_num % MAX_STORE_CIRCLE_NUM;
    copy_size = this->block_instance_.copy_size_[offset_index];
    static_unrool<LoadMultiDatas, tupleTypeList, depth>::with_args(
        this->block_instance_.gdram_ptr_[offset_index], nram_ptr, copy_size);
    // circle compute index.
    if (offset_index == this->block_instance_.next_start_index_) {
      this->circle_rewrite_block_info(repeat_num);
    }
  }

  __mlu_func__ inline void multi_data_load_same_size_with_offset(
      const int& repeat_num,
      void* nram_ptr[],
      int& copy_size,
      const int& offset) {
    const int offset_index = repeat_num % MAX_STORE_CIRCLE_NUM;
    copy_size = this->block_instance_.copy_size_[offset_index];
    static_unrool<LoadMultiDatas, tupleTypeList, depth>::with_args(
        this->block_instance_.gdram_ptr_[offset_index],
        nram_ptr,
        copy_size,
        offset);
    // circle compute index.
    if (offset_index == this->block_instance_.next_start_index_) {
      this->circle_rewrite_block_info(repeat_num);
    }
  }

  __mlu_func__ inline void multi_data_store_same_size(
      const int& repeat_num,
      void* nram_ptr[],
      const int& copy_size) {
    const int offset_index = repeat_num % MAX_STORE_CIRCLE_NUM;
    static_unrool<StoreMultiDatas, tupleTypeList, depth>::with_args(
        this->block_instance_.gdram_ptr_[offset_index], nram_ptr, copy_size);
  }

 private:
  __mlu_func__ void init_block_offset_info() {
    int i = 0;
    for (; i < MAX_STORE_CIRCLE_NUM; ++i) {
      if (i >= this->circle_) {
        break;
      }
      const int start_tensor_index =
          i == 0 ? 0 : this->tensor_index_list_[i - 1];
      this->calculate_block_info(start_tensor_index, i, i * taskDim);
    }
    this->block_instance_.next_start_index_ = i - 1;
  }

  __mlu_func__ void circle_rewrite_block_info(const int& current_repeat_num) {
    // Store is left behind three circle than load data.
    constexpr int max_rewrite_num = MAX_STORE_CIRCLE_NUM - 3;
    // next_index is used to find right array index.
    int next_index = 0;
    for (int i = 0; i < max_rewrite_num; ++i) {
      int next_repeat_num = current_repeat_num + i + 1;
      next_index = next_repeat_num % MAX_STORE_CIRCLE_NUM;
      if (next_repeat_num >= this->circle_) {
        break;
      }
      const int current_index = (current_repeat_num + i) % MAX_STORE_CIRCLE_NUM;
      const int start_tensor_index = this->tensor_index_list_[current_index];
      this->calculate_block_info(
          start_tensor_index, next_index, next_repeat_num * taskDim);
    }
    this->block_instance_.next_start_index_ = next_index;
  }

  __mlu_func__ inline void calculate_block_info(
      const int& start_tensor_index,
      const int& store_index,
      const int& grid_offset) {
    // Each core repeat index, and using offset taskDim to next data block.
    // index * taskDim is mean grid offset. taskId is mean thread offset.
    const int core_current_repeat_num = grid_offset + taskId;
    // Can't using this function in each compute loop, cause stream_load_ge
    // break compute pipeline. So using block, offset and copy size arrays to
    // store all block index, offset and copy size, when core init stack data.
    // After this, only need to read block index and offset in compute
    // pipeline.
    int accumulate_repeat_num = 0;
    int previous_accumulate_repeat_num = 0;
    int left = start_tensor_index;
    int right = this->container_.total_tensor_num;
    int tensor_index = left + (right - left) / 2;
    while (left <= right) {
      accumulate_repeat_num = container_.repeat_block_num[tensor_index];
      if (core_current_repeat_num < accumulate_repeat_num) {
        if (tensor_index == 0)
          break;
        previous_accumulate_repeat_num =
            container_.repeat_block_num[tensor_index - 1];
        if (previous_accumulate_repeat_num <= core_current_repeat_num) {
          break;
        }
        right = tensor_index - 1;
      } else {
        left = tensor_index + 1;
      }
      tensor_index = left + (right - left) / 2;
    }
    int repeat_num = tensor_index == 0
        ? core_current_repeat_num
        : core_current_repeat_num - previous_accumulate_repeat_num;
    const int remain_num = this->container_.remainder_num[tensor_index];
    if (core_current_repeat_num == (accumulate_repeat_num - 1) &&
        remain_num != 0) {
      this->block_instance_.copy_size_[store_index] = remain_num;
    } else {
      this->block_instance_.copy_size_[store_index] =
          this->container_.block_size;
    }
    static_unrool<CaculateDataPtrOffset, tupleTypeList, depth>::with_args(
        this->block_instance_.gdram_ptr_[store_index],
        this->container_.address_array[tensor_index],
        repeat_num,
        this->container_.block_size);
    this->tensor_index_list_[store_index] = tensor_index;
  }

 private:
  // Note [BlockOffset]
  // BlockOffset is used to store block info for GDRAM2NRAM and NRAN2GDRAM.
  // Each block info contains gdram address, copy size, next start index.
  // Gdram address and copy size is used for memcpy, next start index and
  // is used to control circle info of BlockOffset.
  // Assume template NUM is MAX_STORE_CIRCLE_NUM and equal to 8, and how to init
  // BlockOffset. 1) At the beginning, init all eight data space of BlockOffset,
  // which using taskDim offset
  //    to compute gdram address and copy size for current core.
  //    taskDim offset is like current_core_repeat_num * taskDim + taskId
  // 2) Only update part of block offset data each tims, when
  // current_core_repeat_num redirector
  //    index equal to next start index. Part num is MAX_STORE_CIRCLE_NUM - 3
  //    for ping-pong pipeline. More details: Note [Load and Store].
  //
  // Example of core repeat num relationship with next start index and inner
  // circle num: Repeat Num: | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
  // 15, 16, 17, 18, 19, 20, 21, 22, 23| Redirector index is: | 0, 1, 2, 3, 4,
  // 5, 6, 7, 0, 1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7| Next
  // start index is: | -, -, -, -, -, -, -, 7, -, -,  -,  -,  4,  -,  -,  -,  -,
  // 1,  -,  -,  -,  -,  6,  -|
  template <int NUM>
  struct BlockOffset {
    void* gdram_ptr_[NUM][depth];
    int copy_size_[NUM];
    int next_start_index_ = 0;
  };
  const BlockInfoContainer<maxBlockNum, depth>& container_;
  struct BlockOffset<MAX_STORE_CIRCLE_NUM> block_instance_;
  int tensor_index_list_[MAX_STORE_CIRCLE_NUM] = {0};
  int total_repeat_num_ = 0;
  int circle_ = 0;
};

// function invoke

// Now only just support same num inputs and outputs, and it's easy to support
// different num inputs and outputs.
template <
    typename tupleTypeList,
    typename FUNC,
    typename Array_t,
    typename... ARGS,
    std::size_t... I>
__mlu_func__ constexpr void invoke_impl(
    FUNC&& func,
    Array_t&& nram_array,
    std::index_sequence<I...>,
    ARGS&&... args) {
  std::forward<FUNC>(func)(
      (reinterpret_cast<std::tuple_element_t<I, tupleTypeList>*>(
          nram_array[I]))...,
      std::forward<ARGS>(args)...);
}

template <
    int depth,
    typename tupleTypeList,
    typename FUNC,
    typename Array_t,
    typename... ARGS>
__mlu_func__ constexpr void invoke(
    FUNC&& func,
    Array_t&& nram_array,
    ARGS&&... args) {
  invoke_impl<tupleTypeList>(
      func, nram_array, std::make_index_sequence<depth>(), args...);
}

} // namespace torch_mlu::bangcommon