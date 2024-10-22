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

#include <tuple>

namespace torch_mlu::bangcommon {

// Compile-time expansion.
template <
    template <int, typename UtupleTypeList>
    typename className,
    typename tupleTypeList,
    int end,
    int current = 0>
struct static_unrool {
  template <typename... ARGS>
  __mlu_func__ __mlu_host__ static inline void with_args(ARGS&&... args) {
    className<current, tupleTypeList>::apply(std::forward<ARGS>(args)...);
    static_unrool<className, tupleTypeList, end, current + 1>::with_args(
        args...);
  }
};

template <
    template <int, typename UtupleTypeList>
    typename className,
    typename tupleTypeList,
    int end>
struct static_unrool<className, tupleTypeList, end, end> {
  template <typename... ARGS>
  __mlu_func__ __mlu_host__ static inline void with_args(ARGS&&... args) {}
};

template <int index, typename tupleTypeList>
struct CaculateDataPtrOffset {
  template <typename array_t, typename container_t>
  __mlu_func__ __mlu_host__ static inline void apply(
      array_t& array,
      const container_t& data_ptr,
      const int64_t& offset) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    array[index] = reinterpret_cast<T*>(data_ptr[index]) + offset;
  }

  // init data_ptr with offset and index.
  template <typename array_t>
  __mlu_func__ static inline void apply(
      array_t& array,
      char* start_ptr,
      const int& char_element_num) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    array[index] = reinterpret_cast<T*>(start_ptr + index * char_element_num);
  }

  template <typename array_t>
  __mlu_func__ static inline void apply(
      array_t& array,
      void* const data_ptr[],
      const int& repeat_time,
      const int& each_time_element_num) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    array[index] = reinterpret_cast<T*>(data_ptr[index]) +
        repeat_time * each_time_element_num;
  }

  template <typename array_t>
  __mlu_func__ static inline void apply(
      array_t& array,
      void* const data_ptr[]) {
    array[index] = data_ptr[index];
  }
};

// Load multi gdram data to nram, and tupleTypeList is for using each data type.
template <int index, typename tupleTypeList>
struct LoadMultiDatas {
  template <typename array_t, typename container_t>
  __mlu_func__ static inline void apply(
      const container_t& data_ptr,
      const array_t& array,
      const int& copy_size) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    __memcpy_async(
        array[index],
        data_ptr[index],
        copy_size * sizeof(T),
        mluMemcpyDirection_t::GDRAM2NRAM);
  }

  // The offset will only take effect when the data type
  // is set to half and bfloat16.
  template <typename array_t, typename container_t>
  __mlu_func__ static inline void apply(
      const container_t& data_ptr,
      const array_t& array,
      const int& copy_size,
      const int& offset) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    T* nram_addr = reinterpret_cast<T*>(array[index]);
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
      nram_addr += offset;
    }
    __memcpy_async(
        nram_addr,
        data_ptr[index],
        copy_size * sizeof(T),
        mluMemcpyDirection_t::GDRAM2NRAM);
  }
};

// Store multi nram data to gdram, and tupleTypeList is for using each data
// type.
template <int index, typename tupleTypeList>
struct StoreMultiDatas {
  template <typename array_t, typename container_t>
  __mlu_func__ static inline void apply(
      const container_t& data_ptr,
      const array_t& array,
      const int& copy_size) {
    using T = std::tuple_element_t<index, tupleTypeList>;
    if (array[index] == nullptr)
      return;
    __memcpy_async(
        data_ptr[index],
        array[index],
        copy_size * sizeof(T),
        mluMemcpyDirection_t::NRAM2GDRAM);
  }
};

} // namespace torch_mlu::bangcommon