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

#include "aten/operators/bang/internal/memory_policy.h"

namespace torch_mlu::bangcommon {

// function invoke
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

// Now only just support same num inputs and outputs, and it's easy to support
// different num inputs and outputs.
template <
    typename tupleTypeList,
    int maxBlockNum,
    int depth,
    template <typename, int, int>
    typename MemoryPolicy,
    typename Functor,
    typename... ARGS>
__mlu_func__ constexpr void do_three_stage_pipeline_compute(
    MemoryPolicy<tupleTypeList, maxBlockNum, depth>& data_handler,
    Functor&& compute_func,
    void* (&load_ping_nram)[depth],
    void* (&store_ping_nram)[depth],
    void* (&load_pong_nram)[depth],
    void* (&store_pong_nram)[depth],
    const int& load_offset,
    ARGS&&... args) {
  const int circle_num = data_handler.get_circle_num();
  int ping_size = 0;
  int pong_size = 0;
  if (circle_num > 0) {
    data_handler.multi_data_load_same_size_with_offset(
        0, load_ping_nram, ping_size, load_offset);
    __sync_io();
  }
  if (circle_num > 1) {
    // pong and compute
    data_handler.multi_data_load_same_size_with_offset(
        1, load_pong_nram, pong_size, load_offset);
    // compute ping
    invoke<depth, tupleTypeList>(
        compute_func,
        load_ping_nram,
        ping_size,
        load_offset,
        std::forward<ARGS>(args)...);
    __sync_io_move_compute(false, false, true, true, false, false);
    __sync_io_move_compute(true, false, false, false, false, true);
  }
  for (int i = 0; i < circle_num - 2; i++) {
    // store data
    int ping_pong = i % 2;
    data_handler.multi_data_store_same_size(
        i,
        ping_pong ? store_pong_nram : store_ping_nram,
        ping_pong ? pong_size : ping_size);
    data_handler.multi_data_load_same_size_with_offset(
        i + 2,
        ping_pong ? load_pong_nram : load_ping_nram,
        ping_pong ? pong_size : ping_size,
        load_offset);
    // compute data date order is different with load and store data.
    ping_pong = (i + 1) % 2;
    invoke<depth, tupleTypeList>(
        compute_func,
        ping_pong ? load_pong_nram : load_ping_nram,
        ping_pong ? pong_size : ping_size,
        load_offset,
        std::forward<ARGS>(args)...);
    __sync_io_move_compute(false, false, true, true, false, false);
    __sync_io_move_compute(true, false, false, false, false, true);
  }

  // store circle - 1 data
  int ping_pong = circle_num % 2;
  if (circle_num > 1) {
    data_handler.multi_data_store_same_size(
        circle_num - 2,
        ping_pong ? store_pong_nram : store_ping_nram,
        ping_pong ? pong_size : ping_size);
  }

  ping_pong = (circle_num + 1) % 2;
  if (circle_num > 0) {
    // compute last circle
    invoke<depth, tupleTypeList>(
        compute_func,
        ping_pong ? load_pong_nram : load_ping_nram,
        ping_pong ? pong_size : ping_size,
        load_offset,
        std::forward<ARGS>(args)...);
    __sync_compute();
    // store last circle data
    data_handler.multi_data_store_same_size(
        circle_num - 1,
        ping_pong ? store_pong_nram : store_ping_nram,
        ping_pong ? pong_size : ping_size);
  }
}

} // namespace torch_mlu::bangcommon