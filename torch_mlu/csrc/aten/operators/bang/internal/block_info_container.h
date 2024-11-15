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

#include <iostream>

namespace torch_mlu::bangcommon {

// Kernel launch support host data size is 24 * 1024 Bytes, now
// just using 8KB for host data when kernel launch.
// maxBlockNum * (depth * sizeof(void*) + 2 * sizeof(int))
// + 2 * sizeof(int) <= 8 * 1024.
static constexpr int depth_to_max_blockinfo[5] = {400, 250, 180, 140, 116};

/**
 * Note [BlockInfoContainer]
 * BlockInfoContainer is used to store the block info of tensors, which is used
 * in for kernel launch and compute. Each kernel can use block info to load,
 * compute and store data.
 *
 * 1) address_array is a two-dimensional array, inner array is to store tensors'
 * ptr, which memory space need be contiguous, and there have same dtype,
 * layout, size and stride. outter array is to tensor lists of inner array; 2)
 * block_size is computed from nram/wram size for each input; 3)
 * repeat_block_num is an accumulated array, where each current repeat num is
 * the sum of the previous repeat num and the current tensor repeat num. And
 *    current repeat num is equal block_size divided by the tensor size;
 * 4) remainder_num is a array, and to store remain num of each input;
 * 5) total_tensor_num is to store the size of inner array tensors.
 *
 * How to get tensor size for each ptr stored in address_array?
 * Ans: repeat_block_num is an accumulated array, so each tensor repeat num is
 * equal to current repeat num sub previous repeat num, and then through this
 * formula to get tensor size. repeat_num * block_size + remainder part.
 *
 */
template <int maxBlockNum, int depth>
struct BlockInfoContainer {
  static_assert(
      maxBlockNum * (sizeof(void*) * depth + 2 * sizeof(int)) +
              2 * sizeof(int) <
          8 * 1024,
      "Host launch data size is over 8 * 1024.");
  void* address_array[maxBlockNum][depth];
  // Using repeat num to avoid using int64 on device side.
  int repeat_block_num[maxBlockNum];
  int remainder_num[maxBlockNum];
  int total_tensor_num = 0;
  int block_size = 0;
};

// BlockInfoContainer with scalar tensor list.
template <int maxBlockNum, int depth>
struct BlockInfoWithTensorScalarList {
  static_assert(
      sizeof(BlockInfoContainer<maxBlockNum, depth>) +
              sizeof(void*) * maxBlockNum <
          9 * 1024,
      "Host launch data size is over 9 * 1024.");
  BlockInfoContainer<maxBlockNum, depth> block_info_container;
  void* scalar_tensor_list[maxBlockNum];
};

// Add tensor index list for some op, which need to get output for
// each tensor.
template <int maxBlockNum, int depth>
struct BlockInfoWithIndexList {
  static_assert(
      sizeof(BlockInfoContainer<maxBlockNum, depth>) +
              sizeof(int) * maxBlockNum <
          9 * 1024,
      "Host launch data size is over 9 * 1024.");
  BlockInfoContainer<maxBlockNum, depth> block_info_container;
  int index_list[maxBlockNum];
};

// host print for BlockInfoContainer
template <int maxBlockNum, int depth>
std::ostream& operator<<(
    std::ostream& os,
    const BlockInfoContainer<maxBlockNum, depth>& other) {
  os << "\nmaxBlockNum: " << maxBlockNum << " depth: " << depth;
  os << "\nBlockInfoContainer total_tensor_num: " << other.total_tensor_num;
  os << "\nEach block size: " << other.block_size;
  for (int i = 0; i < other.total_tensor_num; ++i) {
    const int repeat_num = i == 0
        ? other.repeat_block_num[i]
        : other.repeat_block_num[i] - other.repeat_block_num[i - 1];
    const int block_size = other.remainder_num[i] == 0
        ? repeat_num * other.block_size
        : (repeat_num - 1) * other.block_size + other.remainder_num[i];
    os << "\nNum size: " << block_size
       << " accumulate block_repeat_num: " << other.repeat_block_num[i]
       << " current block_repeat_num: " << repeat_num
       << " block_remain_num: " << other.remainder_num[i] << " Address info: ";
    for (int j = 0; j < depth; ++j) {
      os << other.address_array[i][j] << " ";
    }
  }
  return os;
}

// device print for BlockInfoContainer
template <int maxBlockNum, int depth>
__mlu_func__ void print_block_info_container(
    const BlockInfoContainer<maxBlockNum, depth>& other) {
  printf("\nTaskId: %d", taskId);
  printf("\nmaxBlockNum: %d depth: %d.", maxBlockNum, depth);
  printf("\nBlockInfoContainer total_tensor_num: %d ", other.total_tensor_num);
  printf("\nEach block size: %d ", other.block_size);
  for (int i = 0; i < other.block_num; ++i) {
    const int repeat_num = i == 0
        ? other.repeat_block_num[i]
        : other.repeat_block_num[i] - other.repeat_block_num[i - 1];
    const int block_size = other.remainder_num[i] == 0
        ? repeat_num * other.block_size
        : (repeat_num - 1) * other.block_size + other.remainder_num[i];
    printf(
        "\nNum size: %d, accumulate block_repeat_num: %d, \
      current block_repeat_num: %d, block_remain_num: %d",
        block_size,
        other.repeat_block_num[i],
        repeat_num,
        other.remainder_num[i]);
    printf("Address info: ");
    for (int j = 0; j < depth; ++j) {
      printf("%p ", other.address_array[i][j]);
    }
  }
}

template <int maxBlockNum, int depth>
std::ostream& operator<<(
    std::ostream& os,
    const BlockInfoWithTensorScalarList<maxBlockNum, depth>& other) {
  os << other.block_info_container;
  os << "\nScalar tensor list of block part address info: ";
  for (int i = 0; i < other.block_info_container.total_tensor_num; ++i) {
    os << other.scalar_tensor_list[i] << " ";
  }
  return os;
}

template <int maxBlockNum, int depth>
std::ostream& operator<<(
    std::ostream& os,
    const BlockInfoWithIndexList<maxBlockNum, depth>& other) {
  os << other.block_info_container;
  os << "\nScalar tensor list of block part address info: ";
  for (int i = 0; i < other.block_info_container.total_tensor_num; ++i) {
    os << other.index_list[i] << " ";
  }
  return os;
}

} // namespace torch_mlu::bangcommon