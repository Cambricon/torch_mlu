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

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/all.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "aten/utils/tensor_util.h"
#include "cncl.h" // NOLINT
#include "framework/core/MLUStream.h"
#include "Utils.h"

#define C10D_CNCL_CHECK(cmd, failure_reason)                              \
  do {                                                                    \
    cnclResult_t error = cmd;                                             \
    if (error != CNCL_RET_SUCCESS) {                                      \
      std::string err = "CNCL error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) + ", " +                               \
          getCnclErrorDetailStr(error, failure_reason);                   \
      TORCH_CHECK(false, err);                                            \
    }                                                                     \
  } while (0)

#define C10D_CNCL_ASSERT(cmd)                 \
  do {                                        \
    cnclResult_t res = cmd;                   \
    if (res != CNCL_RET_SUCCESS) {            \
      std::string err = cnclGetErrorStr(res); \
      fprintf(                                \
          stderr,                             \
          "CNCL error in: %s:%d, %s\n",       \
          __FILE__,                           \
          __LINE__,                           \
          err.c_str());                       \
      abort();                                \
    }                                         \
  } while (0)

// Provides additional detail into CNCL error codes based on when these are
// thrown in the CNCL codebase.
TORCH_MLU_API std::string getCnclErrorDetailStr(
    cnclResult_t error,
    c10::optional<std::string> process_group_failure_reason = c10::nullopt);

namespace torch_mlu {

namespace cncl::detail {

void all2all(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream);

void all2all_single_unequal_split(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType _type,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream);

void gather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream,
    int32_t root = 0);

void scatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream,
    int32_t root = 0);

} // namespace cncl::detail

TORCH_MLU_API std::unordered_map<std::string, MLUStream> getCnclStream(
    const DeviceIndex& device_index);

void updateCnclStream(const cnclCliqueId* cncl_id, MLUStream cncl_stream);

void clearCnclStream(const cnclCliqueId* cncl_id);

std::mutex* getFreeMutex();

} // namespace torch_mlu
