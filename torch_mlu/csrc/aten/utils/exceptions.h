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
#include <stdio.h>
#include "c10/util/Exception.h"
#include "ATen/TensorUtils.h"
#include "c10/macros/Macros.h"
#include "cnrt.h" //NOLINT
#include "cndev.h" // NOLINT
#ifdef USE_MLUOP
#include "mlu_op.h" // NOLINT
#endif

#define TORCH_BANGC_CHECK(EXPR)                                   \
  do {                                                            \
    cnrtRet_t __err = EXPR;                                       \
    if (C10_UNLIKELY(__err != CNRT_RET_SUCCESS)) {                \
      TORCH_CHECK(false, "CNRT error: ", cnrtGetErrorStr(__err)); \
    }                                                             \
  } while (0);

// TORCH_CHECK will throw a c10 Error, and TORCH_WARN just print
// warning information
#define TORCH_MLU_CHECK(cond, ...)       \
  do {                                   \
    if (!(cond)) {                       \
      TORCH_CHECK(false, ##__VA_ARGS__); \
    }                                    \
  } while (0);

// C10_UNLIKELY is:
// #define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
// The __builtin_expect is a macro function provided in GCC, a popular
// C compiler. It is used to provide the compiler with branch prediction
// information, which can be beneficial for optimizing performance-critical
// code.
#define TORCH_CNRT_CHECK(EXPR)                                    \
  do {                                                            \
    cnrtRet_t __err = EXPR;                                       \
    if (C10_UNLIKELY(__err != cnrtSuccess)) {                     \
      auto error_unused C10_UNUSED = cnrtGetLastError();          \
      (void)error_unused;                                         \
      TORCH_CHECK(false, "CNRT error: ", cnrtGetErrorStr(__err)); \
    }                                                             \
  } while (0);

#define TORCH_BANGC_KERNEL_LAUNCH_CHECK() TORCH_BANGC_CHECK(cnrtGetLastError())

#define TORCH_CNRT_WARN(EXPR)                               \
  do {                                                      \
    cnrtRet_t __err = EXPR;                                 \
    if (C10_UNLIKELY(__err != cnrtSuccess)) {               \
      auto error_unused C10_UNUSED = cnrtGetLastError();    \
      (void)error_unused;                                   \
      TORCH_WARN("CNRT warning: ", cnrtGetErrorStr(__err)); \
    }                                                       \
  } while (0);

#define TORCH_CNNL_CHECK(EXPR)                                        \
  do {                                                                \
    cnnlStatus_t status = EXPR;                                       \
    if (C10_UNLIKELY(status != CNNL_STATUS_SUCCESS)) {                \
      TORCH_CHECK(false, "CNNL error: ", cnnlGetErrorString(status)); \
    }                                                                 \
  } while (0);

#define TORCH_CNDEV_CHECK(EXPR)                                         \
  do {                                                                  \
    cndevRet_t status = EXPR;                                           \
    if (C10_UNLIKELY(status != CNDEV_SUCCESS)) {                        \
      auto error_unused C10_UNUSED = cndevGetLastError();               \
      (void)error_unused;                                               \
      TORCH_CHECK(false, "CNDEV error: ", cndevGetErrorString(status)); \
    }                                                                   \
  } while (0);

#ifdef USE_MLUOP
#define TORCH_MLUOP_CHECK(EXPR)                                          \
  do {                                                                   \
    mluOpStatus_t status = EXPR;                                         \
    if (C10_UNLIKELY(status != MLUOP_STATUS_SUCCESS)) {                  \
      TORCH_CHECK(false, "MLUOPS error: ", mluOpGetErrorString(status)); \
    }                                                                    \
  } while (0);
#endif

#define TORCH_CNDRV_CHECK(EXPR)                                       \
  do {                                                                \
    CNresult __err = EXPR;                                            \
    if (C10_UNLIKELY(__err != CN_SUCCESS)) {                          \
      const char* err_str;                                            \
      CNresult get_error_str_err = cnGetErrorString(__err, &err_str); \
      if (get_error_str_err != CN_SUCCESS) {                          \
        TORCH_CHECK(false, "CNDRV error: unknow error.");             \
      } else {                                                        \
        TORCH_CHECK(false, "CNDRV error: ", err_str);                 \
      }                                                               \
    }                                                                 \
  } while (0)

// Indicates that a CNRT error is handled in a non-standard way
#define TORCH_CNRT_ERROR_HANDLED(EXPR) EXPR
