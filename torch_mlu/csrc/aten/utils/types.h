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

#include "c10/util/BFloat16.h"
#include "c10/util/Half.h"
#include "c10/util/typeid.h"
#include "c10/core/ScalarType.h"
#include "utils/Export.h"
#include "cnnl.h" // NOLINT
#include "cnrt.h" // NOLINT

namespace torch_mlu {

TORCH_MLU_API cnnlDataType_t getCnnlDataType(const caffe2::TypeMeta& data_type);

TORCH_MLU_API cnnlDataType_t getCnnlDataType(const at::ScalarType& data_type);

TORCH_MLU_API at::ScalarType cnnlType2ScalarType(cnnlDataType_t cnnl_dtype);

TORCH_MLU_API cnrtDataType_t cnnlType2CnrtType(cnnlDataType_t cnnl_data_type);

TORCH_MLU_API cnrtDataType_V2_t
cnnlType2CnrtType_V2(cnnlDataType_t cnnl_data_type);

TORCH_MLU_API size_t getCnnlTypeSize(cnnlDataType_t cnnl_data_type);

/**
 * Note [CPPTypeToCNRTType]
 *
 * CPPTypeToCNRTType is to convert pytorch cpp type to cnrt type.
 * And There also conver 64bit cpp type to 32bit cpp type. And now only
 * support double, float, c10::Half and c10::BFloat16.
 *
 * MLUOpMathType:
 * @brief get pytorch cpp and return cnrt type
 * @param data_type: pytorch cpp type
 * @return cnrt type
 *
 * Usage:
 * cnrtDataType_V2_t value = CPPTypeToCNRTTypeValue<ori_type>::value;
 *
 */

template <typename scalar_t>
struct CPPTypeToCNRTTypeValue {};

// Promote 16bit floating type to 32bit floating type.
template <>
struct CPPTypeToCNRTTypeValue<at::Half> {
  static constexpr cnrtDataType_V2_t value = cnrtDataType_V2_t::cnrtHalf;
};
template <>
struct CPPTypeToCNRTTypeValue<at::BFloat16> {
  static constexpr cnrtDataType_V2_t value = cnrtDataType_V2_t::cnrtBfloat;
};
template <>
struct CPPTypeToCNRTTypeValue<float> {
  static constexpr cnrtDataType_V2_t value = cnrtDataType_V2_t::cnrtFloat;
};
template <>
struct CPPTypeToCNRTTypeValue<double> {
  static constexpr cnrtDataType_V2_t value = cnrtDataType_V2_t::cnrtFloat;
};

template <typename scalar_t>
inline constexpr cnrtDataType_V2_t CPPTypeToCNRTTypeValue_v =
    CPPTypeToCNRTTypeValue<scalar_t>::value;

} // namespace torch_mlu
