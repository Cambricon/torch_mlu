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

#include "aten/mluop/mluopUtils.h"
#include <string>

namespace torch_mlu {

mluOpDataType_t getMluOpDataType(const caffe2::TypeMeta& data_type) {
  static const std::map<std::string, mluOpDataType_t> mapping_type = {
      {std::string("c10::Half"), MLUOP_DTYPE_HALF},
      {std::string("float"), MLUOP_DTYPE_FLOAT},
      {std::string("double"), MLUOP_DTYPE_DOUBLE},
      {std::string("int8"), MLUOP_DTYPE_INT8},
      {std::string("signed char"), MLUOP_DTYPE_INT8},
      {std::string("short int"), MLUOP_DTYPE_INT16},
      {std::string("short"), MLUOP_DTYPE_INT16},
      {std::string("int"), MLUOP_DTYPE_INT32},
      {std::string("long int"), MLUOP_DTYPE_INT64},
      {std::string("long"), MLUOP_DTYPE_INT64},
      {std::string("unsigned char"), MLUOP_DTYPE_UINT8},
      {std::string("bool"), MLUOP_DTYPE_BOOL},
      {std::string("c10::complex<c10::Half>"), MLUOP_DTYPE_COMPLEX_HALF},
      {std::string("c10::complex<float>"), MLUOP_DTYPE_COMPLEX_FLOAT},
      {std::string("c10::BFloat16"), MLUOP_DTYPE_BFLOAT16}};

  if (mapping_type.find(std::string(data_type.name())) != mapping_type.end()) {
    return mapping_type.find(std::string(data_type.name()))->second;
  }
  return MLUOP_DTYPE_INVALID;
}

mluOpDataType_t cnnlTypeToMluOpType(const cnnlDataType_t& data_type) {
  static const std::map<cnnlDataType_t, mluOpDataType_t> mapping_type = {
      {CNNL_DTYPE_HALF, MLUOP_DTYPE_HALF},
      {CNNL_DTYPE_FLOAT, MLUOP_DTYPE_FLOAT},
      {CNNL_DTYPE_DOUBLE, MLUOP_DTYPE_DOUBLE},
      {CNNL_DTYPE_INT8, MLUOP_DTYPE_INT8},
      {CNNL_DTYPE_INT16, MLUOP_DTYPE_INT16},
      {CNNL_DTYPE_INT32, MLUOP_DTYPE_INT32},
      {CNNL_DTYPE_INT64, MLUOP_DTYPE_INT64},
      {CNNL_DTYPE_UINT8, MLUOP_DTYPE_UINT8},
      {CNNL_DTYPE_BOOL, MLUOP_DTYPE_BOOL},
      {CNNL_DTYPE_COMPLEX_HALF, MLUOP_DTYPE_COMPLEX_HALF},
      {CNNL_DTYPE_COMPLEX_FLOAT, MLUOP_DTYPE_COMPLEX_FLOAT},
      {CNNL_DTYPE_BFLOAT16, MLUOP_DTYPE_BFLOAT16}};

  if (mapping_type.find(data_type) != mapping_type.end()) {
    return mapping_type.find(data_type)->second;
  }
  return MLUOP_DTYPE_INVALID;
}
} // namespace torch_mlu
