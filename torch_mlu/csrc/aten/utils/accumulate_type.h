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
#include "c10/core/ScalarType.h"

namespace torch_mlu {

/**
 * Note [Convert64BitTo32Bit]
 * Only several operators support 64bit data type in CNNL kernel, not like
 * pytorch almost all operators support 64bit data type. So we need to convert
 * 64bit data type to 32bit data type. There are two cases need to convert 64bit
 * data type to 32bit data type: Tensor and Scalar. For Tensor: CATCH cast 64bit
 * tensor to 32bit tensor in H2D or D2H copy. For Scalar: CATCH also need to
 * cast 64bit scalar to 32bit scalar. And this is not easy to handle in common
 * function, so each operator with scalar need to handle this.
 *
 * Note for a special case:
 * If a CNNL kernel support 64bit tensor and with scalar, there are also two
 * strategies to handle this: 1) Call setCnnlDtype API to set the real CNNL data
 * type of the tensor and using 64bit scalar value to run kernel. 2) Just treat
 * the 64bit tensor as 32bit tensor, and cast 64bit scalar value to 32bit scalar
 * value. As discussed with Mr. Kong, we chose option 2 until now.
 */

template <typename scalar_t>
struct Convert64BitTo32Bit {
  using type = scalar_t;
};

template <>
struct Convert64BitTo32Bit<double> {
  using type = float;
};
template <>
struct Convert64BitTo32Bit<c10::complex<double>> {
  using type = c10::complex<float>;
};

template <typename scalar_t>
using Convert64BitTo32Bit_t = typename Convert64BitTo32Bit<scalar_t>::type;

/**
 * Note [MLUAccumulateType]
 *
 * MLUAccumulateType is different with AccumulateType in pytorch. And there are
 * three significate difference: 1) AccumulateType has two template parameter,
 * one is cpp type, another is bool value for gpu or cpu. In MLUAccumulateType,
 * there only one parameter is cpp type. 2) AccumulateType is support 64bit cpp
 * type, like double to double, like float to double. But MLUAccumulateType only
 * using 32bit cpp type to replace 64bit cpp type, like double to float. 3)
 * AccumulateType will keep bool type, but MLUAccumulateType will convert bool
 * type to int type, cause a lot of mlu ops is not support bool, like add, sub
 * and mul.
 *
 * MLUAccumulateType:
 * @brief get ScalarType and return cpp type
 * @param data_type: original cpp type
 * @return cpp type
 *
 * Usage:
 * using new_type = MLUAccumulateType_t<ori_type>;
 *
 * AccumulateType in pytorch:
 * https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/AccumulateType.h
 */

template <typename scalar_t>
struct MLUAccumulateType : public Convert64BitTo32Bit<scalar_t> {};

// Promote less or equal 16bit to 32bit.
template <>
struct MLUAccumulateType<at::Half> {
  using type = float;
};
template <>
struct MLUAccumulateType<at::BFloat16> {
  using type = float;
};
template <>
struct MLUAccumulateType<c10::complex<at::Half>> {
  using type = c10::complex<float>;
};
template <>
struct MLUAccumulateType<int8_t> {
  using type = int;
};
template <>
struct MLUAccumulateType<uint8_t> {
  using type = int;
};
template <>
struct MLUAccumulateType<int16_t> {
  using type = int;
};
template <>
struct MLUAccumulateType<char> {
  using type = int;
};
template <>
struct MLUAccumulateType<bool> {
  using type = int;
};

template <typename scalar_t>
using MLUAccumulateType_t = typename MLUAccumulateType<scalar_t>::type;

/**
 * Note [MLUOpMathType]
 *
 * MLUOpMathType is different with AccumulateType in pytorch. And there are only
 * one significate difference is: OpMathType not change 64bit cpp type, but
 * MLUOpMathType using 32bit cpp type to replace 64bit cpp type, like double to
 * float. Others are same.
 *
 * MLUOpMathType:
 * @brief get ScalarType and return cpp type
 * @param data_type: original cpp type
 * @return cpp type
 *
 * Usage:
 * using new_type = MLUOpMathType_t<ori_type>;
 *
 * OpMathType in pytorch:
 * https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/OpMathType.h
 */

template <typename scalar_t>
struct MLUOpMathType : public Convert64BitTo32Bit<scalar_t> {};

// Promote 16bit floating type to 32bit floating type.
template <>
struct MLUOpMathType<at::Half> {
  using type = float;
};
template <>
struct MLUOpMathType<at::BFloat16> {
  using type = float;
};
template <>
struct MLUOpMathType<c10::complex<at::Half>> {
  using type = c10::complex<float>;
};

template <typename scalar_t>
using MLUOpMathType_t = typename MLUOpMathType<scalar_t>::type;

} // namespace torch_mlu
