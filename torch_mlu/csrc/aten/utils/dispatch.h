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

#include <ATen/Dispatch.h>

namespace torch_mlu {

#define AT_DISPATCH_CASE_MLU(enum_type, scalar_type, ...) \
  case enum_type: {                                       \
    AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type);          \
    using scalar_t = scalar_type;                         \
    return __VA_ARGS__();                                 \
  }

#define AT_DISPATCH_MLU_FLOAT_HALF_AND_INT(TYPE, NAME, ...)                  \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    constexpr const char* at_dispatch_name = NAME;                           \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__)      \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", c10::toString(TYPE), "'"); \
    }                                                                        \
  }()

#define AT_DISPATCH_MLU_FLOAT_AND_INT(TYPE, NAME, ...)                       \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    constexpr const char* at_dispatch_name = NAME;                           \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)       \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", c10::toString(TYPE), "'"); \
    }                                                                        \
  }()

#define AT_DISPATCH_MLU_FLOAT_HALF_INT_AND_BFLOAT16(TYPE, NAME, ...)         \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    constexpr const char* at_dispatch_name = NAME;                           \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__)      \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)       \
      AT_DISPATCH_CASE_MLU(                                                  \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)               \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", c10::toString(TYPE), "'"); \
    }                                                                        \
  }()

#define AT_DISPATCH_MLU_FLOAT_HALF_INT_BOOL_AND_BFLOAT16(TYPE, NAME, ...)    \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    constexpr const char* at_dispatch_name = NAME;                           \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Bool, bool, __VA_ARGS__)          \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__)      \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)       \
      AT_DISPATCH_CASE_MLU(                                                  \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)               \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", c10::toString(TYPE), "'"); \
    }                                                                        \
  }()

#define AT_DISPATCH_MLU_FLOAT_HALF_INT_COMPLEX_AND_BFLOAT16(TYPE, NAME, ...) \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    constexpr const char* at_dispatch_name = NAME;                           \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__)      \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)       \
      AT_DISPATCH_CASE_MLU(                                                  \
          at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__)    \
      AT_DISPATCH_CASE_MLU(                                                  \
          at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(                                                  \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)               \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", c10::toString(TYPE), "'"); \
    }                                                                        \
  }()

#define AT_DISPATCH_ALL_MLU_TYPES_AND_HALF(TYPE, NAME, ...)             \
  [&] {                                                                 \
    constexpr const char* at_dispatch_name = NAME;                      \
    switch (TYPE) {                                                     \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__) \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)  \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }                                                                   \
  }()
// only for cnnl fill and pad op
#define AT_DISPATCH_MLU_TENSOR_SCLAER_TYPES(TYPE, NAME, ...)            \
  [&] {                                                                 \
    constexpr const char* at_dispatch_name = NAME;                      \
    switch (TYPE) {                                                     \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Byte, uint8_t, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__) \
      AT_DISPATCH_CASE_MLU(                                             \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)          \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Bool, bool, __VA_ARGS__)     \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }                                                                   \
  }()

#define AT_DISPATCH_MLU_FLOAT_AND_HALF(TYPE, NAME, ...)                 \
  [&] {                                                                 \
    constexpr const char* at_dispatch_name = NAME;                      \
    switch (TYPE) {                                                     \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__) \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)   \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }                                                                   \
  }()

#define AT_DISPATCH_MLU_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)        \
  [&] {                                                                 \
    constexpr const char* at_dispatch_name = NAME;                      \
    switch (TYPE) {                                                     \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__) \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }                                                                   \
  }()

#define AT_DISPATCH_MLU_FLOATING_TYPES_HALF_AND_BFLOAT16(TYPE, NAME, ...) \
  [&] {                                                                   \
    constexpr const char* at_dispatch_name = NAME;                        \
    switch (TYPE) {                                                       \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)    \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)     \
      AT_DISPATCH_CASE_MLU(                                               \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)            \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__)   \
      default:                                                            \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");   \
    }                                                                     \
  }()

// only for cnnl_cumsum op
#define AT_DISPATCH_MLU_INT_AND_FLOAT_TYPES(TYPE, NAME, ...)            \
  [&] {                                                                 \
    constexpr const char* at_dispatch_name = NAME;                      \
    switch (TYPE) {                                                     \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__) \
      AT_DISPATCH_CASE_MLU(                                             \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)          \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)  \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }                                                                   \
  }()

// only for tri
#define AT_DISPATCH_ALL_MLU_TYPES_AND_HALF_AND_BFLOAT16_EXCEPT_UINT8(   \
    TYPE, NAME, ...)                                                    \
  [&] {                                                                 \
    constexpr const char* at_dispatch_name = NAME;                      \
    switch (TYPE) {                                                     \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Char, int8_t, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__) \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Bool, bool, __VA_ARGS__)     \
      AT_DISPATCH_CASE_MLU(                                             \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)          \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }                                                                   \
  }()
#define AT_DISPATCH_ALL_MLU_TYPES_AND_HALF_AND_BFLOAT16(TYPE, NAME, ...) \
  [&] {                                                                  \
    constexpr const char* at_dispatch_name = NAME;                       \
    switch (TYPE) {                                                      \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Byte, uint8_t, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Char, int8_t, __VA_ARGS__)    \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Short, int16_t, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)    \
      AT_DISPATCH_CASE_MLU(                                              \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)           \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)    \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)   \
      default:                                                           \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");  \
    }                                                                    \
  }()

// only for where
#define AT_DISPATCH_ALL_MLU_TYPES_AND_HALF_AND_BFLOAT16_EXCEPT_UINT8_AND_BOOL( \
    TYPE, NAME, ...)                                                           \
  [&] {                                                                        \
    constexpr const char* at_dispatch_name = NAME;                             \
    switch (TYPE) {                                                            \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Char, int8_t, __VA_ARGS__)          \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Short, int16_t, __VA_ARGS__)        \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)          \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)          \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)         \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)         \
      AT_DISPATCH_CASE_MLU(                                                    \
          at::ScalarType::BFloat16, at::BFloat16, __VA_ARGS__)                 \
      default:                                                                 \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");        \
    }                                                                          \
  }()

// only for unique_consecutive
#define AT_DISPATCH_MLU_ALL_TYPES_AND_HALF_EXCEPT_INT8_AND_UINT8(       \
    TYPE, NAME, ...)                                                    \
  [&] {                                                                 \
    constexpr const char* at_dispatch_name = NAME;                      \
    switch (TYPE) {                                                     \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Half, at::Half, __VA_ARGS__) \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Float, float, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Double, float, __VA_ARGS__)  \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Int, int32_t, __VA_ARGS__)   \
      AT_DISPATCH_CASE_MLU(at::ScalarType::Long, int64_t, __VA_ARGS__)  \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }                                                                   \
  }()
} // namespace torch_mlu
