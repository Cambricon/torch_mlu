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

#include <map>
#include <memory>
#include "aten/operators/cnnl/cnnlOpParams.h"

namespace torch_mlu {

namespace {
/**
 * Note [OP register in opParamsMap]
 * ~~~~~~~~~~~~~~~~
 * opParamsMap is using to store the params of CNNL op, and the key is the name
 * of op, the value is the params of op.
 *
 * There are too many ops need to register to opParamsMap, to reduce the count
 * num of opParamsMap. We using most common op params as default. The most
 * common op is the op which only support float, half type, and don't support
 * mixed input types. So don't register those ops. example: static
 * RegisterCnnlOpParams default_params_register("default", CnnlOpParams());
 *
 * For some ops which support float, half, int type, and don't support mixed
 * input types. we using macro
 * OP_SUPPORT_FLOAT_HALF_INT_DTYPE_AND_WITHOUT_MIX_INPUT to reduce redundant
 * code. example:
 *   #define OP_SUPPORT_FLOAT_HALF_INT_DTYPE_AND_WITHOUT_MIX_INPUT(_) \
 *                                 _(div, f_h_i_types, true)         \
 *
 * For others which support more compute data types and don't support mixed
 * input types, we register them one by one, and using macro
 * REGISTER_PARAMS_WITHOUT_MIXED_INPUT. example:
 *   REGISTER_PARAMS_WITHOUT_MIXED_INPUT(abs,
 *        std::vector<at::ScalarType>({at::kHalf, at::kFloat,
 *                                     at::kComplexFloat}), false)
 *
 * For some ops which support mixed input types, we register them one by one,
 * and using macro REGISTER_PARAMS_WITH_MIXED_INPUT. example:
 * #define OPTENSOR_OP_REGISTER(_) \
 *   _(add, f_h_i_types, true, optensor_input_mix_types)
 *
 * OPTENSOR_OP_REGISTER(REGISTER_PARAMS_WITH_MIXED_INPUT)
 * #undef OPTENSOR_OP_REGISTER
 *
 * Macro function SET_CNNL_OP_PARAMS_VARIABLE is for setting the
 * allow_cpu_scalar_ and allow_strided_memory_ variable of CnnlOpParams.
 *
 * opParamsMap is used by MLUTensorIterator to get ops info. And getCnnlOpParams
 * api is the only way to get op params from opParamsMap.
 *
 * Note [OP name in opParamsMap]
 * ~~~~~~~~~~~~~~~~
 *
 */
static std::map<std::string, std::unique_ptr<torch_mlu::CnnlOpParams>>
    opParamsMap;

} // anonymous namespace

// This class is only for register CNNL op params.
class RegisterCnnlOpParams {
 public:
  AT_DISALLOW_COPY_AND_ASSIGN(RegisterCnnlOpParams);
  explicit RegisterCnnlOpParams(
      const std::vector<at::ScalarType>& support_types,
      const std::string& name,
      bool implicit_type_convert = false,
      bool allow_strided_memory = false,
      bool allow_cpu_scalar = false,
      bool close_types_check = false,
      OpMluMemoryFormatStrategy memory_format_strategy =
          OpMluMemoryFormatStrategy::Default,
      bool close_tensor_broadcast = false,
      const std::vector<std::vector<at::ScalarType>>& mixed_types = {},
      const at::ScalarType& static_type = at::ScalarType::Undefined) {
    TORCH_CHECK(
        opParamsMap.find(name) == opParamsMap.end(),
        "Cnnl op params of ",
        name,
        " has been registered.");
    opParamsMap[name] = std::make_unique<torch_mlu::CnnlOpParams>(
        support_types,
        name,
        implicit_type_convert,
        allow_strided_memory,
        allow_cpu_scalar,
        close_types_check,
        memory_format_strategy,
        close_tensor_broadcast);
    if (mixed_types.size() != 0) {
      opParamsMap[name]->setSupportMixedType(mixed_types);
    }
    if (static_type != at::ScalarType::Undefined) {
      opParamsMap[name]->setStaticType(static_type);
    }
  }
};

namespace details {
void CnnlOpTypeInfo::setSupportMixedType(
    const std::vector<std::vector<at::ScalarType>>& mixed_types_list) {
  TORCH_CHECK(
      this->support_mixed_types_.size() == 0,
      "mixed_types_list is setted in CnnlOpTypeInfo before.");
  TORCH_CHECK(
      mixed_types_list.size() > 0, "Mix types list size is less than 0.");
  const int types_num = mixed_types_list[0].size();
  TORCH_CHECK(types_num >= 3, "Mix types size is greater or equal than 3.");
  // Calculate hash code of each mixed_types_list and store it in
  // unordered_map.
  // Why need add output type in hash value?
  // Because some ops has different output types when using same mixed input
  // types.
  for (const auto& item : mixed_types_list) {
    TORCH_CHECK(
        item.size() == types_num,
        "Each item size of mixed_types_list are not equal.");
    const int64_t hash_value = getScalarTypeListHashWithOutputType(item);
    // error will report when import torch_mlu when same types list in
    // mixed input types list.
    TORCH_CHECK(
        this->support_mixed_types_.find(hash_value) ==
            this->support_mixed_types_.end(),
        "mixed_types_list has same record.")
    // Only support one output now, and it's easy to support more output
    // types for future.
    this->support_mixed_types_[hash_value] = item.back();
  }
}
} // namespace details

const CnnlOpParams& getCnnlOpParams(const std::string& name) {
  // (TODO) need to add op name mapping in this function.
  if (opParamsMap.find(name) == opParamsMap.end()) {
    return *(opParamsMap["default"]);
  }
  return *(opParamsMap[name]);
}

namespace {
// TORCH_MLU double/long tensor is real float/int tensor, so add them to support
// types.

static const std::vector<at::ScalarType> f_types({at::kFloat, at::kDouble});

static const std::vector<at::ScalarType> f_h_types(
    {at::kFloat, at::kHalf, at::kDouble});

static const std::vector<at::ScalarType> f_bf_h_types(
    {at::kFloat, at::kHalf, at::kBFloat16, at::kDouble});

static const std::vector<at::ScalarType> f_h_i_types(
    {at::kFloat, at::kHalf, at::kInt, at::kDouble, at::kLong});

static const std::vector<at::ScalarType> f_bf_h_i_types(
    {at::kFloat, at::kHalf, at::kBFloat16, at::kInt, at::kDouble, at::kLong});

static const std::vector<at::ScalarType> f_bf_h_i_s_types(
    {at::kFloat,
     at::kHalf,
     at::kBFloat16,
     at::kDouble,
     at::kChar,
     at::kShort,
     at::kInt,
     at::kLong});

static const std::vector<at::ScalarType> f_bf_h_i_c_types(
    {at::kFloat,
     at::kHalf,
     at::kBFloat16,
     at::kInt,
     at::kDouble,
     at::kLong,
     c10::ScalarType::ComplexHalf,
     c10::ScalarType::ComplexFloat,
     c10::ScalarType::ComplexDouble});

static const std::vector<at::ScalarType> f_bf_h_i_c_types_without_chalf(
    {at::kFloat,
     at::kHalf,
     at::kBFloat16,
     at::kInt,
     at::kDouble,
     at::kLong,
     c10::ScalarType::ComplexFloat,
     c10::ScalarType::ComplexDouble});

static const std::vector<at::ScalarType>
    all_support_types_without_bfloat16_and_complex(
        {at::kChar,
         at::kShort,
         at::kByte,
         at::kInt,
         at::kBool,
         at::kFloat,
         at::kHalf,
         at::kDouble,
         at::kLong});

static const std::vector<at::ScalarType> all_support_types_without_complex(
    {at::kFloat,
     at::kHalf,
     at::kBFloat16,
     at::kChar,
     at::kShort,
     at::kByte,
     at::kInt,
     at::kBool,
     at::kDouble,
     at::kLong});

static const std::vector<at::ScalarType> all_support_types_without_chalf(
    {at::kFloat,
     at::kHalf,
     at::kBFloat16,
     at::kChar,
     at::kShort,
     at::kByte,
     at::kInt,
     at::kBool,
     at::kDouble,
     at::kLong,
     at::kComplexFloat,
     at::kComplexDouble});

static const std::vector<at::ScalarType> all_support_types_without_chalf_bool(
    {at::kFloat,
     at::kHalf,
     at::kBFloat16,
     at::kChar,
     at::kShort,
     at::kByte,
     at::kInt,
     at::kDouble,
     at::kLong,
     at::kComplexFloat,
     at::kComplexDouble});

// Note1: need to add input types first and then add output type;
// Note2: Only support one output tensor right now, and doesn't find
// any op in pytorch.
static const std::vector<std::vector<at::ScalarType>> optensor_input_mix_types(
    {{at::kFloat, at::kHalf, at::kFloat},
     {at::kFloat, at::kHalf, at::kHalf},
     {at::kFloat, at::kInt, at::kFloat},
     {at::kDouble, at::kInt, at::kFloat},
     {at::kDouble, at::kInt, at::kDouble},
     {at::kHalf, at::kInt, at::kHalf},
     {at::kFloat, at::kBFloat16, at::kFloat},
     {at::kFloat, at::kBFloat16, at::kBFloat16},
     {at::kDouble, at::kBFloat16, at::kBFloat16},
     {at::kDouble, at::kBFloat16, at::kFloat},
     {at::kDouble, at::kBFloat16, at::kDouble},
     {at::kInt, at::kLong, at::kLong},
     {at::kBFloat16, at::kInt, at::kBFloat16}});

static const std::vector<std::vector<at::ScalarType>> logic_input_mix_types(
    {{at::kFloat, at::kInt, at::kBool}, {at::kDouble, at::kInt, at::kBool}});

static const std::vector<std::vector<at::ScalarType>>
    masked_scale_support_mixed_types{
        {at::kFloat, at::kByte, at::kFloat},
        {at::kDouble, at::kByte, at::kDouble},
        {at::kHalf, at::kByte, at::kHalf},
        {at::kBFloat16, at::kByte, at::kBFloat16},
        {at::kFloat, at::kBool, at::kFloat},
        {at::kDouble, at::kBool, at::kDouble},
        {at::kHalf, at::kBool, at::kHalf},
        {at::kBFloat16, at::kBool, at::kBFloat16}};

static const std::vector<std::vector<at::ScalarType>> pow_support_mixed_types{
    {at::kFloat, at::kShort, at::kFloat},
    {at::kDouble, at::kShort, at::kDouble},
    {at::kHalf, at::kShort, at::kHalf},
    {at::kBFloat16, at::kShort, at::kBFloat16}};

// Register default op params.
static RegisterCnnlOpParams default_params_register(
    std::vector<at::ScalarType>(
        {at::kFloat, at::kHalf, at::kBFloat16, at::kDouble}),
    "default",
    false);

#define REGISTER_WITH_SUGGEST_MEMORY_FORMAT_SOFT(       \
    opName, supportTypes, typeConvert, checkTypes)      \
  static RegisterCnnlOpParams opName##_params_register( \
      supportTypes,                                     \
      #opName,                                          \
      typeConvert,                                      \
      false,                                            \
      false,                                            \
      checkTypes,                                       \
      OpMluMemoryFormatStrategy::SuggestMemoryFormatSoft);

#define REGISTER_WITH_SUGGEST_MEMORY_FORMAT_HARD(       \
    opName, supportTypes, typeConvert, checkTypes)      \
  static RegisterCnnlOpParams opName##_params_register( \
      supportTypes,                                     \
      #opName,                                          \
      typeConvert,                                      \
      false,                                            \
      false,                                            \
      checkTypes,                                       \
      OpMluMemoryFormatStrategy::SuggestMemoryFormatHard);

// Register op params without set input mixed types.
#define REGISTER_PARAMS_WITHOUT_MIXED_INPUT(            \
    opName, supportTypes, implicitConvert)              \
  static RegisterCnnlOpParams opName##_params_register( \
      supportTypes, #opName, implicitConvert);

// Register op params with set input mixed types.
#define REGISTER_PARAMS_WITH_MIXED_INPUT(               \
    opName, supportTypes, implicitConvert, mixTypeList) \
  static RegisterCnnlOpParams opName##_params_register( \
      supportTypes,                                     \
      #opName,                                          \
      implicitConvert,                                  \
      false,                                            \
      false,                                            \
      false,                                            \
      OpMluMemoryFormatStrategy::Default,               \
      false,                                            \
      mixTypeList);

// Register op params with set input mixed types and static data type.
#define REGISTER_PARAMS_WITH_MIXED_INPUT_AND_STATIC_TYPE(           \
    opName, supportTypes, staticType, implicitConvert, mixTypeList) \
  static RegisterCnnlOpParams opName##_params_register(             \
      supportTypes,                                                 \
      #opName,                                                      \
      implicitConvert,                                              \
      false,                                                        \
      false,                                                        \
      false,                                                        \
      OpMluMemoryFormatStrategy::Default,                           \
      false,                                                        \
      mixTypeList,                                                  \
      staticType);

// Register index op params without implicit type convert. Also those ops have
// different input types and only support contiguous.
#define REGISTER_INDEX_OP_PARAM(                                          \
    opName, supportTypes, closeTypesCheck, memoryFormat, isNeedBroadcast) \
  static RegisterCnnlOpParams opName##_params_register(                   \
      supportTypes,                                                       \
      #opName,                                                            \
      false,                                                              \
      false,                                                              \
      false,                                                              \
      closeTypesCheck,                                                    \
      memoryFormat,                                                       \
      isNeedBroadcast);

// Register where op params without implicit type convert.
#define REGISTER_WHERE_OP_PARAM(opName, supportTypes, closeTypesCheck) \
  static RegisterCnnlOpParams opName##_params_register(                \
      supportTypes,                                                    \
      #opName,                                                         \
      false,                                                           \
      false,                                                           \
      false,                                                           \
      closeTypesCheck,                                                 \
      OpMluMemoryFormatStrategy::SuggestMemoryFormatSoft);

// Register polar op params without implicit type convert.
#define REGISTER_POLAR_OP_PARAM(opName, supportTypes, closeTypesCheck) \
  static RegisterCnnlOpParams opName##_params_register(                \
      supportTypes, #opName, false, false, false, closeTypesCheck);

// Register op params with implicit type convert and memory_format.
#define REGISTER_REDUCE_OP_PARAM(                        \
    opName, supportTypes, memoryFormat, implicitConvert) \
  static RegisterCnnlOpParams opName##_params_register(  \
      supportTypes,                                      \
      #opName,                                           \
      implicitConvert,                                   \
      false,                                             \
      false,                                             \
      false,                                             \
      memoryFormat);

// Register op params with strided memory, but not support input mixed types.
#define REGISTER_STRIDED_OP_PARAM_WITHOUT_MIXED_INPUT(    \
    opName, supportTypes, implicitConvert, stridedMemory) \
  static RegisterCnnlOpParams opName##_params_register(   \
      supportTypes, #opName, implicitConvert, stridedMemory);

#define SUGGEST_MEMORY_FORMAT_SOFT_OP_REGISTER(_) \
  _(clamp, f_bf_h_i_types, true, false)

SUGGEST_MEMORY_FORMAT_SOFT_OP_REGISTER(REGISTER_WITH_SUGGEST_MEMORY_FORMAT_SOFT)
#undef SUGGEST_MEMORY_FORMAT_SOFT_OP_REGISTER

#define SUGGEST_MEMORY_FORMAT_HARD_OP_REGISTER(_) \
  _(cross, all_support_types_without_chalf_bool, false, false)

SUGGEST_MEMORY_FORMAT_HARD_OP_REGISTER(REGISTER_WITH_SUGGEST_MEMORY_FORMAT_HARD)
#undef SUGGEST_MEMORY_FORMAT_HARD_OP_REGISTER

// Register optensor op params with mixed input types.
#define OPTENSOR_OP_REGISTER(_)                                          \
  _(add, f_bf_h_i_c_types_without_chalf, true, optensor_input_mix_types) \
  _(sub, f_bf_h_i_c_types_without_chalf, true, optensor_input_mix_types) \
  _(mul, f_bf_h_i_c_types, true, optensor_input_mix_types)

OPTENSOR_OP_REGISTER(REGISTER_PARAMS_WITH_MIXED_INPUT)
#undef OPTENSOR_OP_REGISTER

// Register optensor op params with mixed input types.
#define FLOAT_HALF_BFLOAT16_REGISTER(_) \
  _(gelu, f_bf_h_types, false)          \
  _(gelu_backward, f_bf_h_types, false) \
  _(reciprocal, f_bf_h_types, false)

FLOAT_HALF_BFLOAT16_REGISTER(REGISTER_PARAMS_WITHOUT_MIXED_INPUT)
#undef FLOAT_HALF_BFLOAT16_REGISTER

// Register logic op params with mixed input types.
#define LOGIC_OP_REGISTER(_)           \
  _(gt,                                \
    all_support_types_without_complex, \
    at::kBool,                         \
    false,                             \
    logic_input_mix_types)             \
  _(lt,                                \
    all_support_types_without_complex, \
    at::kBool,                         \
    false,                             \
    logic_input_mix_types)             \
  _(ge,                                \
    all_support_types_without_complex, \
    at::kBool,                         \
    false,                             \
    logic_input_mix_types)             \
  _(le,                                \
    all_support_types_without_complex, \
    at::kBool,                         \
    false,                             \
    logic_input_mix_types)             \
  _(eq,                                \
    all_support_types_without_chalf,   \
    at::kBool,                         \
    false,                             \
    logic_input_mix_types)             \
  _(ne,                                \
    all_support_types_without_complex, \
    at::kBool,                         \
    false,                             \
    logic_input_mix_types)             \
  _(logical_and,                       \
    all_support_types_without_complex, \
    at::kBool,                         \
    false,                             \
    logic_input_mix_types)             \
  _(logical_or,                        \
    all_support_types_without_complex, \
    at::kBool,                         \
    false,                             \
    logic_input_mix_types)             \
  _(logical_xor,                       \
    all_support_types_without_complex, \
    at::kBool,                         \
    false,                             \
    logic_input_mix_types)             \
  _(logical_not,                       \
    all_support_types_without_complex, \
    at::kBool,                         \
    false,                             \
    logic_input_mix_types)

LOGIC_OP_REGISTER(REGISTER_PARAMS_WITH_MIXED_INPUT_AND_STATIC_TYPE)
#undef LOGIC_OP_REGISTER

#define POW_OP_REGISTER(_) \
  _(pow, all_support_types_without_complex, false, pow_support_mixed_types)

POW_OP_REGISTER(REGISTER_PARAMS_WITH_MIXED_INPUT)
#undef POW_OP_REGISTER

#define MASK_SCALE_OP_REGISTER(_) \
  _(_masked_scale, f_bf_h_types, true, masked_scale_support_mixed_types)

MASK_SCALE_OP_REGISTER(REGISTER_PARAMS_WITH_MIXED_INPUT)
#undef MASK_SCALE_OP_REGISTER

#define INDEX_OP_REGISTER(_)               \
  _(index,                                 \
    all_support_types_without_complex,     \
    true,                                  \
    OpMluMemoryFormatStrategy::Contiguous, \
    true);

INDEX_OP_REGISTER(REGISTER_INDEX_OP_PARAM)
#undef INDEX_OP_REGISTER

#define REDUCE_OP_REGISTER(_)                                             \
  _(mean, f_bf_h_types, OpMluMemoryFormatStrategy::Contiguous, true)      \
  _(nansum, f_bf_h_types, OpMluMemoryFormatStrategy::Contiguous, true)    \
  _(std_var, f_bf_h_types, OpMluMemoryFormatStrategy::Contiguous, true)   \
  _(sum,                                                                  \
    f_bf_h_i_c_types_without_chalf,                                       \
    OpMluMemoryFormatStrategy::Contiguous,                                \
    true)                                                                 \
  _(norm,                                                                 \
    all_support_types_without_complex,                                    \
    OpMluMemoryFormatStrategy::Contiguous,                                \
    true)                                                                 \
  _(max,                                                                  \
    all_support_types_without_complex,                                    \
    OpMluMemoryFormatStrategy::Contiguous,                                \
    true)                                                                 \
  _(max_all, f_bf_h_i_types, OpMluMemoryFormatStrategy::Contiguous, true) \
  _(min,                                                                  \
    all_support_types_without_complex,                                    \
    OpMluMemoryFormatStrategy::Contiguous,                                \
    true)                                                                 \
  _(min_all, f_bf_h_i_types, OpMluMemoryFormatStrategy::Contiguous, true) \
  _(argmax, f_bf_h_i_types, OpMluMemoryFormatStrategy::Contiguous, true)  \
  _(argmin, f_bf_h_i_types, OpMluMemoryFormatStrategy::Contiguous, true)  \
  _(amax, f_bf_h_i_types, OpMluMemoryFormatStrategy::Contiguous, true)    \
  _(amin, f_bf_h_i_types, OpMluMemoryFormatStrategy::Contiguous, true)    \
  _(any,                                                                  \
    all_support_types_without_complex,                                    \
    OpMluMemoryFormatStrategy::Contiguous,                                \
    true)                                                                 \
  _(all,                                                                  \
    all_support_types_without_complex,                                    \
    OpMluMemoryFormatStrategy::Contiguous,                                \
    true)                                                                 \
  _(prod, f_bf_h_i_types, OpMluMemoryFormatStrategy::Contiguous, true)

REDUCE_OP_REGISTER(REGISTER_REDUCE_OP_PARAM)
#undef REDUCE_OP_REGISTER

#define WHERE_OP_REGISTER(_) _(where, f_bf_h_i_s_types, true);

WHERE_OP_REGISTER(REGISTER_WHERE_OP_PARAM)
#undef WHERE_OP_REGISTER

#define POLAR_OP_REGISTER(_) _(polar, f_types, true);

POLAR_OP_REGISTER(REGISTER_POLAR_OP_PARAM)
#undef POLAR_OP_REGISTER

// Operator params for ops, which compute dtype is float, half, int. And those
// ops don't support mixed input types.
#define OP_SUPPORT_FLOAT_HALF_INT_DTYPE_AND_WITHOUT_MIX_INPUT(_) \
  _(div, f_bf_h_i_types, true)                                   \
  _(neg, f_bf_h_i_types, false)                                  \
  _(maximum, f_bf_h_i_types, true)                               \
  _(minimum, f_bf_h_i_types, true)                               \
  _(clamp_min, f_bf_h_i_types, true)                             \
  _(clamp_max, f_bf_h_i_types, true)                             \
  _(fmod, f_bf_h_i_types, true)                                  \
  _(floor_divide, f_bf_h_i_types, true)                          \
  _(remainder, f_bf_h_i_types, true)

OP_SUPPORT_FLOAT_HALF_INT_DTYPE_AND_WITHOUT_MIX_INPUT(
    REGISTER_PARAMS_WITHOUT_MIXED_INPUT)

#undef OP_SUPPORT_FLOAT_HALF_INT_DTYPE_AND_WITHOUT_MIX_INPUT

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    threshold,
    std::vector<at::ScalarType>(
        {at::kHalf,
         at::kFloat,
         at::kBFloat16,
         at::kByte,
         at::kChar,
         at::kShort,
         at::kInt,
         at::kDouble,
         at::kLong}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    bitwise_not,
    std::vector<at::ScalarType>(
        {at::kBool, at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    _s_where,
    std::vector<at::ScalarType>(
        {at::kBool,
         at::kHalf,
         at::kChar,
         at::kShort,
         at::kInt,
         at::kFloat,
         at::kDouble,
         at::kLong}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    abs,
    std::vector<at::ScalarType>(
        {at::kHalf,
         at::kFloat,
         at::kInt,
         at::kLong,
         at::kComplexFloat,
         at::kBFloat16,
         at::kDouble,
         at::kComplexDouble}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    angle,
    std::vector<at::ScalarType>(
        {at::kChar,
         at::kShort,
         at::kByte,
         at::kInt,
         at::kFloat,
         at::kComplexFloat,
         at::kLong,
         at::kDouble,
         at::kComplexDouble,
         at::kBFloat16}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    lshift,
    std::vector<at::ScalarType>(
        {at::kChar,
         at::kShort,
         at::kByte,
         at::kInt,
         at::kFloat,
         at::kHalf,
         at::kDouble,
         at::kLong}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    rshift,
    std::vector<at::ScalarType>(
        {at::kChar,
         at::kShort,
         at::kByte,
         at::kInt,
         at::kFloat,
         at::kHalf,
         at::kDouble,
         at::kLong}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    bitwise_or,
    std::vector<at::ScalarType>(
        {at::kChar, at::kShort, at::kByte, at::kInt, at::kBool, at::kLong}),
    false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    bitwise_xor,
    std::vector<at::ScalarType>(
        {at::kChar, at::kShort, at::kByte, at::kInt, at::kBool, at::kLong}),
    false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    bitwise_and,
    std::vector<at::ScalarType>(
        {at::kChar, at::kShort, at::kByte, at::kInt, at::kBool, at::kLong}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    bitwise_left_shift,
    std::vector<at::ScalarType>(
        {at::kByte, at::kChar, at::kInt, at::kLong, at::kShort}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    bitwise_right_shift,
    std::vector<at::ScalarType>(
        {at::kByte, at::kChar, at::kInt, at::kLong, at::kShort}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    conj,
    std::vector<at::ScalarType>(
        {at::kChar,
         at::kShort,
         at::kByte,
         at::kInt,
         at::kLong,
         at::kBool,
         at::kFloat,
         at::kComplexFloat,
         at::kComplexDouble,
         at::kDouble,
         at::kLong,
         at::kHalf}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    exp2,
    std::vector<at::ScalarType>(
        {at::kFloat,
         at::kHalf,
         at::kBFloat16,
         at::kDouble,
         at::kComplexFloat,
         at::kComplexDouble}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    nan_to_num,
    std::vector<at::ScalarType>(
        {at::kBool, at::kFloat, at::kHalf, at::kBFloat16, at::kDouble}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    round,
    std::vector<at::ScalarType>(
        {at::kChar,
         at::kShort,
         at::kByte,
         at::kInt,
         at::kLong,
         at::kFloat,
         at::kBFloat16,
         at::kHalf,
         at::kDouble}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(bce, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(bce_backward, f_bf_h_types, false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    addr,
    std::vector<at::ScalarType>(
        {at::kFloat, at::kHalf, at::kDouble, at::kBFloat16}),
    false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(ceil, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(trunc, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(floor, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(frac, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(lerp, f_bf_h_types, false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    sin,
    std::vector<at::ScalarType>(
        {at::kHalf, at::kBFloat16, at::kFloat, at::kDouble}),
    false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(sinh, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    cos,
    std::vector<at::ScalarType>(
        {at::kHalf, at::kBFloat16, at::kFloat, at::kDouble}),
    false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(cosh, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(tan, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(tanh, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(asin, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(asinh, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(atan, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(atanh, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(atan2, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(acosh, f_bf_h_types, false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(acos, f_bf_h_types, false)

REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    addcmul,
    std::vector<at::ScalarType>(
        {at::kHalf, at::kBFloat16, at::kFloat, at::kDouble}),
    false)
REGISTER_PARAMS_WITHOUT_MIXED_INPUT(
    addcdiv,
    std::vector<at::ScalarType>(
        {at::kHalf, at::kBFloat16, at::kFloat, at::kDouble}),
    false)

// Register op params with strided memory.
REGISTER_STRIDED_OP_PARAM_WITHOUT_MIXED_INPUT(
    sgn,
    std::vector<at::ScalarType>(
        {at::kHalf,
         at::kBFloat16,
         at::kFloat,
         at::kDouble,
         at::kComplexFloat,
         at::kComplexDouble}),
    false,
    true)
REGISTER_STRIDED_OP_PARAM_WITHOUT_MIXED_INPUT(
    sign,
    std::vector<at::ScalarType>(
        {at::kHalf, at::kBFloat16, at::kFloat, at::kDouble}),
    false,
    true)

// Add a fake op for gtest.
static RegisterCnnlOpParams fake_op_params_register(
    f_bf_h_types,
    "fake_op",
    true,
    true,
    false);
// Add a channel_last_op for gtest.
static RegisterCnnlOpParams channel_last_op_params_register(
    f_bf_h_types,
    "channel_last_op",
    false,
    false,
    false,
    false,
    OpMluMemoryFormatStrategy::ChannelsLast);
static RegisterCnnlOpParams suggest_memory_format_op_params_register(
    f_bf_h_types,
    "suggest_memory_format_soft_op",
    false,
    false,
    false,
    false,
    OpMluMemoryFormatStrategy::SuggestMemoryFormatSoft);

static RegisterCnnlOpParams fix_type_op_params_register(
    f_bf_h_types,
    "fix_type_op",
    false,
    false,
    false,
    false,
    OpMluMemoryFormatStrategy::Default,
    false,
    logic_input_mix_types,
    at::kInt);

} // anonymous namespace

} // namespace torch_mlu
