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

#include <unordered_map>
#include <iostream>
#include <set>
#include "c10/core/ScalarType.h"
#include "c10/core/MemoryFormat.h"
#include "aten/utils/exceptions.h"

// For mixed input types, we need to hash the input and output types.
// The value of hash is used to find the specific mixed input types and output
// types.
namespace std {

template <>
struct hash<at::ScalarType> {
  int64_t operator()(
      const at::ScalarType& type,
      int64_t base_value,
      int output_offset = 0) const {
    // Copied from boost::hash_combine.
    base_value ^= std::hash<int64_t>()(static_cast<int>(type) + output_offset) +
        0x9e3779b9 + (base_value << 6) + (base_value >> 2);
    return base_value;
  }
};

} // namespace std

namespace torch_mlu {

// Get hash value of scalar type list. Container need support function begin()
// and end(), and value type is at::ScalarType.
template <
    typename Container,
    std::enable_if_t<
        std::is_convertible<
            typename std::iterator_traits<
                decltype(std::declval<Container>().begin())>::iterator_category,
            std::input_iterator_tag>::value &&
            std::is_convertible<
                typename std::iterator_traits<
                    decltype(std::declval<Container>()
                                 .end())>::iterator_category,
                std::input_iterator_tag>::value &&
            std::is_same<
                typename std::remove_reference<Container>::type::value_type,
                at::ScalarType>::value,
        int> = 0>
int64_t getScalarTypeListHashWithOutputType(
    Container&& scalar_type_list,
    const int output_num = 1) {
  const int container_size = scalar_type_list.end() - scalar_type_list.begin();
  std::hash<at::ScalarType> hash_fn;
  // Each input type is position-insensitive in the container.
  // Construct a multiset to reorder input types.
  // Why need to reorder input types?
  // Because the range of at::ScalarType is 0-17, and there maybe
  // repeat types in sequence, like (at::kFloat, at::kHalf, and at::kFloat).
  // This problem is transformed into how to represent a specific sequence,
  // and the sequence data range is certain, and the data can be repeated,
  // and don't need to care the position of each data.
  // multiset is used to specific this sequence, and get a hash code of those
  // reorder input types.
  const int input_num = container_size - output_num;
  std::multiset<at::ScalarType> input_types(
      scalar_type_list.begin(), scalar_type_list.begin() + input_num);
  int64_t hash_value = 0;
  for (const auto& item : input_types) {
    hash_value ^= hash_fn(item, hash_value);
  }
  // Output type is position-sensitive, also add offset for output.
  // int64_t output_hash_value = 0;
  for (size_t i = 0; i < output_num; ++i) {
    auto it = scalar_type_list.begin() + input_num + i;
    hash_value ^= hash_fn(*(it), hash_value, i);
  }
  return hash_value;
}

/**
 * Note [OpMluMemoryFormatStrategy]
 *
 * OpMluMemoryFormatStrategy is designed to describe the strategy of memory
 * format for each op. There are four kinds of memory format strategy: 1)
 * Default memory format strategy: TORCH_MLU decides a memory layout from input
 *    tensors. Support at::MemoryFormat::Contiguous,
 * at::MemoryFormat::ChannelsLast at::MemoryFormat::ChannelsLast3d and
 * is_non_overlapping_and_dense. If TORCH_MLU can't find a better memory format,
 * then fallback to at::MemoryFormat::Contiguous when dim value less than 4 or
 * greater than 5. Using at::MemoryFormat::ChannelsLast or
 *    at::MemoryFormat::ChannelsLast3d when dim value is 4 ot 5.
 *    Notice: In this strategy, when all input tensors are
 * non_overlapping_and_dense, the iteratorbridge maintains the strides of input
 * tensors unchanged, that means, the cnnl kernel must support strided input
 * tensors. 2) Contiguous strategy: Align input and output tensors memory format
 * to at::MemoryFormat::Contiguous. Allows cnnl kernel not support strided input
 * tensors. 3) ChannelLast strategy: Align input and output tensors memory
 * format to at::MemoryFormat::ChannelsLast or at::MemoryFormat::ChannelsLast3d.
 * Allows cnnl kernel not support strided input tensors. 4)
 * SuggestMemoryFormatSoft strategy: Almost same with default memory format
 * strategy, Only one difference is that always using result of
 * suggest_memory_format of first tensor when TORCH_MLU can't find a better
 * memory format. Notice: In this strategy, when all input tensors are
 * non_overlapping_and_dense, the iteratorbridge maintains the strides of input
 * tensors unchanged, that means, the cnnl kernel must support strided input
 * tensors. 5) SuggestMemoryFormatHard strategy: Align input and output tensors
 * memory format to the first input tensor's suggest_memory_format. Unlike
 * SuggestMemoryFormatSoft strategy, this strategy allows cnnl kernel not
 * support strided input tensors.
 *
 */

enum class OpMluMemoryFormatStrategy : uint8_t {
  // Default memory format strategy.
  Default = 0,
  // align input and output tensors memory format to
  // at::MemoryFormat::Contiguous.
  Contiguous = 1,
  // align input and output tensors memory format to
  // at::MemoryFormat::ChannelsLast or
  // at::MemoryFormat::ChannelsLast3d.
  ChannelsLast = 2,
  // Almost same with default, Only one difference is that always using result
  // of
  // suggest_memory_format of first tensor when TORCH_MLU can't find a better
  // memory
  // format.
  SuggestMemoryFormatSoft = 3,
  // Align input and output tensors memory format to the first input tensor's
  // suggest_memory_format.
  // Unlike SuggestMemoryFormatSoft strategy, this strategy allows cnnl kernel
  // not support strided
  // input tensors.
  SuggestMemoryFormatHard = 4,
  // Keep stride without do any contiguous.
  AllowStrided = 5,
};

/**
 * Note [CnnlOpTypeInfo]
 *
 * CnnlOpTypeInfo is designed to store of data types of each op. Like static
 * type, support types list and mixed types map. Now only used in CnnlOpParams.
 *
 * static_type:
 * static_type is only used for output tensor in TensorIteratorBridge,
 * and keep same behavior with TensorIteratorBase. If gpu op set static_dtype_
 * in build TensorIteratorConfig, TORCH_MLU also need to set static_type
 * CnnlOpTypeInfo, except CNNL op not support. In gpu side, there will be two
 * situations to set this variable: 1) Like comparison op: 1.1) If output is
 * undefined, static_dtype is setted; 1.2) If output is defined and type is not
 * bool, static_dtype is undefined. 2) Like unary_force_boolean op: output type
 * only support static dtype, and static_dtype is always setted.
 *   ""
 * In TORCH_MLU side, this value is always setted for op if op registered with
 * static dtype, Default value is at::ScalarType::Undefined, if this value
 * setted, TORCH_MLU may using this static data type as output tensor data type.
 * Here is not equal with GPU side. Why we need this? 1) In pytorch community,
 * static_dtype_ is stored in TensorIteratorConfig, and used to set output
 * operand target data type. TORCH_MLU can't get this config, because
 * TensorIteratorBridge is deconstructed after TensorIteratorBase build. 2) In
 * TORCH_MLU side, a lot of ops don't support dynamic data type. So TORCH_MLU
 * need to cast tensor data type to common data type for CNNL op. But some ops
 * like logic op are very special, output tensor data type may be not aligned
 * with common data type, it can be a static dtype at::ScalarType::Bool. More
 * details in [static_dtype vs common_dtype]
 *
 * close_types_check:
 * Close all type checks in tensor iterator bridge. It mean CNNL op is same with
 * gpu in support type list. Otherwise keep this value to false or do type check
 * in op side.
 *
 * allow_implicit_type_convert_:
 * For some historical reasons, some ops has been implicit converted input type
 * to CNNL support type in torch_mlu side. Like add, div and ...
 *
 * support_types:
 * CNNL op support types list, this is different with aten op support types.
 * Like aten add op is support a lot of types, but CNNL add op is only support
 * float, half BFloat16 and int.
 * Also different CNNL op support type list is different, not like gpu. so we
 * need to define a support type list for each CNNL op.
 * Note: Set default support types to float and half.
 *
 * support_mixed_types_:
 * Nullary and unary op is not supported mixed input types, so don't set this
 * option. Dynamic type cast is supported in gpu side, but not all ops support
 * dynamic type cast in torch_mlu side. Only several ops support specific mixed
 * input types in torch_mlu side. Like logic op support float + int32 input
 * types, and output type is bool. We need to add input types and output type in
 * this data format.
 * {{half, int, bool}, {float, int, bool}}
 * CAUTION: Always keep input types first and their position-insensitive. Add
 * output types after input types, and their position-sensitive. Now only
 * support one output type, if you want to and more output types support, maybe
 * you need to add output num in function setInputMixedType in CnnlOpParams and
 * modify std::unordered_map<int64_t, at::ScalarType> to
 * std::unordered_map<int64_t, std::vector<at::ScalarType>>.
 * Function getScalarTypeListHashWithOutputType is using for convert
 * human-readable mixed types list to a unordered_map with hash value and output
 * types.
 *
 */

class CnnlOpParams;

namespace details {
class __attribute__((visibility("hidden"))) CnnlOpTypeInfo {
  friend class torch_mlu::CnnlOpParams;

 public:
  AT_DISALLOW_COPY_AND_ASSIGN(CnnlOpTypeInfo);

  explicit CnnlOpTypeInfo(
      const std::vector<at::ScalarType>& support_types,
      bool close_types_check,
      bool allow_implicit_type_convert)
      : support_types_(support_types),
        close_types_check_(close_types_check),
        allow_implicit_type_convert_(allow_implicit_type_convert) {}

  inline void setStaticType(const at::ScalarType& type) {
    TORCH_CHECK(
        type != at::ScalarType::Undefined,
        "static type can't be assigned with undefined scalar type.")
    this->static_type_ = type;
  }

  void setSupportMixedType(
      const std::vector<std::vector<at::ScalarType>>& input_mixed_types_list);

 private:
  // variable
  at::ScalarType static_type_ = at::ScalarType::Undefined;
  bool close_types_check_ = false;
  bool allow_implicit_type_convert_ = false;
  std::vector<at::ScalarType> support_types_ = {};
  std::unordered_map<int64_t, at::ScalarType> support_mixed_types_{};
};
} // namespace details

class __attribute__((visibility("hidden"))) CnnlOpParams {
 public:
  explicit CnnlOpParams(
      const std::vector<at::ScalarType>& support_types,
      const std::string& name,
      bool implicit_type_convert = false,
      bool allow_strided_memory = false,
      bool allow_cpu_scalar = false,
      bool close_types_check = false,
      OpMluMemoryFormatStrategy memory_format_strategy =
          OpMluMemoryFormatStrategy::Default,
      bool close_tensor_broadcast = false)
      : name_(name),
        op_type_info_(support_types, close_types_check, implicit_type_convert),
        allow_strided_memory_(allow_strided_memory),
        allow_cpu_scalar_(allow_cpu_scalar),
        memory_format_strategy_(memory_format_strategy),
        close_tensor_broadcast_(close_tensor_broadcast) {}

  AT_DISALLOW_COPY_AND_ASSIGN(CnnlOpParams);

  inline void setSupportMixedType(
      const std::vector<std::vector<at::ScalarType>>& input_mixed_types_list) {
    TORCH_CHECK(
        !this->is_support_mixed_types_,
        "Mixed type of ",
        this->name_,
        " already setted.");
    this->op_type_info_.setSupportMixedType(input_mixed_types_list);
    this->is_support_mixed_types_ = true;
  }

  inline void setStaticType(const at::ScalarType& type) {
    this->op_type_info_.setStaticType(type);
  }

  inline at::ScalarType getStaticType() const {
    return this->op_type_info_.static_type_;
  }

  inline const std::vector<at::ScalarType>& getSupportType() const {
    return op_type_info_.support_types_;
  }

  inline const std::unordered_map<int64_t, at::ScalarType>& getSupportMixedType()
      const {
    return op_type_info_.support_mixed_types_;
  }

  inline bool isSupportMixedInputTypes() const {
    return is_support_mixed_types_;
  }

  inline bool isCloseTypeCheck() const {
    return op_type_info_.close_types_check_;
  }

  inline bool isAllowImplicitTypeConvert() const {
    return op_type_info_.allow_implicit_type_convert_;
  }

 public:
  // Op name in torch_mlu params, and there is a little different with aten op
  // name. Such as 'add.Tensor', 'add.out', 'add_.Tensor' in aten, 'opTensor' is
  // torch_mlu params name.
  std::string name_ = "";

  // TODO(shangang): Using OpMluMemoryFormatStrategy::AllowStrided to replace
  // this. Almost CNNL op not support strided memory, so we need to check input
  // and output tensor memory format in torch_mlu side. This option is for
  // future use, maybe some op need to support strided memory in high priority,
  // and others are not urgent.
  bool allow_strided_memory_ = false;

  // Not using now, pytorch different ops may cast to cpu scalar to different
  // type, some ops to common type of input tensor, some ops to type of output
  // tensor. Pytorch do cpu scalar to tensor type in op kernel side, so we need
  // do more research for this function. Almost CNNL op not support cpu scalar,
  // so we need to check input and output tensor in torch_mlu side. This option
  // is for future use, maybe some op need to support cpu scalar in high
  // priority, and others are not urgent.
  bool allow_cpu_scalar_ = false;

  // Index ops have different input types and only support contiguous memory
  // format, and this is very different with element-wise op. So add those two
  // ops in here.
  OpMluMemoryFormatStrategy memory_format_strategy_ =
      OpMluMemoryFormatStrategy::Default;

  // Close broadcast in tensor iterator bridge.
  // More details in Note TensorBroadcastOnTensorIteratorBridge.
  bool close_tensor_broadcast_ = false;

 private:
  // More details in Note [CnnlOpTypeInfo] support_mixed_types_.
  // This variable is for fast check.
  bool is_support_mixed_types_ = false;
  details::CnnlOpTypeInfo op_type_info_;
};

// Always return a CNNL op params. For most CNNL op, the params is same.
// So if op name has been registered, return the specific params.
// Otherwise return the default params.
const CnnlOpParams& getCnnlOpParams(const std::string& name);

} // namespace torch_mlu
