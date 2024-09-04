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
#include <algorithm>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "aten/TensorIteratorBridge.h"
#include "ATen/native/ReduceOps.h"
#include "ATen/native/ReduceAllOps.h"
#include "ATen/native/ReduceOpsUtils.h"
#include "c10/core/ScalarType.h"
#include "ATen/native/TensorCompare.h"
#include "c10/util/DimVector.h"
#include "ATen/WrapDimUtils.h"

namespace torch_mlu {
namespace ops {
using namespace at::native;

namespace {
// TODO(CNNLCORE-13711): reduce min/max support all dtype
static const std::vector<at::ScalarType> f_bf16_h_i_types(
    {at::kFloat, at::kHalf, at::kBFloat16, at::kInt, at::kDouble, at::kLong});
} // namespace

inline at::ScalarType get_dtype_from_result(
    at::Tensor& result,
    c10::optional<at::ScalarType> dtype) {
  TORCH_CHECK(
      result.defined(),
      "Cannot create a new tensor inside a reduction op.",
      "You likely tried to call an operator with an out argument ",
      "but the out argument was an undefined tensor.");
  if (dtype.has_value()) {
    return dtype.value();
  } else {
    return result.scalar_type();
  }
}

static at::TensorOptions options_to_value_type(at::TensorOptions opts) {
  auto scalar_type = c10::typeMetaToScalarType(opts.dtype());
  return opts.dtype(c10::toRealValueType(scalar_type));
}

std::vector<int64_t> reduces_dims_from_shape(
    const at::IntArrayRef& input_shape,
    const at::IntArrayRef& output_shape) {
  std::vector<int64_t> dims;
  if (input_shape.size() == 1) {
    dims = {0};
    return dims;
  }

  // Get the index of all elements with a value of 1 in the output shape
  // Ingore keedim param, because the output_shape is derived based on
  // keepdim=true in meta func. See aten/src/ATen/native/ReduceOpsUtils.h
  // review_reduce_result func for more details.
  TORCH_CHECK(
      input_shape.size() == output_shape.size(),
      "the input and output shape's size for reduce kernel shoule be same.");
  if (input_shape == output_shape) {
    int64_t idx = std::distance(
        output_shape.begin(),
        std::find(output_shape.begin(), output_shape.end(), 1));
    dims.push_back(idx);
  } else {
    for (int i = 0; i < output_shape.size(); i++) {
      if ((input_shape[i] != output_shape[i]) && (output_shape[i] == 1)) {
        dims.push_back(i);
      }
    }
  }
  return dims;
}

void reduce_stub(
    at::TensorIterator& iter,
    const cnnlReduceOp_t reduce_mode,
    const cnnlReduceIndices_t reduce_indices,
    const char* name,
    float norm_p = 0.) {
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, name);
  auto iter_out = iter_bridge.output(iter, 0);
  auto iter_in = iter_bridge.input(iter, 0);
  auto dims =
      std::move(reduces_dims_from_shape(iter_in.sizes(), iter_out.sizes()));

  at::Tensor result;
  at::Tensor index;
  if (reduce_indices == CNNL_REDUCE_ONLY_INDICES) {
    index = iter_out;
  } else {
    result = iter_out;
    if (iter.noutputs() > 1) {
      index = iter_bridge.output(iter, 1);
    }
  }
  auto input = cast_long_to_int_if_needed(iter_in);
  auto out = create_int_tensor_if_needed(result);
  cnnl_reduce_internal(
      input, out, index, dims, reduce_mode, reduce_indices, norm_p);
  cast_int_to_long_if_needed(out, result);
  iter.cast_outputs();
}

void reduce_input_support_int64_stub(
    at::TensorIterator& iter,
    const cnnlReduceOp_t reduce_mode,
    const cnnlReduceIndices_t reduce_indices,
    const char* name,
    float norm_p = 0.) {
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, name);
  auto iter_out = iter_bridge.output(iter, 0);
  auto iter_in = iter_bridge.input(iter, 0);
  auto input_shape = iter_in.sizes().vec();
  auto output_shape = iter_out.sizes().vec();
  auto dims = reduces_dims_from_shape(input_shape, output_shape);

  at::Tensor result;
  at::Tensor index;
  if (reduce_indices == CNNL_REDUCE_ONLY_INDICES) {
    index = iter_out;
  } else {
    result = iter_out;
    if (iter.noutputs() > 1) {
      index = iter_bridge.output(iter, 1);
    }
  }
  auto input = iter_in;
  auto out = result;
  cnnl_reduce_internal(
      input, out, index, dims, reduce_mode, reduce_indices, norm_p);
  iter.cast_outputs();
}

/***************************************sum****************************************/
static void sum_mlu_kernel(at::TensorIterator& iter) {
  reduce_stub(iter, CNNL_REDUCE_ADD, CNNL_REDUCE_NO_INDICES, "sum");
}
/***************************************mean****************************************/
static void mean_mlu_kernel(at::TensorIterator& iter) {
  reduce_stub(iter, CNNL_REDUCE_AVG, CNNL_REDUCE_NO_INDICES, "mean");
}
/***************************************norm****************************************/
void norm_mlu_kernel(at::TensorIterator& iter, const at::Scalar& val) {
  auto p_value = val.to<float>();
  if (iter.numel() == 0) {
    iter.output().fill_((p_value < 0) ? INFINITY : 0);
    return;
  }
  TORCH_CHECK(
      p_value != INFINITY && p_value != -INFINITY,
      "torch_mlu does not support inf-Norm as p=inf/-inf.");
  auto reduce_mode = p_value == 1.0
      ? CNNL_REDUCE_NORM1
      : (p_value == 2.0 ? CNNL_REDUCE_NORM2 : CNNL_REDUCE_NORMP);
  if (reduce_mode == CNNL_REDUCE_NORMP) {
    reduce_stub(iter, reduce_mode, CNNL_REDUCE_NO_INDICES, "norm", p_value);
  } else {
    reduce_stub(iter, reduce_mode, CNNL_REDUCE_NO_INDICES, "norm");
  }
}
/***************************************max****************************************/
void max_mlu_kernel(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto iter = at::meta::make_reduction(
      self, result, indice, dim, keepdim, self.scalar_type(), at::kLong);
  auto input = cast_long_to_int_if_needed(cnnl_contiguous(iter.input(0)));
  auto output = iter.output(0);
  auto index = iter.output(1);
  auto output_contiguous = create_int_tensor_if_needed(cnnl_contiguous(output));
  auto index_contiguous = cnnl_contiguous(index);
  cnnl_reduce_internal(
      input,
      output_contiguous,
      index_contiguous,
      {dim},
      CNNL_REDUCE_MAX,
      CNNL_REDUCE_FLATTENED_INDICES);
  if (is_copy_necessary(output, output_contiguous)) {
    output.copy_(output_contiguous);
  }
  if (is_copy_necessary(index, index_contiguous)) {
    index.copy_(index_contiguous);
  }
}

at::Tensor cnnl_max(const at::Tensor& self) {
  return at::native::max(self);
}

void max_all_mlu_kernel(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = at::native::make_reduction(
      "max_all", result, input, IntArrayRef{}, false, dtype);
  reduce_stub(iter, CNNL_REDUCE_MAX, CNNL_REDUCE_NO_INDICES, "max_all");
}
/***************************************min****************************************/
at::Tensor cnnl_min(const at::Tensor& self) {
  return at::native::min(self);
}

void min_mlu_kernel(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto iter = at::meta::make_reduction(
      self, result, indice, dim, keepdim, self.scalar_type(), at::kLong);
  auto input = cast_long_to_int_if_needed(cnnl_contiguous(iter.input(0)));
  auto output = iter.output(0);
  auto index = iter.output(1);
  auto output_contiguous = create_int_tensor_if_needed(cnnl_contiguous(output));
  auto index_contiguous = cnnl_contiguous(index);
  cnnl_reduce_internal(
      input,
      output_contiguous,
      index_contiguous,
      {dim},
      CNNL_REDUCE_MIN,
      CNNL_REDUCE_FLATTENED_INDICES);
  if (is_copy_necessary(output, output_contiguous)) {
    output.copy_(output_contiguous);
  }
  if (is_copy_necessary(index, index_contiguous)) {
    index.copy_(index_contiguous);
  }
}

void min_all_mlu_kernel(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = at::native::make_reduction(
      "min_all", result, input, IntArrayRef{}, false, input.scalar_type());
  reduce_stub(iter, CNNL_REDUCE_MIN, CNNL_REDUCE_NO_INDICES, "min_all");
}
/***************************************argmax****************************************/
void argmax_mlu_kernel(at::TensorIterator& iter) {
  auto input = cast_long_to_int_if_needed(cnnl_contiguous(iter.input(0)));
  auto result = iter.output(0);
  auto index = cnnl_contiguous(result);
  at::Tensor output;

  auto dims = std::move(reduces_dims_from_shape(input.sizes(), index.sizes()));
  auto in_dtype = input.scalar_type();
  auto compute_dtype =
      get_torch_mlu_promote_types(in_dtype, f_bf16_h_i_types, "argmax", true);
  auto in = input;
  if (in_dtype != compute_dtype) {
    in = input.to(compute_dtype);
  }
  cnnl_reduce_internal(
      in, output, index, dims, CNNL_REDUCE_MAX, CNNL_REDUCE_ONLY_INDICES);
  if (is_copy_necessary(result, index)) {
    result.copy_(index);
  }
}
/***************************************argmin****************************************/
void argmin_mlu_kernel(at::TensorIterator& iter) {
  auto input = cast_long_to_int_if_needed(cnnl_contiguous(iter.input(0)));
  auto result = iter.output(0);
  auto index = cnnl_contiguous(result);
  at::Tensor output;

  auto dims = std::move(reduces_dims_from_shape(input.sizes(), index.sizes()));
  auto in_dtype = input.scalar_type();
  auto compute_dtype =
      get_torch_mlu_promote_types(in_dtype, f_bf16_h_i_types, "argmin", true);
  auto in = input;
  if (in_dtype != compute_dtype) {
    in = input.to(compute_dtype);
  }
  cnnl_reduce_internal(
      in, output, index, dims, CNNL_REDUCE_MIN, CNNL_REDUCE_ONLY_INDICES);
  if (is_copy_necessary(result, index)) {
    result.copy_(index);
  }
}
/***************************************amax****************************************/
void max_values_mlu_kernel(at::TensorIterator& iter) {
  reduce_stub(iter, CNNL_REDUCE_MAX, CNNL_REDUCE_NO_INDICES, "amax");
}
/***************************************amin****************************************/
void min_values_mlu_kernel(at::TensorIterator& iter) {
  reduce_stub(iter, CNNL_REDUCE_MIN, CNNL_REDUCE_NO_INDICES, "amin");
}
/***************************************std/var****************************************/
void std_var_mlu_kernel(
    at::TensorIterator& iter,
    at::OptionalIntArrayRef dim,
    double correction_value,
    bool take_sqrt) {
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "std_var");
  auto result = iter_bridge.output(iter, 0);
  auto self = iter_bridge.input(iter, 0);
  auto dim_value = dim.value_or(IntArrayRef{});
  if (take_sqrt) {
    cnnl_std_internal(self, result, dim_value, correction_value);
  } else {
    cnnl_var_internal(self, result, dim_value, correction_value);
  }
  iter.cast_outputs();
}

static at::Tensor& cnnl_std_var_out(
    const char* fname,
    at::Tensor& result,
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction_opt,
    bool keepdim,
    bool take_sqrt) {
  TORCH_CHECK(
      self.layout() == at::Layout::Strided,
      "std and var only supports strided layout, got: ",
      self.layout());
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()) ||
          at::isComplexType(self.scalar_type()),
      "std and var only support floating point and complex dtypes");
  if (at::isComplexType(self.scalar_type())) {
    LOG(ERROR) << "Complex type input is not supported yet!";
  }
  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  TORCH_MLU_CHECK(
      correction == 0 || correction == 1,
      "now correction only supports 0 and 1 but got ",
      correction);
  at::ScalarType dtype = get_dtype_from_result(result, {});
  auto iter =
      at::native::make_reduction(fname, result, self, dim, keepdim, dtype);
  TORCH_CHECK(
      at::canCast(self.scalar_type(), result.scalar_type()),
      "result type ",
      self.scalar_type(),
      " can't be cast to the "
      "desired output type ",
      result.scalar_type());

  if (iter.numel() == 0) {
    // Trivial reduction
    result.fill_(std::numeric_limits<double>::quiet_NaN());
    return result;
  } else {
    std_var_mlu_kernel(iter, dim, correction, take_sqrt);
  }
  return result;
}

at::Tensor cnnl_std(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim) {
  at::Tensor result = at::empty({0}, options_to_value_type(self.options()));
  return cnnl_std_var_out("std", result, self, dim, correction, keepdim, true);
}

at::Tensor& cnnl_std_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim,
    at::Tensor& out) {
  return cnnl_std_var_out("std", out, self, dim, correction, keepdim, true);
}

at::Tensor cnnl_var(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim) {
  at::Tensor result = at::empty({0}, options_to_value_type(self.options()));
  return cnnl_std_var_out("var", result, self, dim, correction, keepdim, false);
}

at::Tensor& cnnl_var_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim,
    at::Tensor& out) {
  return cnnl_std_var_out("var", out, self, dim, correction, keepdim, false);
}

/***************************************all/any****************************************/
void and_mlu_kernel(at::TensorIterator& iter) {
  reduce_input_support_int64_stub(
      iter, CNNL_REDUCE_AND, CNNL_REDUCE_NO_INDICES, "all");
}

void or_mlu_kernel(at::TensorIterator& iter) {
  reduce_input_support_int64_stub(
      iter, CNNL_REDUCE_OR, CNNL_REDUCE_NO_INDICES, "any");
}

/***************************************prod****************************************/
static void prod_mlu_kernel(at::TensorIterator& iter) {
  reduce_stub(iter, CNNL_REDUCE_MUL, CNNL_REDUCE_NO_INDICES, "prod");
}

at::Tensor cnnl_prod(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype) {
  return at::native::prod(self, dtype);
}
/***************************************nansum****************************************/
static void nansum_mlu_kernel(at::TensorIterator& iter) {
  reduce_stub(iter, CNNL_REDUCE_NANSUM, CNNL_REDUCE_NO_INDICES, "nansum");
}
at::Tensor cnnl_nansum(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype) {
  return at::native::nansum(self, dim, keepdim, dtype);
}
at::Tensor& cnnl_nansum_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& out) {
  return at::native::nansum_out(self, dim, keepdim, dtype, out);
}
/*********************REGISTER_PRIVATEUSE1_DISPATCH******************/
REGISTER_PRIVATEUSE1_DISPATCH(max_stub, &max_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(max_all_stub, &max_all_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(min_stub, &min_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(min_all_stub, &min_all_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(sum_stub, &sum_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(mean_stub, &mean_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(norm_stub, &norm_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(argmax_stub, &argmax_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(argmin_stub, &argmin_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(max_values_stub, &max_values_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(min_values_stub, &min_values_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(and_stub, &and_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(or_stub, &or_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(prod_stub, &prod_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(nansum_stub, &nansum_mlu_kernel);
} // namespace ops
} // namespace torch_mlu
