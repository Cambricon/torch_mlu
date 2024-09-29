#include <gtest/gtest.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include "aten/MLUFallback.h"
#include "utils/assert_tensor.h"

namespace torch_mlu {
namespace {

struct abs_out {
  using schema = at::Tensor&(const at::Tensor&, at::Tensor&);
  using ptr_schema = schema*;
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(
      name,
      "test_mlu_sparse_fail_fallback::abs")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(
      schema_str,
      "abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
  static at::Tensor& call(const at::Tensor& self, at::Tensor& out);
};

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(
    abs_out,
    name,
    "test_mlu_sparse_fail_fallback::abs")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(abs_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(
    abs_out,
    schema_str,
    "abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

static C10_NOINLINE c10::TypedOperatorHandle<abs_out::schema>
create_abs_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(abs_out::name, abs_out::overload_name)
      .typed<abs_out::schema>();
}

at::Tensor& abs_out::call(const at::Tensor& self, at::Tensor& out) {
  static auto op = create_abs_out_typed_handle();
  return op.call(self, out);
}

struct tensor_list_abs_out {
  using schema = void(at::TensorList, at::TensorList);
  using ptr_schema = schema*;
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(
      name,
      "test_mlu_sparse_fail_fallback::tensor_list_abs")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(
      schema_str,
      "tensor_list_abs.out(Tensor[] tensors, Tensor(a!)[] out) -> ()")
  static void call(at::TensorList tensors, at::TensorList outs);
};

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(
    tensor_list_abs_out,
    name,
    "test_mlu_sparse_fail_fallback::tensor_list_abs")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(
    tensor_list_abs_out,
    overload_name,
    "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(
    tensor_list_abs_out,
    schema_str,
    "tensor_list_abs.out(Tensor[] tensors, Tensor(a!)[] out) -> ()")

static C10_NOINLINE c10::TypedOperatorHandle<tensor_list_abs_out::schema>
create_tensor_list_abs_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(
          tensor_list_abs_out::name, tensor_list_abs_out::overload_name)
      .typed<tensor_list_abs_out::schema>();
}

void tensor_list_abs_out::call(at::TensorList tensors, at::TensorList outs) {
  static auto op = create_tensor_list_abs_out_typed_handle();
  op.call(tensors, outs);
}

template <typename Return, typename F1, typename F2, typename... Args>
static Return op_call(F1 impl_call, F2 fallback_call, Args&&... args) {
  try {
    return impl_call(std::forward<Args>(args)...);
  } catch (std::exception& e) {
    CNLOG(INFO) << e.what();
    return fallback_call(std::forward<Args>(args)...);
  }
}

at::Tensor& wrapper_abs_out(const at::Tensor& self, at::Tensor& out) {
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  auto* result_impl = at::sparse::get_sparse_impl(out);
  auto result_values = result_impl->values();
  result_values.resize_(self._values().sizes());
  TORCH_CHECK(false, "For testing fail_fallback.");
  return out;
}

at::Tensor& wrapper_wrapper_abs_out(const at::Tensor& self, at::Tensor& out) {
  auto impl_fn = wrapper_abs_out;
  auto fallback_fn = at::native::
      call_fallback_fn_symint<&mlu_sparse_fail_fallback, abs_out>::call;
  return op_call<at::Tensor&>(impl_fn, fallback_fn, self, out);
}

at::Tensor& wrapper_SparseCPU_abs_out(const at::Tensor& self, at::Tensor& out) {
  return at::native::abs_sparse_out(self, out);
}

void wrapper_tensor_list_abs_out(at::TensorList tensors, at::TensorList outs) {
  const c10::OptionalDeviceGuard device_guard(device_of(tensors));
  auto* result_impl = at::sparse::get_sparse_impl(outs[0]);
  auto result_values = result_impl->values();
  result_values.resize_(tensors[0]._values().sizes());
  TORCH_CHECK(false, "For testing fail_fallback.");
}

void wrapper_wrapper_tensor_list_abs_out(
    at::TensorList tensors,
    at::TensorList outs) {
  auto impl_fn = wrapper_tensor_list_abs_out;
  auto fallback_fn = at::native::call_fallback_fn_symint<
      &mlu_sparse_fail_fallback,
      tensor_list_abs_out>::call;
  op_call<void>(impl_fn, fallback_fn, tensors, outs);
}

void wrapper_SparseCPU_tensor_list_abs_out(
    at::TensorList tensors,
    at::TensorList outs) {
  for (int i = 0; i < tensors.size(); i++) {
    at::Tensor tensor = tensors[i];
    at::Tensor out = outs[i];
    at::native::abs_sparse_out(tensor, out);
  }
}

} // namespace

TORCH_LIBRARY(test_mlu_sparse_fail_fallback, m) {
  m.def("abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", {});
  m.def("tensor_list_abs.out(Tensor[] tensors, Tensor(a!)[] out) -> ()", {});
}

TORCH_LIBRARY_IMPL(test_mlu_sparse_fail_fallback, SparsePrivateUse1, m) {
  m.impl("abs.out", TORCH_FN(wrapper_wrapper_abs_out));
  m.impl("tensor_list_abs.out", TORCH_FN(wrapper_wrapper_tensor_list_abs_out));
}

TORCH_LIBRARY_IMPL(test_mlu_sparse_fail_fallback, SparseCPU, m) {
  m.impl("abs.out", TORCH_FN(wrapper_SparseCPU_abs_out));
  m.impl(
      "tensor_list_abs.out", TORCH_FN(wrapper_SparseCPU_tensor_list_abs_out));
}

TEST(MLUFailFallbackWithProcessOutArguments, TestTensor) {
  std::vector<int64_t> sizes = {2, 20};
  at::Tensor indices = at::ones({2, 10}, at::kLong);
  at::Tensor values = at::randn({10}, at::kFloat);
  at::sparse::SparseTensor sparse =
      at::sparse_coo_tensor(indices, values, sizes);
  at::Tensor out = at::empty({0}, sparse.options());
  at::sparse::SparseTensor sparse_mlu =
      sparse.to(at::Device(at::Device::Type::PrivateUse1));
  at::Tensor out_mlu = out.to(at::Device(at::Device::Type::PrivateUse1));

  at::sparse::SparseTensor result_cpu = abs_out::call(sparse, out);
  at::sparse::SparseTensor result_mlu = abs_out::call(sparse_mlu, out_mlu);

  assertTensorsEqual(out, out_mlu.cpu(), 0.0, true, false, false);
}

TEST(FallbackWithProcessOutArguments, TestTensorList) {
  std::vector<int64_t> sizes = {2, 20};
  at::Tensor indices = at::ones({2, 10}, at::kLong);
  at::Tensor values = at::randn({10}, at::kFloat);
  at::Tensor values1 = at::randn({10}, at::kFloat);
  at::sparse::SparseTensor sparse0 =
      at::sparse_coo_tensor(indices, values, sizes);
  at::sparse::SparseTensor sparse1 =
      at::sparse_coo_tensor(indices, values1, sizes);
  at::Tensor out0 = at::empty({0}, sparse0.options());
  at::Tensor out1 = at::empty({0}, sparse0.options());
  at::sparse::SparseTensor sparse0_mlu =
      sparse0.to(at::Device(at::Device::Type::PrivateUse1));
  at::sparse::SparseTensor sparse1_mlu =
      sparse1.to(at::Device(at::Device::Type::PrivateUse1));
  at::Tensor out0_mlu = out0.to(at::Device(at::Device::Type::PrivateUse1));
  at::Tensor out1_mlu = out1.to(at::Device(at::Device::Type::PrivateUse1));

  tensor_list_abs_out::call({sparse0, sparse1}, {out0, out1});
  tensor_list_abs_out::call({sparse0_mlu, sparse1_mlu}, {out0_mlu, out1_mlu});

  assertTensorsEqual(out0, out0_mlu.cpu(), 0.0, true, false, false);
  assertTensorsEqual(out1, out1_mlu.cpu(), 0.0, true, false, false);
}

} // namespace torch_mlu
