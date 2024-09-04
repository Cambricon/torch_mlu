#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_pow_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& exponent) {
  if (input.numel() == 0 || exponent.numel() == 0) {
    return output;
  }
  TORCH_CHECK(
      at::isFloatingType(input.scalar_type()),
      "pow input only support floating type");
  TORCH_CHECK(
      at::isFloatingType(exponent.scalar_type()) ||
          at::isIntegralType(exponent.scalar_type()),
      "pow exponent only support floating/integral type");

  auto cnnl_layout = suggest_cnnl_layout(input);
  size_t sz = 0;
  auto handle = getCurrentHandle();

  auto input_impl = getMluTensorImpl(input);
  auto descInput = getTensorDesc(input_impl, cnnl_layout);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto exp_impl = getMluTensorImpl(exponent);
  auto descExp = getTensorDesc(exp_impl, cnnl_layout);
  auto exp_ptr = mlu_data_ptr(exp_impl);

  auto output_impl = getMluTensorImpl(output);
  auto descOutput = getTensorDesc(output_impl, cnnl_layout);
  auto output_ptr = mlu_data_ptr(output_impl);

  TORCH_CNNL_CHECK(cnnlGetPowWorkspaceSize(
      handle, descInput.get(), descExp.get(), descOutput.get(), &sz));
  auto workspace_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(sz);
  // set descriptor config
  cnnlComputationPreference_t high_precision = CNNL_COMPUTATION_HIGH_PRECISION;
  TORCH_CNNL_CHECK(cnnlPow(
      handle,
      high_precision,
      descInput.get(),
      input_ptr,
      descExp.get(),
      exp_ptr,
      workspace_ptr.get(),
      sz,
      descOutput.get(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
