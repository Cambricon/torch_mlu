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
  TORCH_MLU_CHECK(
      at::isFloatingType(input.scalar_type()),
      "pow input only support floating type");
  TORCH_MLU_CHECK(
      at::isFloatingType(exponent.scalar_type()) ||
          at::isIntegralType(exponent.scalar_type()),
      "pow exponent only support floating/integral type");

  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descExp;
  CnnlTensorDescriptor descOutput;

  auto cnnl_layout = suggest_cnnl_layout(input);

  descInput.set(input, cnnl_layout);
  descExp.set(exponent, cnnl_layout);
  descOutput.set(output, cnnl_layout);
  size_t sz = 0;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetPowWorkspaceSize(
      handle, descInput.desc(), descExp.desc(), descOutput.desc(), &sz));

  auto input_impl = getMluTensorImpl(input);
  auto exp_impl = getMluTensorImpl(exponent);
  auto output_impl = getMluTensorImpl(output);
  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto exp_ptr = exp_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto workspace_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(sz);
  // set descriptor config
  cnnlComputationPreference_t high_precision = CNNL_COMPUTATION_HIGH_PRECISION;
  TORCH_CNNL_CHECK(cnnlPow(
      handle,
      high_precision,
      descInput.desc(),
      input_ptr,
      descExp.desc(),
      exp_ptr,
      workspace_ptr.get(),
      sz,
      descOutput.desc(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
