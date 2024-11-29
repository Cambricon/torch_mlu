#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_pow_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const std::optional<at::Tensor>& exponent_t,
    const std::optional<at::Scalar>& exponent_s) {
  at::Tensor exp_t;
  at::Scalar exp_s;

  if (input.numel() == 0 ||
      (exponent_t.has_value() && (exponent_t.value().numel() == 0))) {
    return output;
  }

  auto cnnl_layout = suggest_cnnl_layout(input);

  size_t sz = 0;
  auto handle = getCurrentHandle();

  auto input_impl = getMluTensorImpl(input);
  auto descInput = getTensorDesc(input_impl, cnnl_layout);
  auto input_ptr = input_impl->mlu_data_ptr();

  auto output_impl = getMluTensorImpl(output);
  auto descOutput = getTensorDesc(output_impl, cnnl_layout);
  auto output_ptr = output_impl->mlu_data_ptr();

  tensorDescPtr_t descExp;
  void* exp_ptr;

  TORCH_CHECK(
      exponent_t.has_value() || exponent_s.has_value(),
      "pow requires exponent to be specified");

  AT_DISPATCH_ALL_TYPES_AND2(
      at::kHalf, at::kBFloat16, input.scalar_type(), "pow_mlu", [&]() {
        if (exponent_t.has_value()) {
          exp_t = exponent_t.value();

          auto exp_impl = getMluTensorImpl(exp_t);
          descExp = getTensorDesc(exp_impl, cnnl_layout);
          exp_ptr = exp_impl->mlu_data_ptr();
        } else {
          exp_s = exponent_s.value();
          exp_ptr = new torch_mlu::Convert64BitTo32Bit_t<scalar_t>(
              exp_s.to<torch_mlu::Convert64BitTo32Bit_t<scalar_t>>());
          auto cnnl_type = getCnnlDataType(input.scalar_type());
          descExp = getCpuTensorDesc(
              cnnl_type, CNNL_POINTER_MODE_HOST);
        }

        TORCH_CNNL_CHECK(cnnlGetPowWorkspaceSize(
            handle, descInput.get(), descExp.get(), descOutput.get(), &sz));
        auto workspace_ptr =
            torch_mlu::MLUCachingAllocator::get()->allocate(sz);

        TORCH_CNNL_CHECK(cnnlPow_v2(
            handle,
            descInput.get(),
            input_ptr,
            descExp.get(),
            exp_ptr,
            workspace_ptr.get(),
            sz,
            descOutput.get(),
            output_ptr));
      });

  return output;

}

} // namespace ops
} // namespace torch_mlu
