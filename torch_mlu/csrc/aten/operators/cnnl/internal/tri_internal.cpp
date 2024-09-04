#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/dispatch.h"
namespace torch_mlu {
namespace ops {

void cnnl_tri_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t diagonal,
    bool tri_up_mode) {
  auto handle = getCurrentHandle();
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto desc_a = getTensorDesc(input_impl);
  auto desc_b = getTensorDesc(output_impl);

  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  // GPU not support BFloat16, CPU support BFloat16
  // Although CNNL kernel support BFloat16, TORCH_MLU don't add BFloat16 type
  // support. cnnl kernel only support int.
  int diagonal_int = static_cast<int>(diagonal);
  AT_DISPATCH_ALL_MLU_TYPES_AND_HALF_AND_BFLOAT16_EXCEPT_UINT8(
      input.scalar_type(), "tri_out_mlu_impl", [&] {
        TORCH_CNNL_CHECK(cnnlTri_v2(
            handle,
            diagonal_int,
            tri_up_mode,
            desc_a.get(),
            input_ptr,
            desc_b.get(),
            output_ptr));
      });
}

} // namespace ops
} // namespace torch_mlu
