#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"

namespace torch_mlu {
namespace ops {

bool bang_dump(const at::Tensor& input) {
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = mlu_data_ptr(input_impl);
  int32_t size = input.numel();
  cnrtDataType_V2_t cnrt_type = cnnlType2CnrtType_V2(getCnnlType(input_impl));
  TORCH_CHECK(
      cnrtFloat == cnrt_type || cnrtHalf == cnrt_type,
      "Currently only support float32 and float16 dtype, ",
      "not implemented for ",
      toString(input.scalar_type()));
  cnrtDim3_t dim = {1, 1, 1};
  cnrtFunctionType_t ktype = cnrtFuncTypeBlock;
  auto stream = getCurMLUStream();
  dump(input_ptr, size, dim, ktype, stream, cnrt_type);
  cnrtQueueSync(stream);
  return true;
}

} // namespace ops
} // namespace torch_mlu
