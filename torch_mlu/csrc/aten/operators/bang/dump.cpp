#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"

namespace torch_mlu {
namespace ops {

bool bang_dump(const at::Tensor& input) {
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = mlu_data_ptr(input_impl);
  int32_t size = input.numel();
  cnrtDataType_t cnrt_type = cnnlType2CnrtType(getCnnlType(input_impl));
  TORCH_CHECK(
      CNRT_FLOAT32 == cnrt_type || CNRT_FLOAT16 == cnrt_type,
      "Currently only support float32 and float16 dtype, ",
      "not implemented for ",
      toString(input.scalar_type()));
  cnrtDim3_t dim = {1, 1, 1};
  cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_BLOCK;
  auto stream = getCurMLUStream();
  dump(input_ptr, size, dim, ktype, stream, cnrt_type);
  cnrtQueueSync(stream);
  return true;
}

} // namespace ops
} // namespace torch_mlu
