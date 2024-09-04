#include <ATen/autocast_mode.h>
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/internal_util.h"
#include "aten/utils/types.h"
#include "framework/core/device.h"
#include "framework/core/device_utils.h"

namespace torch_mlu {
namespace ops {

// Cause on-chip and off-chip type maybe different, so we can only use
// on-chip type to determine the cast type.
using pair = std::pair<cnnlDataType_t, cnnlDataType_t>;
static const std::map<pair, cnnlCastDataType_t> cast_map = {
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_HALF}, CNNL_CAST_FLOAT_TO_HALF},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT32}, CNNL_CAST_FLOAT_TO_INT32},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT16}, CNNL_CAST_FLOAT_TO_INT16},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT8}, CNNL_CAST_FLOAT_TO_INT8},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_UINT8}, CNNL_CAST_FLOAT_TO_UINT8},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_BOOL}, CNNL_CAST_FLOAT_TO_BOOL},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_BFLOAT16}, CNNL_CAST_FLOAT_TO_BFLOAT16},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT64}, CNNL_CAST_FLOAT_TO_INT64},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_DOUBLE}, CNNL_CAST_FLOAT_TO_DOUBLE},
    {pair{CNNL_DTYPE_BFLOAT16, CNNL_DTYPE_FLOAT}, CNNL_CAST_BFLOAT16_TO_FLOAT},
    {pair{CNNL_DTYPE_BFLOAT16, CNNL_DTYPE_BOOL}, CNNL_CAST_BFLOAT16_TO_BOOL},
    // without half to double
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_FLOAT}, CNNL_CAST_HALF_TO_FLOAT},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_INT32}, CNNL_CAST_HALF_TO_INT32},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_INT16}, CNNL_CAST_HALF_TO_INT16},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_INT8}, CNNL_CAST_HALF_TO_INT8},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_UINT8}, CNNL_CAST_HALF_TO_UINT8},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_BOOL}, CNNL_CAST_HALF_TO_BOOL},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_INT64}, CNNL_CAST_HALF_TO_INT64},
    // without int32 to double
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_FLOAT}, CNNL_CAST_INT32_TO_FLOAT},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_HALF}, CNNL_CAST_INT32_TO_HALF},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_INT8}, CNNL_CAST_INT32_TO_INT8},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_INT16}, CNNL_CAST_INT32_TO_INT16},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_BOOL}, CNNL_CAST_INT32_TO_BOOL},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_UINT8}, CNNL_CAST_INT32_TO_UINT8},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_INT64}, CNNL_CAST_INT32_TO_INT64},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_BFLOAT16}, CNNL_CAST_INT32_TO_BFLOAT16},
    // Only support INT16 to INT32, Float, Half
    {pair{CNNL_DTYPE_INT16, CNNL_DTYPE_FLOAT}, CNNL_CAST_INT16_TO_FLOAT},
    {pair{CNNL_DTYPE_INT16, CNNL_DTYPE_HALF}, CNNL_CAST_INT16_TO_HALF},
    {pair{CNNL_DTYPE_INT16, CNNL_DTYPE_INT32}, CNNL_CAST_INT16_TO_INT32},
    // Only support INT8 to INT16, INT32, Float, Half
    {pair{CNNL_DTYPE_INT8, CNNL_DTYPE_FLOAT}, CNNL_CAST_INT8_TO_FLOAT},
    {pair{CNNL_DTYPE_INT8, CNNL_DTYPE_HALF}, CNNL_CAST_INT8_TO_HALF},
    {pair{CNNL_DTYPE_INT8, CNNL_DTYPE_INT32}, CNNL_CAST_INT8_TO_INT32},
    {pair{CNNL_DTYPE_INT8, CNNL_DTYPE_INT16}, CNNL_CAST_INT8_TO_INT16},
    // Only support UINT8 to INT32, INT64, Float, Half
    {pair{CNNL_DTYPE_UINT8, CNNL_DTYPE_FLOAT}, CNNL_CAST_UINT8_TO_FLOAT},
    {pair{CNNL_DTYPE_UINT8, CNNL_DTYPE_HALF}, CNNL_CAST_UINT8_TO_HALF},
    {pair{CNNL_DTYPE_UINT8, CNNL_DTYPE_INT32}, CNNL_CAST_UINT8_TO_INT32},
    {pair{CNNL_DTYPE_UINT8, CNNL_DTYPE_INT64}, CNNL_CAST_UINT8_TO_INT64},
    // Only support BOOL to INT32, Float, Half, Bfloat16
    {pair{CNNL_DTYPE_BOOL, CNNL_DTYPE_FLOAT}, CNNL_CAST_BOOL_TO_FLOAT},
    {pair{CNNL_DTYPE_BOOL, CNNL_DTYPE_HALF}, CNNL_CAST_BOOL_TO_HALF},
    {pair{CNNL_DTYPE_BOOL, CNNL_DTYPE_INT32}, CNNL_CAST_BOOL_TO_INT32},
    {pair{CNNL_DTYPE_BOOL, CNNL_DTYPE_BFLOAT16}, CNNL_CAST_BOOL_TO_BFLOAT16},
    {pair{CNNL_DTYPE_INT64, CNNL_DTYPE_INT32}, CNNL_CAST_INT64_TO_INT32},
    {pair{CNNL_DTYPE_INT64, CNNL_DTYPE_FLOAT}, CNNL_CAST_INT64_TO_FLOAT},
    {pair{CNNL_DTYPE_INT64, CNNL_DTYPE_HALF}, CNNL_CAST_INT64_TO_HALF},
    {pair{CNNL_DTYPE_DOUBLE, CNNL_DTYPE_FLOAT}, CNNL_CAST_DOUBLE_TO_FLOAT},
    // cast to complex_float
    {pair{CNNL_DTYPE_DOUBLE, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_DOUBLE_TO_COMPLEX_FLOAT},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_FLOAT_TO_COMPLEX_FLOAT},
    {pair{CNNL_DTYPE_BFLOAT16, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_BFLOAT16_TO_COMPLEX_FLOAT},
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_HALF_TO_COMPLEX_FLOAT},
    {pair{CNNL_DTYPE_INT64, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_INT64_TO_COMPLEX_FLOAT},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_INT32_TO_COMPLEX_FLOAT},
    {pair{CNNL_DTYPE_INT16, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_INT16_TO_COMPLEX_FLOAT},
    {pair{CNNL_DTYPE_INT8, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_INT8_TO_COMPLEX_FLOAT},
    {pair{CNNL_DTYPE_UINT8, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_UINT8_TO_COMPLEX_FLOAT},
    {pair{CNNL_DTYPE_BOOL, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_BOOL_TO_COMPLEX_FLOAT},
    {pair{CNNL_DTYPE_COMPLEX_HALF, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_COMPLEX_HALF_TO_COMPLEX_FLOAT},
    {pair{CNNL_DTYPE_COMPLEX_DOUBLE, CNNL_DTYPE_COMPLEX_FLOAT},
     CNNL_CAST_COMPLEX_DOUBLE_TO_COMPLEX_FLOAT},
    // cast to complex_half
    {pair{CNNL_DTYPE_HALF, CNNL_DTYPE_COMPLEX_HALF},
     CNNL_CAST_HALF_TO_COMPLEX_HALF},
    {pair{CNNL_DTYPE_FLOAT, CNNL_DTYPE_COMPLEX_HALF},
     CNNL_CAST_FLOAT_TO_COMPLEX_HALF},
    {pair{CNNL_DTYPE_BFLOAT16, CNNL_DTYPE_COMPLEX_HALF},
     CNNL_CAST_BFLOAT16_TO_COMPLEX_HALF},
    {pair{CNNL_DTYPE_INT64, CNNL_DTYPE_COMPLEX_HALF},
     CNNL_CAST_INT64_TO_COMPLEX_HALF},
    {pair{CNNL_DTYPE_INT32, CNNL_DTYPE_COMPLEX_HALF},
     CNNL_CAST_INT32_TO_COMPLEX_HALF},
    {pair{CNNL_DTYPE_INT16, CNNL_DTYPE_COMPLEX_HALF},
     CNNL_CAST_INT16_TO_COMPLEX_HALF},
    {pair{CNNL_DTYPE_INT8, CNNL_DTYPE_COMPLEX_HALF},
     CNNL_CAST_INT8_TO_COMPLEX_HALF},
    {pair{CNNL_DTYPE_UINT8, CNNL_DTYPE_COMPLEX_HALF},
     CNNL_CAST_UINT8_TO_COMPLEX_HALF},
    {pair{CNNL_DTYPE_BOOL, CNNL_DTYPE_COMPLEX_HALF},
     CNNL_CAST_BOOL_TO_COMPLEX_HALF},
    // cast from complex_float
    {pair{CNNL_DTYPE_COMPLEX_FLOAT, CNNL_DTYPE_INT8},
     CNNL_CAST_COMPLEX_FLOAT_TO_INT8},
    {pair{CNNL_DTYPE_COMPLEX_FLOAT, CNNL_DTYPE_UINT8},
     CNNL_CAST_COMPLEX_FLOAT_TO_UINT8},
    {pair{CNNL_DTYPE_COMPLEX_FLOAT, CNNL_DTYPE_INT16},
     CNNL_CAST_COMPLEX_FLOAT_TO_INT16},
    {pair{CNNL_DTYPE_COMPLEX_FLOAT, CNNL_DTYPE_INT32},
     CNNL_CAST_COMPLEX_FLOAT_TO_INT32},
    {pair{CNNL_DTYPE_COMPLEX_FLOAT, CNNL_DTYPE_INT64},
     CNNL_CAST_COMPLEX_FLOAT_TO_INT64},
    {pair{CNNL_DTYPE_COMPLEX_FLOAT, CNNL_DTYPE_BOOL},
     CNNL_CAST_COMPLEX_FLOAT_TO_BOOL},
    {pair{CNNL_DTYPE_COMPLEX_FLOAT, CNNL_DTYPE_FLOAT},
     CNNL_CAST_COMPLEX_FLOAT_TO_FLOAT},
    {pair{CNNL_DTYPE_COMPLEX_FLOAT, CNNL_DTYPE_HALF},
     CNNL_CAST_COMPLEX_FLOAT_TO_HALF},
    {pair{CNNL_DTYPE_COMPLEX_FLOAT, CNNL_DTYPE_COMPLEX_DOUBLE},
     CNNL_CAST_COMPLEX_FLOAT_TO_COMPLEX_DOUBLE},
    {pair{CNNL_DTYPE_COMPLEX_FLOAT, CNNL_DTYPE_COMPLEX_HALF},
     CNNL_CAST_COMPLEX_FLOAT_TO_COMPLEX_HALF},
    {pair{CNNL_DTYPE_COMPLEX_FLOAT, CNNL_DTYPE_BFLOAT16},
     CNNL_CAST_COMPLEX_FLOAT_TO_BFLOAT16},
    // cast from complex_half
    {pair{CNNL_DTYPE_COMPLEX_HALF, CNNL_DTYPE_INT8},
     CNNL_CAST_COMPLEX_HALF_TO_INT8},
    {pair{CNNL_DTYPE_COMPLEX_HALF, CNNL_DTYPE_UINT8},
     CNNL_CAST_COMPLEX_HALF_TO_UINT8},
    {pair{CNNL_DTYPE_COMPLEX_HALF, CNNL_DTYPE_INT16},
     CNNL_CAST_COMPLEX_HALF_TO_INT16},
    {pair{CNNL_DTYPE_COMPLEX_HALF, CNNL_DTYPE_INT32},
     CNNL_CAST_COMPLEX_HALF_TO_INT32},
    {pair{CNNL_DTYPE_COMPLEX_HALF, CNNL_DTYPE_INT64},
     CNNL_CAST_COMPLEX_HALF_TO_INT64},
    {pair{CNNL_DTYPE_COMPLEX_HALF, CNNL_DTYPE_BOOL},
     CNNL_CAST_COMPLEX_HALF_TO_BOOL},
    {pair{CNNL_DTYPE_COMPLEX_HALF, CNNL_DTYPE_HALF},
     CNNL_CAST_COMPLEX_HALF_TO_HALF},
    {pair{CNNL_DTYPE_COMPLEX_HALF, CNNL_DTYPE_BFLOAT16},
     CNNL_CAST_COMPLEX_HALF_TO_BFLOAT16},
    {pair{CNNL_DTYPE_COMPLEX_HALF, CNNL_DTYPE_FLOAT},
     CNNL_CAST_COMPLEX_HALF_TO_FLOAT}};

// This function is intentionally designed this way; the detailed reasons are
// explained in wiki pageid=71770906.
bool check_amp_mode() {
  static bool has_enabled = false;
  if (!has_enabled) {
    has_enabled = at::autocast::is_privateuseone_enabled();
  }
  return has_enabled;
}

static void cnnl_cast_functional(
    const at::Tensor& input,
    at::Tensor& output,
    cnnlCastDataType_t cast_direction) {
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // add canCast in here?
  cnnlDataType_t src_dtype = getCnnlType(input_impl);
  cnnlDataType_t dst_dtype = getCnnlType(output_impl);
  auto input_may_contiguous = input;
  auto output_may_contiguous = output;
  tensorDescPtr_t input_desc;
  tensorDescPtr_t output_desc;

  if (is_same_format_tensor({output_may_contiguous, input_may_contiguous})) {
    input_desc = getTensorDescAndCoalesceDims(input_impl);
    output_desc = getTensorDescAndCoalesceDims(output_impl);
  } else {
    auto memory_format = output.suggest_memory_format();
    output_may_contiguous = cnnl_contiguous(output, memory_format);
    input_may_contiguous = cnnl_contiguous(input, memory_format);
    input_impl = getMluTensorImpl(input_may_contiguous);
    output_impl = getMluTensorImpl(output_may_contiguous);
    auto layout = suggestCnnlLayout(output_impl);
    input_desc = getTensorDesc(input_impl, src_dtype, layout);
    output_desc = getTensorDesc(output_impl, dst_dtype, layout);
  }

  auto handle = getCurrentHandle();
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  DeviceProp* prop = torch_mlu::getDeviceProperties(input.get_device());
  // check PyTorch AMP has enabled or not.
  if (cast_direction == CNNL_CAST_HALF_TO_FLOAT && check_amp_mode() &&
      (prop->major == 3)) {
    cast_direction = CNNL_CAST_HALF_TO_FLOAT_INF;
    CNLOG(INFO) << "In AMP mode, cnnlCastDataType use cast_type"
                   " CNNL_CAST_HALF_TO_FLOAT_INF"
                   " instead of CNNL_CAST_HALF_TO_FLOAT.";
  }

  TORCH_CNNL_CHECK(cnnlCastDataType(
      handle,
      input_desc.get(),
      input_ptr,
      cast_direction,
      output_desc.get(),
      output_ptr));

  if (is_copy_necessary(output, output_may_contiguous)) {
    output.copy_(output_may_contiguous);
  }
}

void cnnl_cast_internal(const at::Tensor& input, at::Tensor& output) {
  if (input.numel() == 0)
    return;
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // add canCast in here?
  cnnlDataType_t src_dtype = getCnnlType(input_impl);
  cnnlDataType_t dst_dtype = getCnnlType(output_impl);
  if (src_dtype == dst_dtype) {
    cnnl_copy_internal(output, input);
    return;
  }

  // Determine the data conversion type.
  auto iter = cast_map.find(
      std::pair<cnnlDataType_t, cnnlDataType_t>({src_dtype, dst_dtype}));
  if (iter == cast_map.end()) {
    // If output tensor is floating point type, then cast to float first.
    at::ScalarType internal_type = at::kFloat;
    cnnlDataType_t internal_cnnl_type = CNNL_DTYPE_FLOAT;
    // If input and output are both integral type, then cast to int32 first.
    if (c10::isIntegralType(input.scalar_type()) &&
        c10::isIntegralType(output.scalar_type())) {
      internal_type = at::kInt;
      internal_cnnl_type = CNNL_DTYPE_INT32;
    }
    auto iter_to_tmp = cast_map.find(pair({src_dtype, internal_cnnl_type}));
    auto iter_to_output = cast_map.find(pair({internal_cnnl_type, dst_dtype}));
    auto tmp = at::empty_like(input, input.options().dtype(internal_type));
    cnnl_cast_functional(input, tmp, iter_to_tmp->second);
    cnnl_cast_functional(tmp, output, iter_to_output->second);
  } else {
    cnnl_cast_functional(input, output, iter->second);
  }
}

} // namespace ops
} // namespace torch_mlu
