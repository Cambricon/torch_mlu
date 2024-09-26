#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/philox_utils.h"
#include "framework/generator/generator_impl.h"

#include "ATen/Generator.h"
#include "ATen/Utils.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_normal_internal(
    at::Tensor& output,
    double mean,
    double std,
    c10::optional<at::Generator> gen) {
  size_t output_num = static_cast<size_t>(output.numel());
  if (output_num == 0) {
    return output;
  }

  // prepare output tensor
  auto* output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);

  // get current handle
  auto handle = getCurrentHandle();

  auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(
      gen, getDefaultMLUGenerator());
  const int64_t nelem = output.numel();
  cnnlRandRngType_t rng_type = CNNL_RAND_RNG_PHILOX;
  PhiloxMLUState rng_engine_inputs;
  int thread_num = 0;
  TORCH_CNNL_CHECK(
      cnnlGetRandSimulateThreadNum_v2(handle, rng_type, &thread_num));
  auto counter_offset = calc_counter_offset(nelem, (int64_t)thread_num);
  {
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    rng_engine_inputs = gen_impl->philox_mlu_state(counter_offset);
  }

  if (rng_engine_inputs.captured_) {
    TORCH_CNNL_CHECK(cnnlGenerateRandNormal(
        handle,
        rng_type,
        true,
        0,
        0,
        rng_engine_inputs.seed_.ptr,
        rng_engine_inputs.offset_.ptr,
        rng_engine_inputs.offset_intragraph_,
        mean,
        std,
        output_desc.desc(),
        output_ptr));
  } else {
    TORCH_CNNL_CHECK(cnnlGenerateRandNormal(
        handle,
        rng_type,
        false,
        rng_engine_inputs.seed_.val,
        rng_engine_inputs.offset_.val,
        nullptr,
        nullptr,
        0,
        mean,
        std,
        output_desc.desc(),
        output_ptr));
  }
  return output;
}

} // namespace ops
} // namespace torch_mlu
