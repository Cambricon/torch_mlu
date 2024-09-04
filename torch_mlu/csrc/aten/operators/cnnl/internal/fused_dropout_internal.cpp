#include "framework/generator/generator_impl.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/philox_utils.h"

#include "ATen/Generator.h"
#include "ATen/Utils.h"

namespace torch_mlu {
namespace ops {

void fused_dropout_internal(
    at::Tensor& output,
    at::Tensor& mask,
    const at::Tensor& self,
    double p,
    c10::optional<at::Generator> gen) {
  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);
  auto mask_impl = getMluTensorImpl(mask);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto mask_ptr = mask_impl->mlu_data_ptr();
  // Get Cnnl Desc
  auto descInput = getTensorDescAndCoalesceDims(input_impl);
  auto descMask = getTensorDescAndCoalesceDims(mask_impl);
  auto descOutput = getTensorDescAndCoalesceDims(output_impl);

  auto handle = getCurrentHandle();
  auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(
      gen, getDefaultMLUGenerator());
  const int64_t nelem = self.numel();
  cnnlRandRngType_t rng_type = CNNL_RAND_RNG_PHILOX;
  PhiloxMLUState rng_engine_inputs;
  int thread_num = 0;
  TORCH_CNNL_CHECK(cnnlRandGetSimulateThreadNum(handle, &thread_num));
  auto counter_offset = calc_counter_offset(nelem, (int64_t)thread_num);
  {
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    rng_engine_inputs = gen_impl->philox_mlu_state(counter_offset);
  }

  if (rng_engine_inputs.captured_) {
    TORCH_CNNL_CHECK(cnnlFusedDropout_v3(
        handle,
        rng_type,
        descInput.get(),
        input_ptr,
        true,
        0,
        0,
        rng_engine_inputs.seed_.ptr,
        rng_engine_inputs.offset_.ptr,
        rng_engine_inputs.offset_intragraph_,
        1 - p,
        descMask.get(),
        mask_ptr,
        descOutput.get(),
        output_ptr));
  } else {
    TORCH_CNNL_CHECK(cnnlFusedDropout_v3(
        handle,
        rng_type,
        descInput.get(),
        input_ptr,
        false,
        rng_engine_inputs.seed_.val,
        rng_engine_inputs.offset_.val,
        nullptr,
        nullptr,
        0,
        1 - p,
        descMask.get(),
        mask_ptr,
        descOutput.get(),
        output_ptr));
  }
}

} // namespace ops
} // namespace torch_mlu
