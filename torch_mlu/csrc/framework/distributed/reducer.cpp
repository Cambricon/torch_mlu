#include "framework/distributed/reducer.h"

namespace torch_mlu {
void Reducer_mlu::initialize_local_used_map() {
  const auto variable_count = params_.size();
  at::TensorOptions options;
  options = options.dtype(at::kInt);
  // TODO(zhiguangda): remove this when CNCL can garantee the performance of Int
  // type
  if (params_[0].device().is_privateuseone()) {
    options = options.dtype(at::kFloat);
  }

  // Deliberately don't pin the memory even if local_used_map_dev_ will
  // be cuda. See Note [local_used_map_ -> local_used_map_dev copying]
  local_used_map_ = at::zeros({static_cast<long>(variable_count)}, options);

  // This tensor needs to be on the same device as the replica params because
  // backend such as NCCL may not support CPU tensors, and hence it might not
  // work if we always put it on CPU. The dist backend for MTIA doesn't support
  // int32 allreduce for now, so it has to be placed on CPU.
  options = options.device(
      (params_[0].is_mtia()) ? c10::Device(c10::DeviceType::CPU)
                             : params_[0].device());
  local_used_map_dev_ = at::empty({static_cast<long>(variable_count)}, options);
}
} // namespace torch_mlu
