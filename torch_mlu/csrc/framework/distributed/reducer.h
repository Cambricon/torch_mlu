#pragma once
#include <torch/csrc/distributed/c10d/reducer.hpp>

namespace torch_mlu {
class Reducer_mlu : public c10d::Reducer {
  using Reducer::Reducer;
  // NOTE: Inheriting from the Reducer class here is only to
  // shadow the following function. If you need to use this class, please be
  // extra cautious.
  void initialize_local_used_map();
};

} // namespace torch_mlu
