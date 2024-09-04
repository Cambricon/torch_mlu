#pragma once

#include <torch/csrc/python_headers.h>

namespace torch_mlu {
namespace tensors {

// Initializes the Python tensor type objects: torch.mlu.FloatTensor,
// torch.mlu.DoubleTensor, etc. and binds them in their containing modules.
void initialize_python_bindings();

} // namespace tensors
} // namespace torch_mlu
