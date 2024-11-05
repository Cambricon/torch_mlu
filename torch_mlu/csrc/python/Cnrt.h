#pragma once
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch_mlu {

void initCnrtBindings(PyObject* module);

} // namespace torch_mlu
