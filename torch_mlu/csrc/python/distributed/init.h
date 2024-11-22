#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include "framework/distributed/reducer.h"
#include "utils/Export.h"
namespace torch_mlu {

TORCH_MLU_API void initC10dMlu(PyObject* module);

class Logger_mlu : public c10d::Logger {
  using Logger::Logger;
};

} // namespace torch_mlu
