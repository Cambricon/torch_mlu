#include "Cnrt.h"
#include "cnrt.h"

namespace torch_mlu {

void initCnrtBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cnrt = m.def_submodule("_cnrt", "libcnrt.so bindings");

  py::enum_<cnrtRet_t>(
      cnrt,
      "mlu"
      "Error")
      .value("success", cnrtSuccess);
  cnrt.def(
      "mlu"
      "GetErrorStr",
      cnrtGetErrorStr);
  cnrt.def(
      "mlu"
      "ProfilerStart",
      cnrtProfilerStart);
  cnrt.def(
      "mlu"
      "ProfilerStop",
      cnrtProfilerStop);
}

} // namespace torch_mlu
