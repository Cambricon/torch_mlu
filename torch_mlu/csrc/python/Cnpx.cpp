#include "Cnpx.h"
#include "cnpx.h"

namespace torch_mlu {
void initCnpxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cnpx = m.def_submodule("_cnpx", "cnpx bindings");
  cnpx.def("rangePush", cnpxRangePush);
  cnpx.def("rangePop", cnpxRangePop);
  cnpx.def("rangeStart", [](const char* message) {
    void* handle_;
    cnpxRangeStart(message, &handle_);
    return handle_;
  });
  cnpx.def("rangeEnd", cnpxRangeEnd);
  cnpx.def("mark", cnpxMark);
}

} // namespace torch_mlu
