#ifndef THMP_PROCESSGROUPCNCL_INC
#define THMP_PROCESSGROUPCNCL_INC
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch_mlu {

void THMPProcessGroupCNCL_init(PyObject* module);

} // namespace torch_mlu

#endif // THMP_PROCESSGROUPCNCL_INC