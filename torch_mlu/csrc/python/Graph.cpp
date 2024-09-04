/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/pytorch/pytorch/graphs/contributors Redistribution and use in
source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <pybind11/chrono.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include "python/Graph.h"
#include "framework/graphs/MLUGraph.h"

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THMPGraph_init(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def("_graph_pool_handle", &torch_mlu::graph_pool_handle);

  shared_ptr_class_<torch_mlu::MLUGraph>(m, "_MLUGraph")
      .def(py::init<>())
      .def(
          "capture_begin",
          [](torch_mlu::MLUGraph& self,
             std::optional<torch_mlu::MempoolId_t> pool_opt,
             std::string capture_error_mode) {
            cnrtQueueCaptureMode capture_mode;
            torch_mlu::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : torch_mlu::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = cnrtQueueCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = cnrtQueueCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = cnrtQueueCaptureModeRelaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"),
          py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&torch_mlu::MLUGraph::capture_end))
      .def(
          "register_generator_state",
          [](torch_mlu::MLUGraph& self, py::handle raw_generator) {
            auto generator = THPGenerator_Unwrap(raw_generator.ptr());
            // We've unwrapped Python object to C++ object,
            // so we could release GIL before calling into C++
            py::gil_scoped_release release;
            return self.register_generator_state(generator);
          },
          py::arg("generator"))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&torch_mlu::MLUGraph::replay))
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&torch_mlu::MLUGraph::reset))
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&torch_mlu::MLUGraph::pool))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(&torch_mlu::MLUGraph::debug_dump))
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &torch_mlu::MLUGraph::enable_debug_mode))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(&torch_mlu::MLUGraph::debug_dump),
          py::arg("debug_path"));
}
