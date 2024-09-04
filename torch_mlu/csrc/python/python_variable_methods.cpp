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

#include <ATen/Parallel.h>

#include "torch/csrc/python_headers.h" // the python headers should be first included
#include "python/python_variable_methods.h"
#include "python/Stream.h"
#include "python/Event.h"
#include "python/Graph.h"
#include "python/StorageSharing.h"
#include "python/ProcessGroupCNCL.h"
#include "python/Autocast.h"
#include "python/Cnpx.h"

#include "c10/core/Device.h"
#include "c10/util/Optional.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/variable.h"

#include "framework/core/memory_allocator.h"
#include "framework/hooks/MLUHooks.h"
#include "aten/operators/cnnl/internal/cnfft_plan_cache.h"
#include "framework/core/caching_allocator.h"
#include "framework/core/device.h"
#include "aten/operators/cnnl/convolution_utils.h"
#include "framework/core/device_utils.h"
#include "utils/version.h"
#include "utils/cndumper.h"
#include "python/THMP.h"
#include "aten/operators/cnnl/transformers/sdp_utils.h"

#if USE_PROFILE
#include "profile_mlu.h" // NOLINT
#endif

PyObject* module;
static std::vector<PyMethodDef> methods;

namespace torch_mlu {
namespace {

void PythonVariableMethods(py::module& m) {
  THMPStream_init(m.ptr());
  THMPEvent_init(m.ptr());
  THMPGraph_init(m.ptr());
  THMPProcessGroupCNCL_init(m.ptr());
  THMPModule_methods(m.ptr());
  registerMLUDeviceProperties(m.ptr());
  registerMluAllocator(m.ptr());
  initCnpxBindings(m.ptr());
  THMPStorage_Sharing_methods(m.ptr());

  // Device Management.
  m.def("_current_device", []() -> int {
    return static_cast<int>(torch_mlu::current_device());
  });
  // Memory Management.
  m.def("_mluHostAllocator", []() -> uint64_t {
    return reinterpret_cast<uint64_t>(torch_mlu::getMLUCachingHostAllocator());
  });
  m.def("_mlu_mem_get_info", [](int device) -> std::pair<size_t, size_t> {
    return torch_mlu::MLUCachingAllocator::MemGetInfo(device);
  });

  // TF32 mode management
  m.def("_get_cnnl_allow_tf32", []() -> bool {
    return torch_mlu::Global::instance().allowCNNLTF32();
  });
  m.def("_set_cnnl_allow_tf32", [](bool b) {
    torch_mlu::Global::instance().setAllowCNNLTF32(b);
  });
  m.def("_get_mlu_custom_allow_tf32", []() -> bool {
    return torch_mlu::Global::instance().allowMLUCustomTF32();
  });
  m.def("_set_mlu_custom_allow_tf32", [](bool b) {
    torch_mlu::Global::instance().setAllowMLUCustomTF32(b);
  });
  m.def("_get_cnmatmul_allow_tf32", []() -> bool {
    return torch_mlu::Global::instance().allowTF32CnMatMul();
  });
  m.def("_set_cnmatmul_allow_tf32", [](bool b) {
    torch_mlu::Global::instance().setAllowTF32CnMatMul(b);
  });

  // Fusion Ops Management, currently only torch.nn.LSTM.
  m.def("_get_mlufusion_enabled", []() -> bool {
    return torch_mlu::Global::instance().allowOpFusion();
  });
  m.def("_set_mlufusion_enabled", [](bool b) {
    torch_mlu::Global::instance().setAllowOpFusion(b);
  });

  // Dumptools API
  m.def("_dump_cnnl_gencase", [&](int mode) {
    torch_mlu::_dump_cnnl_gencase(mode);
  });
  m.def("_get_version", []() { return torch_mlu::getVersion(); });

  // The default logic is to return the memory format of the Contiguous type,
  // but to align with the eager mode, it needs to be changed to default to
  // returning the ChannelsLast memory format.
  m.def(
      "_conv_determine_backend_memory_format",
      torch_mlu::ops::_determine_backend_memory_format);

// profiler API
#if USE_PROFILE
  m.def("_enable_mlu_profiler", &torch_mlu::profiler::enableMluProfiler);
#endif
}
} // namespace

} // namespace torch_mlu

PyObject* initMLUModule() {
  at::internal::lazy_init_num_threads();
  THPUtils_addPyMethodDefs(methods, autocast_functions());
  static struct PyModuleDef mlu_module = {
      PyModuleDef_HEAD_INIT, "torch_mlu._MLUC", nullptr, -1, methods.data()};
  module = PyModule_Create(&mlu_module);
  auto py_module = py::reinterpret_borrow<py::module>(module);
  torch_mlu::PythonVariableMethods(py_module);

  // Scaled Dot Product Attention utilities
  py::class_<torch_mlu::sdp::sdp_params>(py_module, "_SDPAParams")
      .def(py::init([](at::Tensor const& query,
                       at::Tensor const& key,
                       at::Tensor const& value,
                       c10::optional<at::Tensor> attn_mask,
                       double dropout,
                       bool is_causal) {
        return torch_mlu::sdp::sdp_params{
            query, key, value, std::move(attn_mask), dropout, is_causal};
      }))
      .def_readonly("query", &torch_mlu::sdp::sdp_params::query)
      .def_readonly("key", &torch_mlu::sdp::sdp_params::key)
      .def_readonly("value", &torch_mlu::sdp::sdp_params::value)
      .def_readonly("attn_mask", &torch_mlu::sdp::sdp_params::attn_mask)
      .def_readonly("dropout", &torch_mlu::sdp::sdp_params::dropout)
      .def_readonly("is_causal", &torch_mlu::sdp::sdp_params::is_causal);

  py::enum_<torch_mlu::sdp::SDPBackend>(
      py_module,
      "_SDPBackend",
      "An enum-like class that contains the different backends for scaled dot product attention.\n\n... warning:: This class is in beta and subject to change.\n\n"
      "This backend class is designed to be used with the sdpa_kernel context manager."
      "See :func: torch.nn.attention.sdpa_kernel for more details.")
      .value("ERROR", torch_mlu::sdp::SDPBackend::error)
      .value("MATH", torch_mlu::sdp::SDPBackend::math)
      .value("FLASH_ATTENTION", torch_mlu::sdp::SDPBackend::flash_attention)
      .value(
          "EFFICIENT_ATTENTION",
          torch_mlu::sdp::SDPBackend::efficient_attention);

  py_module.def(
      "_can_use_flash_attention",
      [](const torch_mlu::sdp::sdp_params& params, bool debug) {
        return torch_mlu::sdp::can_use_flash_attention(params, debug);
      });
  py_module.def(
      "_can_use_mem_efficient_attention",
      [](const torch_mlu::sdp::sdp_params& params, bool debug) {
        return torch_mlu::sdp::can_use_mem_efficient_attention(params, debug);
      });
  return module;
}
