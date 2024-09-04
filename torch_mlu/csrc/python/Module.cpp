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

#include <pybind11/pybind11.h>
#include <array>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <sstream>
#include <ATen/ATen.h>
#include <libshm.h>
#include <ATen/MapAllocator.h>
#include <python/THMP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/mlu_lazy_init.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/cuda/python_comm.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/python_headers.h>
#include "framework/core/MLUStream.h"
#include "framework/core/device.h"
#include "framework/core/device_utils.h"
#include "framework/core/caching_allocator.h"
#include "framework/core/caching_allocator_config.h"
#include "framework/core/memory_snapshot.h"
#include "framework/distributed/cncl_utils.h"
#include "framework/generator/generator_impl.h" // MLU generator
#include "framework/graphs/MLUGraph.h"
#include "python/PythonTensor.h"
#include "python/combined_traceback.h"
#include "python/MluIPCTypes.h"

static bool in_bad_fork = false; // True for children forked after mlu init

#ifndef WIN32
// Called in the forked child if mlu has already been initialized
static void forked_child() {
  in_bad_fork = true;
  torch::utils::set_requires_mlu_init(true);
}
#endif

// Should be called before the first mlu call.
// Note: This is distinct from initExtension because a stub mlu implementation
// has some working functions (e.g. device_count) but cannot fully initialize.
static void poison_fork() {
#ifndef WIN32
  static std::once_flag flag;
  std::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

static at::Tensor dispatch_to(
    const at::Tensor& self,
    Device device,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing.
  // However, the behavior of aten::to is different with respect to
  // TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they
  // should be populated with the default values (eg. float for scalar type). By
  // explicitly copying over the tensor options here we fully specify all tensor
  // options and thus record the proper trace
  return self.to(
      self.options().device(device).memory_format(optional_memory_format),
      non_blocking,
      copy);
}

// Callback for python part. Used for additional initialization of python
// classes
static PyObject* THMPModule_initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!in_bad_fork); // Handled at python level
  poison_fork();
  at::globalContext().lazyInitMLU();
  auto m = THPObjectPtr(PyImport_ImportModule("torch.mlu"));
  if (!m)
    throw python_error();
  THMPStorage_postInit(m);

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  auto num_mlus = torch_mlu::device_count();
  auto default_mlu_generators = PyTuple_New(static_cast<Py_ssize_t>(num_mlus));
  for (int i = 0; i < num_mlus; i++) {
    auto gen = torch_mlu::getDefaultMLUGenerator(i);
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_mlu_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_mlu_generators);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Callback for python part. Used for additional initialization of python
// classes. Note: Unlike THMPModule_initExtension above, what
// THPModule_initExtension_mlu does is device-independent(so far, only
// THMPStorage_postInit is called), and they have the same relationship to each
// other as THPModule_initExtension and THCPModule_initExtension.
static PyObject* THPModule_initExtension_mlu(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS

  torch_mlu::tensors::initialize_python_bindings();
  auto module = THPObjectPtr(PyImport_ImportModule("torch.mlu"));
  if (!module)
    throw python_error();

  THMPStorage_postInit(module);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_getCurrentMLUStream_wrap(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index),
      "invalid argument to getCurrentMLUStream");
  int64_t device = THPUtils_unpackLong(device_index);
  auto stream = torch_mlu::getCurrentMLUStream(device);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple,
      1,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_getCurrentMLUStream_raw(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index),
      "invalid argument to getCurrentMLUStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromVoidPtr(torch_mlu::getCurrentMLUStream(device).stream());
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_getDefaultStream_wrap(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index),
      "invalid argument to getDefaultMLUStream");
  int64_t device = THPUtils_unpackLong(device_index);
  auto stream = torch_mlu::getDefaultMLUStream(device);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple,
      1,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_getCnclStream_wrap(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getCnclStream");
  int64_t device = THPUtils_unpackLong(device_index);
  auto streams = torch_mlu::getCnclStream(device);
  PyObject* list_streams = PyList_New(0);
  for (const auto& pair : streams) {
    const std::string& clique_id_data = pair.first;
    const torch_mlu::MLUStream stream = pair.second;
    PyObject* output_tuple = PyTuple_New(4);
    PyTuple_SetItem(
        output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
    PyTuple_SetItem(
        output_tuple,
        1,
        THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
    PyTuple_SetItem(
        output_tuple,
        2,
        THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
    PyTuple_SetItem(
        output_tuple,
        3,
        PyBytes_FromStringAndSize(
            clique_id_data.data(), clique_id_data.size()));
    PyList_Append(list_streams, output_tuple);
    Py_DECREF(output_tuple);
  }
  return list_streams;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_setStream_wrap(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  auto stream = torch_mlu::MLUStream::unpack3(
      stream_id, device_index, static_cast<c10::DeviceType>(device_type));

  auto device = static_cast<int>(torch_mlu::current_device());
  if (device != stream.device_index()) {
    torch_mlu::setDevice(static_cast<c10::DeviceIndex>(stream.device_index()));
  }
  torch_mlu::setCurrentMLUStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_canDeviceAccessPeer_wrap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* arg1 = nullptr;
  PyObject* arg2 = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "can_device_peer_access",
        1,
        "(int device, int peer_device);");
    return nullptr;
  }
  THPUtils_assert(
      THPUtils_checkLong(arg1), "invalid argument to canDeviceAccessPeer");
  THPUtils_assert(
      THPUtils_checkLong(arg2), "invalid argument to canDeviceAccessPeer");
  int64_t device = THPUtils_unpackLong(arg1);
  int64_t peer_device = THPUtils_unpackLong(arg2);

  torch::utils::mlu_lazy_init();
  auto can_access = torch_mlu::canDeviceAccessPeer(device, peer_device);
  return PyBool_FromLong(can_access);
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::mlu_lazy_init();
  auto device = static_cast<int>(torch_mlu::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  poison_fork();
  return THPUtils_packUInt64(torch_mlu::device_count());
  END_HANDLE_TH_ERRORS
}

torch::CapturedTraceback* getFromContext(
    const std::shared_ptr<c10::GatheredContext>& x) {
  if (torch::CapturedTraceback* sc =
          dynamic_cast<torch::CapturedTraceback*>(x.get())) {
    return sc;
  }
  TORCH_CHECK(
      false,
      "attempting to gather stack context from the wrong StackContext type.");
}

PyObject* THMPModule_setMemoryFraction(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* fraction_o = nullptr;
  PyObject* device_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &fraction_o, &device_o)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "set_memory_fraction",
        1,
        "(double fraction, int device);");
    return nullptr;
  }
  double fraction = PyFloat_AsDouble(fraction_o);
  int64_t device = PyLong_AsLongLong(device_o);

  torch_mlu::MLUCachingAllocator::setMemoryFraction(fraction, device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THMPModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch_mlu::MLUCachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THMPModule_memoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  const int device = (int)THPUtils_unpackLong(arg);

  using torch_mlu::MLUCachingAllocator::MemoryStats;
  using torch_mlu::MLUCachingAllocator::Stat;
  using torch_mlu::MLUCachingAllocator::StatArray;
  using torch_mlu::MLUCachingAllocator::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const MemoryStats stats =
      torch_mlu::MLUCachingAllocator::getMemoryStats(device);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["max_split_size"] = stats.max_split_size;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["requested_bytes"] = statArrayToDict(stats.requested_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_resetAccumulatedMemoryStats(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "invalid argument to reset_accumulated_memory_stats");
  const int device = (int)THPUtils_unpackLong(arg);
  torch_mlu::MLUCachingAllocator::resetAccumulatedStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THMPModule_resetPeakMemoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const int device = (int)THPUtils_unpackLong(arg);
  torch_mlu::MLUCachingAllocator::resetPeakStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THMPModule_memorySnapshot(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS

  using torch_mlu::MLUCachingAllocator::ChunkInfo;
  using torch_mlu::MLUCachingAllocator::SegmentInfo;

  py::str device_s = "device";
  py::str address_s = "address";
  py::str total_size_s = "total_size";
  py::str allocated_size_s = "allocated_size";
  py::str active_size_s = "active_size";
  py::str requested_size_s = "requested_size";
  py::str stream_s = "stream";
  py::str segment_type_s = "segment_type";
  py::str segment_pool_id = "segment_pool_id";
  py::str large_s = "large";
  py::str small_s = "small";
  py::str size_s = "size";
  py::str state_s = "state";
  py::str active_allocated_s = "active_allocated";
  py::str active_pending_free_s = "active_pending_free";
  py::str inactive_s = "inactive";
  py::str addr_s = "addr";
  py::str cpp_frames_s = "cpp_frames";
  py::str blocks_s = "blocks";
  py::str is_expandable_s = "is_expandable";
  py::str frames_s = "frames";

  py::list empty_frames;
  std::vector<torch::CapturedTraceback*> to_gather_frames;
  std::vector<py::dict> to_gather_dest;

  auto add_frame_key = [&](const py::dict& d,
                           const std::shared_ptr<c10::GatheredContext>& ctx) {
    if (ctx) {
      auto sc = getFromContext(ctx);
      to_gather_frames.emplace_back(sc);
      to_gather_dest.emplace_back(d);
    } else {
      d[frames_s] = empty_frames;
    }
  };

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict[device_s] = segmentInfo.device;
    segmentDict[address_s] = segmentInfo.address;
    segmentDict[total_size_s] = segmentInfo.total_size;
    segmentDict[allocated_size_s] = segmentInfo.allocated_size;
    segmentDict[active_size_s] = segmentInfo.active_size;
    segmentDict[requested_size_s] = segmentInfo.requested_size;
    // we want the python objects to pickle easily so use an int to
    // represent the stream rather than a torch.cuda.stream object
    segmentDict[stream_s] = int64_t(segmentInfo.stream);
    segmentDict[segment_type_s] = (segmentInfo.is_large ? large_s : small_s);
    segmentDict[segment_pool_id] = segmentInfo.owner_private_pool_id;
    segmentDict[is_expandable_s] = segmentInfo.is_expandable;
    add_frame_key(segmentDict, segmentInfo.context_when_allocated);

    auto address = segmentInfo.address;
    py::list chunks;
    for (const auto& chunkInfo : segmentInfo.chunks) {
      py::dict chunkDict;
      chunkDict[address_s] = address;
      chunkDict[size_s] = chunkInfo.size;
      chunkDict[requested_size_s] = chunkInfo.requested_size;
      chunkDict[state_s] =
          (chunkInfo.allocated
               ? active_allocated_s
               : (chunkInfo.active ? active_pending_free_s : inactive_s));
      add_frame_key(chunkDict, chunkInfo.context_when_allocated);
      chunks.append(chunkDict);
      address += chunkInfo.size;
    }
    segmentDict[blocks_s] = chunks;

    return segmentDict;
  };

  auto snapshot = torch_mlu::MLUCachingAllocator::snapshot();

  py::list segments;

  for (const auto& segmentInfo : snapshot.segments) {
    segments.append(segmentInfoToDict(segmentInfo));
  }

  py::list traces;
  py::str action_s = "action";
  py::str alloc_s = "alloc";
  py::str free_requested_s = "free_requested";
  py::str free_completed_s = "free_completed";
  py::str segment_alloc_s = "segment_alloc";
  py::str segment_free_s = "segment_free";
  py::str segment_map_s = "segment_map";
  py::str segment_unmap_s = "segment_unmap";

  py::str snapshot_s = "snapshot";
  py::str oom_s = "oom";
  py::str device_free_s = "device_free";

  using namespace torch_mlu::MLUCachingAllocator;

  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
      case TraceEntry::SEGMENT_UNMAP:
        return segment_unmap_s;
      case TraceEntry::SEGMENT_MAP:
        return segment_map_s;
    }
    throw std::runtime_error("unreachable");
  };

  for (const auto& traceInfo : snapshot.device_traces) {
    py::list trace;
    for (const auto& te : traceInfo) {
      py::dict trace_entry;
      if (te.context_) {
        // without further compression frames can get really large on dump
        auto sc = getFromContext(te.context_);
        to_gather_frames.emplace_back(sc);
        to_gather_dest.emplace_back(trace_entry);
      }
      trace_entry[action_s] = action_to_str(te.action_);
      trace_entry[TraceEntry::OOM == te.action_ ? device_free_s : addr_s] =
          te.addr_;
      trace_entry[size_s] = te.size_;
      trace_entry[stream_s] = int64_t(te.stream_);
      trace.append(trace_entry);
    }
    traces.append(trace);
  }

  py::dict allocator_settings;
  py::str last_allocator_settings_s = "PYTORCH_MLU_ALLOC_CONF";
  py::str max_split_size_s = "max_split_size";
  py::str garbage_collection_threshold_s = "garbage_collection_threshold";
  py::str expandable_segments_s = "expandable_segments";
  py::str roundup_power2_divisions_s = "roundup_power2_divisions";

  allocator_settings[last_allocator_settings_s] =
      snapshot.config_metadata.last_allocator_settings;
  allocator_settings[max_split_size_s] =
      int64_t(snapshot.config_metadata.max_split_size);
  allocator_settings[garbage_collection_threshold_s] =
      snapshot.config_metadata.garbage_collection_threshold;
  allocator_settings[expandable_segments_s] =
      snapshot.config_metadata.expandable_segments;
  unsigned int roundup_key = 1;
  py::dict roundup_settings;
  for (const auto& v : snapshot.config_metadata.roundup_power2_divisions) {
    py::str roundup_key_s = std::to_string(roundup_key);
    roundup_settings[roundup_key_s] = int64_t(v);
    roundup_key *= 2;
  }
  allocator_settings[roundup_power2_divisions_s] = roundup_settings;

  py::dict result;
  result["segments"] = segments;
  result["device_traces"] = traces;
  result["allocator_settings"] = allocator_settings;

  auto frames = torch_mlu::py_symbolize(to_gather_frames);
  for (auto i : c10::irange(frames.size())) {
    to_gather_dest.at(i)[frames_s] = frames.at(i);
  }

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_mluCachingAllocator_raw_alloc(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* size_o = nullptr;
  PyObject* stream_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &size_o, &stream_o)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "caching_allocator_alloc",
        1,
        "(ssize_t size, intptr_t stream);");
    return nullptr;
  }
  auto size = PyLong_AsSsize_t(size_o);
  cnrtQueue_t stream = static_cast<cnrtQueue_t>(PyLong_AsVoidPtr(stream_o));
  void* mem = nullptr;
  {
    pybind11::gil_scoped_release no_gil;
    mem = torch_mlu::MLUCachingAllocator::raw_alloc_with_stream(size, stream);
  }
  return PyLong_FromVoidPtr(mem);
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_mluCachingAllocator_raw_delete(
    PyObject* _unused,
    PyObject* obj) {
  HANDLE_TH_ERRORS
  void* mem_ptr = PyLong_AsVoidPtr(obj);
  {
    pybind11::gil_scoped_release no_gil;
    torch_mlu::MLUCachingAllocator::raw_delete(mem_ptr);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_mluCachingAllocator_set_allocator_settings(
    PyObject* _unused,
    PyObject* env) {
  HANDLE_TH_ERRORS
  torch_mlu::MLUCachingAllocator::setAllocatorSettings(
      THPUtils_unpackString(env));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_getAllocatorBackend(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(torch_mlu::MLUCachingAllocator::name());
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_mlu(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {"mlu(Tensor self, Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
       "mlu(Tensor self, Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"});
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  auto device =
      r.isNone(1) ? at::Device(at::DeviceType::PrivateUse1) : r.device(1);
  auto opt_memory_format = r.memoryformatOptional(3);
  TORCH_CHECK(device.is_privateuseone(), "Invalid device, must be mlu device");
  torch::utils::mlu_lazy_init();
  return THPVariable_Wrap(
      dispatch_to(self_, device, r.toBool(2), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPModule_isCurrentStreamCapturing_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // If there's no mlu context, torch_mlu::currentStreamCaptureStatus returns
  // CaptureStatus::None without initializing a context.
  if (torch_mlu::currentStreamCaptureStatus() ==
      torch_mlu::CaptureStatus::None) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_mluIPCCollect(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch_mlu::MluIPCCollect();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPModule_hasPrimaryContext(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to has_primary_context");
  int64_t device_index = static_cast<int64_t>(THPUtils_unpackLong(arg));
  if (torch_mlu::hasPrimaryContext(device_index)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef _THMPModule_methods[] = {
    {"_initExtension", THPModule_initExtension_mlu, METH_NOARGS, nullptr},
    {"_mlu_init", THMPModule_initExtension, METH_NOARGS, nullptr},
    {"_mlu_getCurrentMLUStream",
     THMPModule_getCurrentMLUStream_wrap,
     METH_O,
     nullptr},
    {"_mlu_getCurrentRawStream",
     THMPModule_getCurrentMLUStream_raw,
     METH_O,
     nullptr},
    {"_mlu_getDefaultStream",
     THMPModule_getDefaultStream_wrap,
     METH_O,
     nullptr},
    {"_mlu_getCnclStream", THMPModule_getCnclStream_wrap, METH_O, nullptr},
    {"_mlu_setStream",
     castPyCFunctionWithKeywords(THMPModule_setStream_wrap),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_mlu_isCurrentStreamCapturing",
     THMPModule_isCurrentStreamCapturing_wrap,
     METH_NOARGS,
     nullptr},
    {"_mlu_setDevice", THMPModule_setDevice_wrap, METH_O, nullptr},
    {"_mlu_ipc_collect", THMPModule_mluIPCCollect, METH_NOARGS, nullptr},
    {"_mlu_canDeviceAccessPeer",
     THMPModule_canDeviceAccessPeer_wrap,
     METH_VARARGS,
     nullptr},
    {"_mlu_getDevice", THMPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_mlu_getDeviceCount",
     THMPModule_getDeviceCount_wrap,
     METH_NOARGS,
     nullptr},
    {"_mlu_setMemoryFraction",
     THMPModule_setMemoryFraction,
     METH_VARARGS,
     nullptr},
    {"_mlu_emptyCache", THMPModule_emptyCache, METH_NOARGS, nullptr},
    {"_mlu_memoryStats", THMPModule_memoryStats, METH_O, nullptr},
    {"_mlu_resetAccumulatedMemoryStats",
     THMPModule_resetAccumulatedMemoryStats,
     METH_O,
     nullptr},
    {"_mlu_resetPeakMemoryStats",
     THMPModule_resetPeakMemoryStats,
     METH_O,
     nullptr},
    {"_mlu_memorySnapshot", THMPModule_memorySnapshot, METH_NOARGS, nullptr},
    {"_mlu_mluCachingAllocator_raw_alloc",
     THMPModule_mluCachingAllocator_raw_alloc,
     METH_VARARGS,
     nullptr},
    {"_mlu_mluCachingAllocator_raw_delete",
     THMPModule_mluCachingAllocator_raw_delete,
     METH_O,
     nullptr},
    {"_mlu_mluCachingAllocator_set_allocator_settings",
     THMPModule_mluCachingAllocator_set_allocator_settings,
     METH_O,
     nullptr},
    {"_mlu_getAllocatorBackend",
     THMPModule_getAllocatorBackend,
     METH_NOARGS,
     nullptr},
    {"_mlu_isInBadFork", THMPModule_isInBadFork, METH_NOARGS, nullptr},
    {"mlu",
     castPyCFunctionWithKeywords(THPVariable_mlu),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_mlu_hasPrimaryContext", THMPModule_hasPrimaryContext, METH_O, nullptr},
    {nullptr}};

void THMPModule_methods(PyObject* module) {
  if (PyModule_AddFunctions(module, _THMPModule_methods) < 0) {
    throw python_error();
  }
}

void THMPModule_initModule() {
  THMPModule_initExtension(nullptr, nullptr);
}

PyObject* THMPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to setDevice");
  int64_t device = THPUtils_unpackLong(arg);

  torch::utils::mlu_lazy_init();
  torch_mlu::setDevice(static_cast<c10::DeviceIndex>(device));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

void registerMLUDeviceProperties(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  pybind11::class_<torch_mlu::DeviceProp>(m, "_MLUDeviceProperties")
      .def_readonly("name", &torch_mlu::DeviceProp::name)
      .def_readonly("major", &torch_mlu::DeviceProp::major)
      .def_readonly("minor", &torch_mlu::DeviceProp::minor)
      .def_readonly(
          "multi_processor_count",
          &torch_mlu::DeviceProp::multi_processor_count)
      .def_readonly("cluster_count", &torch_mlu::DeviceProp::cluster_count)
      .def_readonly(
          "core_num_per_cluster", &torch_mlu::DeviceProp::core_num_per_cluster)
      .def_readonly("total_memory", &torch_mlu::DeviceProp::total_memory)
      .def_readonly("frequency", &torch_mlu::DeviceProp::frequency)
      .def_readonly("dram_bandwidth", &torch_mlu::DeviceProp::dram_bandwidth)
      .def_readonly(
          "supports_linear_memory",
          &torch_mlu::DeviceProp::supports_linear_memory)
      .def("__repr__", [](const torch_mlu::DeviceProp& prop) {
        std::ostringstream stream;
        stream << "_MLUDeviceProperties(name='" << prop.name
               << "', major=" << prop.major << ", minor=" << prop.minor
               << ", total_memory=" << prop.total_memory / (1024 * 1024)
               << "MB, multi_processor_count=" << prop.multi_processor_count
               << ")";
        return stream.str();
      });

  m.def(
      "_mlu_record_memory_history_legacy",
      static_cast<void (*)(bool, bool, int64_t, bool, bool)>(
          torch_mlu::_record_memory_history));

  m.def(
      "_mlu_record_memory_history",
      static_cast<void (*)(
          c10::optional<std::string>,
          c10::optional<std::string>,
          std::string,
          size_t)>(torch_mlu::_record_memory_history));

  m.def("_mlu_isHistoryEnabled", []() {
    return torch_mlu::MLUCachingAllocator::isHistoryEnabled();
  });
}

// We choose to ignore certain blocks that are currently allocated
// when we set the pool to its checkpoint. For those blocks, we need
// to swap out the deleter function of their corresponding blocks
// so that a deallocation is not triggered when they die.
void removeStorageDeleterFns(
    const std::vector<c10::StorageImpl*>& stale_live_storages,
    std::unordered_set<void*> definitely_stale_pointers) {
  for (c10::StorageImpl* stale_storage : stale_live_storages) {
    auto ptr = stale_storage->data_ptr().get();
    auto allocated_pointer = definitely_stale_pointers.find(ptr);
    TORCH_CHECK(allocated_pointer != definitely_stale_pointers.end());
    auto t = torch_mlu::MLUCachingAllocator::get();
    bool succeeded = stale_storage->mutable_data_ptr().compare_exchange_deleter(
        t->raw_deleter(), &c10::detail::deleteNothing);

    TORCH_CHECK(
        succeeded,
        "Unexpected deleter function on storage, could not swap function");
  }
}

void addStorageDeleterFns(
    std::vector<c10::StorageImpl*>& storages_to_add_deleters_to,
    torch_mlu::MLUCachingAllocator::CheckpointDelta& delta) {
  std::unordered_map<void*, c10::StorageImpl*> storages;
  for (auto& storage : storages_to_add_deleters_to) {
    storages[storage->data_ptr().get()] = storage;
  }

  for (auto& data_ptr : delta.dataptrs_allocd) {
    auto storage_pair = storages.find(data_ptr.get());
    if (storage_pair != storages.end()) {
      auto ctx = storage_pair->second->data_ptr().get_context();
      TORCH_CHECK(ctx == nullptr, " Not expecting deleter function");
      storage_pair->second->set_data_ptr_noswap(std::move(data_ptr));
    } else {
      data_ptr.release_context();
    }
  }
}

// expose allocator API to Python
void registerMluAllocator(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<
      torch_mlu::MLUCachingAllocator::AllocatorState,
      std::shared_ptr<torch_mlu::MLUCachingAllocator::AllocatorState>>(
      m, "_mlu_MLUAllocator_AllocatorState");

  m.def("_mlu_getCheckpointState", [](int device, torch_mlu::MempoolId_t id) {
    return torch_mlu::MLUCachingAllocator::getCheckpointState(device, id);
  });

  m.def(
      "_mlu_beginAllocateCurrentStreamToPool",
      [](int device, torch_mlu::MempoolId_t mempool_id) {
        auto stream = torch_mlu::getCurrentMLUStream(device);
        TORCH_CHECK(stream, "Expected stream capture to be under way");
        torch_mlu::MLUCachingAllocator::beginAllocateToPool(
            device, mempool_id, [stream](cnrtQueue_t target) {
              return target == stream;
            });
      });

  m.def(
      "_mlu_endAllocateCurrentStreamToPool",
      [](int device, torch_mlu::MempoolId_t mempool_id) {
        torch_mlu::MLUCachingAllocator::endAllocateToPool(device, mempool_id);
      });

  m.def("_mlu_releasePool", [](int device, torch_mlu::MempoolId_t mempool_id) {
    torch_mlu::MLUCachingAllocator::releasePool(device, mempool_id);
  });

  m.def(
      "_mlu_checkPoolLiveAllocations",
      [](int device,
         torch_mlu::MempoolId_t mempool_id,
         const py::set& expected_live_allocations) {
        std::unordered_set<void*> allocations;
        allocations.reserve(expected_live_allocations.size());
        for (auto& elem : expected_live_allocations) {
          allocations.insert(reinterpret_cast<void*>(py::cast<size_t>(elem)));
        }
        return torch_mlu::MLUCachingAllocator::checkPoolLiveAllocations(
            device, mempool_id, allocations);
      });

  m.def(
      "_mlu_setCheckpointPoolState",
      [](int device,
         std::shared_ptr<torch_mlu::MLUCachingAllocator::AllocatorState> pps,
         const std::vector<size_t>& stale_storages_ptr,
         const std::vector<size_t>& storages_to_add_deleters_to_ptr = {}) {
        std::unordered_set<c10::StorageImpl*> ptr_set;
        // iterate on std::vector for determinism
        std::vector<c10::StorageImpl*> ptrs;
        for (size_t ptr_int : stale_storages_ptr) {
          c10::StorageImpl* ptr = (c10::StorageImpl*)ptr_int;
          if (!ptr_set.count(ptr)) {
            ptrs.push_back(ptr);
            ptr_set.insert(ptr);
          }
        }
        auto delta = torch_mlu::MLUCachingAllocator::setCheckpointPoolState(
            device, std::move(pps));
        auto& freed_pointers = delta.ptrs_freed;

        std::unordered_set<void*> allocd_set;
        for (auto& data_ptr : delta.dataptrs_allocd) {
          allocd_set.insert(data_ptr.get());
        }
        std::unordered_set<void*> freed_pointer_set;
        size_t definite_freed_count = 0;
        for (void* ptr : freed_pointers) {
          if (!allocd_set.count(ptr)) {
            definite_freed_count += 1;
          }
          freed_pointer_set.insert((ptr));
        }
        // that block has already been freed,
        // so even those this will error, so too will the allocator
        // when the corresponding tensor dies because there is no
        // live tensor corresponding to it
        TORCH_CHECK(
            ptr_set.size() >= definite_freed_count,
            "Any stale tensors which are being manually freed"
            " must be passed to set checkpoint");

        removeStorageDeleterFns(ptrs, freed_pointer_set);
        std::vector<c10::StorageImpl*> storages_to_add_deleters_to;
        storages_to_add_deleters_to.reserve(
            storages_to_add_deleters_to_ptr.size());
        for (size_t ptr_int : storages_to_add_deleters_to_ptr) {
          storages_to_add_deleters_to.push_back((c10::StorageImpl*)ptr_int);
        }

        addStorageDeleterFns(storages_to_add_deleters_to, delta);
      });
}
