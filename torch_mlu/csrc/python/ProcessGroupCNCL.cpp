#include <torch/python.h>
#include <pybind11/chrono.h>
#include "ProcessGroupCNCL.h"
#include "framework/distributed/process_group_cncl.hpp"
#include "framework/distributed/cncl_utils.h"

using namespace pybind11::literals;
namespace {

// This code is copied from torch/csrc/distributed/c10d/init.cpp
// TODO: Remove this code after the source code has been moved somewhere more
// generally useful.
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_;

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(IntrusivePtrNoGilDestructor&&) =
      default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  C10_NODISCARD T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }
};
} // anonymous namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true);

namespace torch_mlu {

namespace {

std::tuple<int, int, int> get_cncl_version() {
  int major = -1, minor = -1, patch = -1;
  C10D_CNCL_CHECK(cnclGetLibVersion(&major, &minor, &patch), c10::nullopt);
  return std::make_tuple(major, minor, patch);
}

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ =
    py::class_<T, IntrusivePtrNoGilDestructor<T>>;

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

void init_internal(PyObject* module) {
  auto torch_c10d_module =
      THPObjectPtr(PyImport_ImportModule("torch.distributed"));
  if (!torch_c10d_module) {
    throw python_error();
  }

  auto torch_c10d_m = py::handle(torch_c10d_module).cast<py::module>();

  py::module_ dist = py::module_::import("torch._C._distributed_c10d");
  if (py::hasattr(torch_c10d_m, "ProcessGroup")) {
    auto processGroupCNCL =
        intrusive_ptr_no_gil_destructor_class_<ProcessGroupCNCL>(
            module, "ProcessGroupCNCL", dist.attr("Backend"))
            .def(
                py::init<
                    const c10::intrusive_ptr<::c10d::Store>&,
                    int,
                    int,
                    c10::intrusive_ptr<ProcessGroupCNCL::Options>>(),
                py::call_guard<py::gil_scoped_release>())
            .def(
                py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                            int rank,
                            int size,
                            const std::chrono::milliseconds& timeout) {
                  auto options = ProcessGroupCNCL::Options::create();
                  options->is_high_priority_stream = false;
                  options->timeout = timeout;
                  return c10::make_intrusive<ProcessGroupCNCL>(
                      store, rank, size, options);
                }),
                py::arg("store"),
                py::arg("rank"),
                py::arg("size"),
                py::arg("timeout") = kProcessGroupDefaultTimeout,
                py::call_guard<py::gil_scoped_release>())
            .def("get_cncl_comm", &ProcessGroupCNCL::getCnclComm)
            .def_property_readonly("options", &ProcessGroupCNCL::getOptions);

    intrusive_ptr_class_<ProcessGroupCNCL::Options>(
        processGroupCNCL,
        "Options",
        torch_c10d_m.attr("ProcessGroup").attr("Options"),
        R"(
    ProcessGroup options for the CNCL backend

    Arguments:
        is_high_priority_stream (bool, optional): flag to enable/disable process
                group to pick up high priority mlu streams. It lets MLU driver
                to prioritize CNCL kernels when there are compute kernels waiting.
                Default is False.

    Example::
        >>> import torch.distributed as dist
        >>>
        >>> nccl_options = dist.ProcessGroupCNCL.Options(is_high_priority_stream=True)
        >>> # initialize a nccl process group with the options just created
        >>> dist.init_process_group("nccl", pg_options=nccl_options)
        )")
        .def(py::init<bool>(), py::arg("is_high_priority_stream") = false)
        .def_readwrite(
            "is_high_priority_stream",
            &ProcessGroupCNCL::Options::is_high_priority_stream);
    // TODO(zhiguangda): support this
    // processGroupCNCL.def_static(
    //     "_group_start", []() { ProcessGroupCNCL::groupStart(); });
    // processGroupCNCL.def_static(
    //     "_group_end", []() { ProcessGroupCNCL::groupEnd(); });
  }

  auto mluc_m = py::handle(module).cast<py::module>();
  mluc_m.def("_cncl_version", &get_cncl_version);
}

} // namespace

__attribute__((visibility("default"))) void THMPProcessGroupCNCL_init(
    PyObject* module) {
  init_internal(module);
}

} // namespace torch_mlu
