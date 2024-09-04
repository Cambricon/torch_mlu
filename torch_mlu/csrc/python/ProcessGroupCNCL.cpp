#include "ProcessGroupCNCL.h"
#include "framework/distributed/process_group_cncl.hpp"
#include "framework/distributed/cncl_utils.h"

using namespace pybind11::literals;
namespace {

bool acquire_gil() {
  // basically if this function can acquire the gil, it will return quickly.
  // if not, it will hang forever.  The idea is to call this from a thread
  // wrapped in a future, and then check the future after a timeout, to
  // determine whether we're facing gil contention.
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    return true;
  }

  // If we end up here, its probably still a "pass" from the perspective of
  // checking whether python is stuck. but currently we don't check the return
  // value of this function anyway, just check whether it returned quickly vs
  // timing out.  Taking a long time is the main sign of trouble.  Fast return
  // with true or with false is both OK from the perspective of debugging python
  // hangs.
  return false;
}

bool registerGilChecker() {
  torch_mlu::get_gil_checker() = &acquire_gil;
  return true;
}

static bool registered = registerGilChecker();

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
            .def(
                "_shutdown",
                [](const c10::intrusive_ptr<ProcessGroupCNCL>& self) {
                  return self->shutdown();
                },
                py::call_guard<py::gil_scoped_release>())
            .def("get_cncl_comm", &ProcessGroupCNCL::getCnclComm)
            .def_property_readonly("options", &ProcessGroupCNCL::getOptions)
            .def_property_readonly("uid", &ProcessGroupCNCL::getUid);

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
            &ProcessGroupCNCL::Options::is_high_priority_stream)
        .def_readwrite("group_name", &ProcessGroupCNCL::Options::group_name)
        .def_readwrite(
            "global_ranks_in_group",
            &ProcessGroupCNCL::Options::global_ranks_in_group);
  }

  dist.def("_dump_cncl_trace", []() {
    return py::bytes(torch_mlu::dump_cncl_trace());
  });

  auto mluc_m = py::handle(module).cast<py::module>();
  mluc_m.def("_cncl_version", &get_cncl_version);
}

} // namespace

__attribute__((visibility("default"))) void THMPProcessGroupCNCL_init(
    PyObject* module) {
  init_internal(module);
}

} // namespace torch_mlu
