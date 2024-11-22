#include "python/distributed/init.h"
#include <torch/csrc/distributed/c10d/python_comm_hook.h>

namespace torch_mlu {

// NOTE: The following two functions are directly copied from PyTorch. The
// reason for needing to monkey patch these two functions is that the functions
// in the PyTorch repository only accept the Reducer class, rather than the
// Reducer_mlu class, as parameters.

// Called from DDP's Python API to create a
// c10d Python comm hook object. The input state and callable comm_hook are
// Python objects. It later calls register_comm_hook function of the reducer
// input to register the hook.
void _register_comm_hook(
    torch_mlu::Reducer_mlu& reducer,
    py::object state,
    py::object comm_hook) {
  reducer.register_comm_hook(std::make_unique<::c10d::PythonCommHook>(
      std::move(state), std::move(comm_hook)));
}

// Called from DDP's Python API to create a c10d C++ comm hook.
// The input is an enum hook type. It later calls register_builtin_comm_hook
// function of the reducer input to set the hook type.
void _register_builtin_comm_hook(
    torch_mlu::Reducer_mlu& reducer,
    ::c10d::BuiltinCommHookType comm_hook_type) {
  reducer.register_builtin_comm_hook(comm_hook_type);
}

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void initC10dMlu(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto c10d_mlu = m.def_submodule("_c10d_mlu", "c10d_mlu bindings");
  shared_ptr_class_<torch_mlu::Reducer_mlu>(c10d_mlu, "Reducer")
      .def(
          py::init<
              std::vector<at::Tensor>,
              std::vector<std::vector<size_t>>,
              std::vector<size_t>,
              c10::intrusive_ptr<::c10d::ProcessGroup>,
              std::vector<bool>,
              int64_t,
              bool,
              bool,
              std::unordered_map<size_t, std::string>,
              int64_t>(),
          py::arg("params"),
          py::arg("bucket_indices"),
          py::arg("per_bucket_size_limits"),
          py::arg("process_group"),
          py::arg("expect_sparse_gradients") = std::vector<bool>(),
          py::arg("bucket_bytes_cap") = ::c10d::kDefaultBucketBytesCap,
          py::arg("find_unused_parameters") = false,
          py::arg("gradient_as_bucket_view") = false,
          py::arg("param_to_name_mapping") =
              std::unordered_map<size_t, std::string>(),
          py::arg("first_bucket_bytes_cap") = ::c10d::kDefaultFirstBucketBytes,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "prepare_for_forward",
          &torch_mlu::Reducer_mlu::prepare_for_forward,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "prepare_for_backward",
          &torch_mlu::Reducer_mlu::prepare_for_backward,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "prepare_for_backward",
          [](torch_mlu::Reducer_mlu& reducer, const at::Tensor& output)
              -> void { reducer.prepare_for_backward({output}); },
          py::call_guard<py::gil_scoped_release>())
      .def("get_backward_stats", &torch_mlu::Reducer_mlu::get_backward_stats)
      .def(
          "_install_post_backward_futures",
          [](torch_mlu::Reducer_mlu& reducer,
             const std::vector<
                 std::shared_ptr<torch::jit::PythonFutureWrapper>>& futs) {
            c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures(
                c10::FutureType::create(c10::TensorType::get()));
            for (const auto& fut : futs) {
              futures.push_back(fut->fut);
            }
            reducer.install_futures(std::move(futures));
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_rebuild_buckets",
          &torch_mlu::Reducer_mlu::rebuild_buckets,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_get_zeros_like_grad_buckets",
          [](torch_mlu::Reducer_mlu& reducer) {
            return reducer.get_grad_buckets(/* return_zero_tensors */ true);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_optimizer_in_backward",
          [](torch_mlu::Reducer_mlu& reducer) {
            reducer.set_optimizer_in_backward();
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_sparse_metadata",
          &torch_mlu::Reducer_mlu::setSparseMetadata,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_mixed_precision_param_dtype",
          [](torch_mlu::Reducer_mlu& reducer, py::object data_type_obj) {
            auto scalar_type =
                reinterpret_cast<THPDtype*>(data_type_obj.ptr())->scalar_type;
            reducer.set_mixed_precision_param_dtype(scalar_type);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_push_all_rebuilt_params",
          &torch_mlu::Reducer_mlu::push_rebuilt_params_for_all_indices,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_forward_pass_work_handle",
          &torch_mlu::Reducer_mlu::set_forward_pass_work_handle,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_get_local_used_map",
          &torch_mlu::Reducer_mlu::get_local_used_map_on_device)
      .def(
          "_set_ddp_runtime_logging_sample_rate",
          &torch_mlu::Reducer_mlu::set_ddp_runtime_logging_sample_rate,
          py::arg("sample_rate"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_static_graph",
          &torch_mlu::Reducer_mlu::set_static_graph,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_ddp_graph_static",
          &torch_mlu::Reducer_mlu::ddp_graph_static,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_delay_all_reduce",
          &torch_mlu::Reducer_mlu::delay_all_reduce,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_run_comm_hook",
          [](torch_mlu::Reducer_mlu& reducer, ::c10d::GradBucket& bucket)
              -> std::shared_ptr<torch::jit::PythonFutureWrapper> {
            c10::intrusive_ptr<c10::ivalue::Future> fut =
                reducer.run_comm_hook(bucket);
            return std::make_shared<torch::jit::PythonFutureWrapper>(fut);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_run_allreduce_hook",
          [](torch_mlu::Reducer_mlu& reducer, ::c10d::GradBucket& bucket)
              -> std::shared_ptr<torch::jit::PythonFutureWrapper> {
            c10::intrusive_ptr<c10::ivalue::Future> fut =
                reducer.run_allreduce_hook(bucket);
            return std::make_shared<torch::jit::PythonFutureWrapper>(fut);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_autograd_hook",
          [](torch_mlu::Reducer_mlu& reducer, int index) -> void {
            reducer.autograd_hook(index);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_logger",
          [](torch_mlu::Reducer_mlu& reducer,
             const std::shared_ptr<torch_mlu::Logger_mlu> logger) {
            std::weak_ptr<torch_mlu::Logger_mlu> logger_weakref = logger;
            reducer.set_logger(logger_weakref);
          })
      .def(
          "_remove_autograd_hooks",
          [](torch_mlu::Reducer_mlu& reducer) {
            reducer.remove_autograd_hooks();
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_check_reducer_finalized",
          [](torch_mlu::Reducer_mlu& reducer) {
            return reducer.check_finalized();
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_reset_state",
          [](torch_mlu::Reducer_mlu& reducer) { return reducer.reset_state(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_update_process_group",
          [](torch_mlu::Reducer_mlu& reducer,
             c10::intrusive_ptr<::c10d::ProcessGroup> new_process_group) {
            return reducer.update_process_group(new_process_group);
          },
          py::call_guard<py::gil_scoped_release>());
  c10d_mlu
      .def(
          "_register_comm_hook",
          &_register_comm_hook,
          py::arg("reducer"),
          py::arg("state"),
          py::arg("comm_hook"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_register_builtin_comm_hook",
          &_register_builtin_comm_hook,
          py::arg("reducer"),
          py::arg("comm_hook_type"));
  shared_ptr_class_<torch_mlu::Logger_mlu>(c10d_mlu, "Logger")
      .def(
          py::init<std::shared_ptr<torch_mlu::Reducer_mlu>>(),
          py::arg("reducer"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_construction_data_and_log",
          &torch_mlu::Logger_mlu::set_construction_data_and_log,
          py::arg("module_name"),
          py::arg("device_ids"),
          py::arg("output_device"),
          py::arg("broadcast_buffers"),
          py::arg("has_sync_bn"),
          py::arg("static_graph"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_runtime_stats_and_log",
          &torch_mlu::Logger_mlu::set_runtime_stats_and_log,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_error_and_log",
          [](torch_mlu::Logger_mlu& logger, const std::string& error) {
            logger.set_error_and_log(error);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_get_ddp_logging_data",
          &torch_mlu::Logger_mlu::get_ddp_logging_data,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_comm_hook_name",
          &torch_mlu::Logger_mlu::set_comm_hook,
          py::arg("comm_hook"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_uneven_input_join",
          &torch_mlu::Logger_mlu::set_uneven_input_join,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_static_graph",
          &torch_mlu::Logger_mlu::set_static_graph,
          py::call_guard<py::gil_scoped_release>());
}

} // namespace torch_mlu
