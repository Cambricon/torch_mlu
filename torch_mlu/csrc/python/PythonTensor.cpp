#include <torch/csrc/tensor/python_tensor.h>

#include <pybind11/pybind11.h>
#include <structmember.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_types.h>

#include <ATen/ATen.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "framework/core/device.h"

// Copied from pytorch/torch/csrc/tensor/python_tensor.cpp,
// and replace cuda with mlu(PrivateUse1) because we want to
// use some methods like torch.mlu.FloatTensor.

namespace torch_mlu {
namespace tensors {

using namespace at;
using namespace torch::autograd;

std::vector<std::pair<Backend, ScalarType>> all_declared_mlu_types() {
  std::vector<std::pair<Backend, ScalarType>> ret;
  std::vector<Backend> backends = {Backend::PrivateUse1};
  std::vector<ScalarType> scalar_types = {
      ScalarType::Byte,
      ScalarType::Char,
      ScalarType::Double,
      ScalarType::Float,
      ScalarType::Int,
      ScalarType::Long,
      ScalarType::Short,
      ScalarType::Half,
      ScalarType::Bool,
      ScalarType::BFloat16};

  for (auto& backend : backends) {
    for (auto& scalar_type : scalar_types) {
      ret.emplace_back(backend, scalar_type);
    }
  }

  return ret;
}

struct PyTensorType {
  PyTypeObject py_type;
  THPDtype* dtype;
  THPLayout* layout;
  bool is_mlu;
  char name[64];
  int backend;
  int scalar_type;

  Backend get_backend() const {
    return static_cast<Backend>(backend);
  }

  DispatchKey get_dispatch_key() const {
    return backendToDispatchKey(static_cast<Backend>(backend));
  }

  ScalarType get_scalar_type() const {
    return static_cast<ScalarType>(scalar_type);
  }
};

static_assert(
    std::is_standard_layout<PyTensorType>::value,
    "PyTensorType must be standard layout");

// static Backend default_backend = Backend::CPU;

static void py_bind_tensor_types(
    const std::vector<PyTensorType*>& tensor_types);

static PyObject* Tensor_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto& tensor_type = *((PyTensorType*)type);
  TORCH_CHECK_TYPE(
      !tensor_type.is_mlu || torch_mlu::device_count() > 0,
      "type ",
      tensor_type.name,
      " not available. Torch not compiled with MLU enabled.")
  if (tensor_type.is_mlu) {
    TORCH_WARN_ONCE(
        "The torch.mlu.*DtypeTensor constructors are no longer recommended. "
        "It's best to use methods such as torch.tensor(data, dtype=*, device='mlu') to create tensors.")
  }
  return THPVariable_Wrap(torch::utils::legacy_tensor_ctor(
      tensor_type.get_dispatch_key(),
      tensor_type.get_scalar_type(),
      args,
      kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject* Tensor_instancecheck(PyObject* _self, PyObject* arg) {
  HANDLE_TH_ERRORS
  auto self = (PyTensorType*)_self;
  if (THPVariable_Check(arg)) {
    const auto& var = THPVariable_Unpack(arg);
    if (legacyExtractDispatchKey(var.key_set()) == self->get_dispatch_key() &&
        var.scalar_type() == static_cast<ScalarType>(self->scalar_type)) {
      Py_RETURN_TRUE;
    }
  }
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

static PyObject* Tensor_dtype(PyTensorType* self, void* unused) {
  return torch::autograd::utils::wrap(self->dtype);
}

static PyObject* Tensor_layout(PyTensorType* self, void* unused) {
  return torch::autograd::utils::wrap(self->layout);
}

static PyObject* Tensor_is_mlu(PyTensorType* self, void* unused) {
  if (self->is_mlu) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyObject* Tensor_is_sparse(PyTensorType* self, void* unused) {
  if (self->layout->layout == at::Layout::Strided) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
}

static PyObject* Tensor_is_sparse_csr(PyTensorType* self, void* unused) {
  if (self->layout->layout == at::Layout::SparseCsr) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static struct PyMethodDef metaclass_methods[] = {
    {"__instancecheck__", Tensor_instancecheck, METH_O, nullptr},
    {nullptr}};

typedef PyObject* (*getter)(PyObject*, void*);

static struct PyGetSetDef metaclass_properties[] = {
    {"dtype", (getter)Tensor_dtype, nullptr, nullptr, nullptr},
    {"layout", (getter)Tensor_layout, nullptr, nullptr, nullptr},
    {"is_mlu", (getter)Tensor_is_mlu, nullptr, nullptr, nullptr},
    {"is_sparse", (getter)Tensor_is_sparse, nullptr, nullptr, nullptr},
    {"is_sparse_csr", (getter)Tensor_is_sparse_csr, nullptr, nullptr, nullptr},
    {nullptr}};

static PyTypeObject metaclass = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.tensortype", /* tp_name */
    sizeof(PyTypeObject) /* tp_basicsize */
};

static void py_initialize_metaclass(PyTypeObject& metaclass) {
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_methods = metaclass_methods;
  metaclass.tp_getset = metaclass_properties;
  metaclass.tp_base = &PyType_Type;
  if (PyType_Ready(&metaclass) < 0) {
    throw python_error();
  }
}

static PyTypeObject tensor_type_prototype = {
    PyVarObject_HEAD_INIT(&metaclass, 0) nullptr, /* tp_name */
    sizeof(PyTensorType) /* tp_basicsize */
};

static void py_initialize_tensor_type(
    PyTypeObject& type,
    const char* name,
    PyObject* tp_dict) {
  // NOTE: we don't use the typical static declaration of PyTypeObject because
  // we need to initialize as many types as there are VariableType instances.
  // We copy the basic object fields from a prototype definition and initialize
  // the remaining fields below.
  memcpy(&type, &tensor_type_prototype, sizeof(PyTypeObject));
  // Subclassing from torch.<ScalarType>Tensor isn't supported.
  // (Py_TPFLAGS_BASETYPE omitted). Subclassing torch.Tensor still allowed.
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  type.tp_name = name;
  type.tp_new = Tensor_new;
  if (PyType_Ready(&type) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(type.tp_dict, tp_dict, 0) < 0) {
    throw python_error();
  }
}

static const std::string get_module(Backend backend) {
  switch (backend) {
    case Backend::CPU:
      return "torch";
    case Backend::CUDA:
      return "torch.cuda";
    case Backend::SparseCPU:
      return "torch.sparse";
    case Backend::SparseCUDA:
      return "torch.cuda.sparse";
    case Backend::PrivateUse1:
      return "torch." + c10::get_privateuse1_backend();
    default:
      AT_ERROR("invalid backend: ", toString(backend));
  }
}

static std::string get_name(Backend backend, ScalarType scalarType) {
  std::ostringstream ss;
  ss << get_module(backend) << "." << toString(scalarType) << "Tensor";
  return ss.str();
}

static void set_type(
    PyTensorType& type_obj,
    Backend backend,
    ScalarType scalarType) {
  // This field is lazily initialized from backend and scalar_type
  type_obj.backend = static_cast<int>(backend);
  type_obj.scalar_type = static_cast<int>(scalarType);
  type_obj.layout = torch::getTHPLayout(layout_from_backend(backend));
  type_obj.dtype = torch::getTHPDtype(scalarType);
  type_obj.is_mlu = (backend == at::Backend::PrivateUse1);
}

static void set_name(PyTensorType& type_obj, const std::string& name) {
  size_t n = sizeof(type_obj.name);
  strncpy(type_obj.name, name.c_str(), n);
  type_obj.name[n - 1] = '\0';
}

static THPObjectPtr get_tensor_dict() {
  auto torch = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch)
    throw python_error();

  auto tensor_class = THPObjectPtr(PyObject_GetAttrString(torch, "Tensor"));
  if (!tensor_class)
    throw python_error();

  auto tensor_type = (PyTypeObject*)tensor_class.get();
  TORCH_CHECK(tensor_type->tp_base, "missing base type for Tensor");

  auto res = THPObjectPtr(PyDict_New());
  if (!res)
    throw python_error();

  if (PyDict_Merge(res.get(), tensor_type->tp_dict, 0) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(res.get(), tensor_type->tp_base->tp_dict, 0) < 0) {
    throw python_error();
  }

  return res;
}

// A note about the lifetime of the various PyTensorType: normally
// PyTypeObject instances are statically allocated, but we want to create them
// dynamically at init time, because their exact number depends on
// torch::utils::all_declared_types(). The memory for each PyTensorType is
// allocated by initialize_aten_types() and never freed: technically it's a
// leak, but it's not a problem since we want them to be alive for the whole
// time of the process anyway.
//
// An alternative is to use a std::vector<PyTensorType> instead, and let
// std::vector to manage the lifetime of its items. This is problematic
// though, because it means that the memory of PyTensorType is deallocated at
// some point during the exit: if by chance we have another global destructor
// and/or atexit() function which tries to access the PyTensorTypes, we risk
// an use-after-free error. This happens for example if we embed CPython and
// call Py_Finalize inside an atexit() function which was registered before
// importing torch.
static std::vector<PyTensorType*> tensor_types;

static void initialize_aten_types(std::vector<PyTensorType*>& tensor_types) {
  // add mlu supported types
  auto declared_types = all_declared_mlu_types();
  tensor_types.resize(declared_types.size());

  for (size_t i = 0, end = declared_types.size(); i != end; i++) {
    tensor_types[i] = new PyTensorType();
    auto& tensor_type = *tensor_types[i];
    Backend backend = declared_types[i].first;
    ScalarType scalar_type = declared_types[i].second;
    set_type(tensor_type, backend, scalar_type);
    set_name(tensor_type, get_name(backend, scalar_type));
  }
}

void initialize_python_bindings() {
  // Initialize the at::Type* pointers, name, and properties of the PyTensorType
  // vector. After this call, the vector must not be resized.
  initialize_aten_types(tensor_types);

  // Initialize the Python metaclass for the torch.FloatTensor, etc. types.
  // The metaclass handles __instancecheck__ checks and binds the dtype property
  // on the type objects.
  py_initialize_metaclass(metaclass);

  // Get the tp_dict of the Variable class. We copy function definitions
  // onto each Tensor type object so that they can be accessed via e.g.
  // `torch.FloatTensor.add`.
  auto tensor_dict = get_tensor_dict();

  // Initialize each Python type object torch.FloatTensor, torch.DoubleTensor,
  // etc.
  for (auto& tensor_type : tensor_types) {
    py_initialize_tensor_type(
        tensor_type->py_type, tensor_type->name, tensor_dict.get());
  }

  // Add the type objects to their corresponding modules. e.g. torch.FloatTensor
  // is added to the `torch` module as `FloatTensor`. Also add all the type
  // objects to the set torch._tensor_classes.
  py_bind_tensor_types(tensor_types);
}

static void py_bind_tensor_types(
    const std::vector<PyTensorType*>& tensor_types) {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module)
    throw python_error();

  auto tensor_classes = THPObjectPtr(
      PyObject_GetAttrString(torch_module.get(), "_tensor_classes"));
  if (!tensor_classes)
    throw python_error();

  for (auto& tensor_type : tensor_types) {
    auto name = std::string(tensor_type->name);
    auto idx = name.rfind('.');
    auto type_name = name.substr(idx + 1);
    auto module_name = name.substr(0, idx);

    auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module_obj)
      throw python_error();

    PyObject* type_obj = (PyObject*)tensor_type;
    Py_INCREF(type_obj);
    if (PyModule_AddObject(module_obj.get(), type_name.c_str(), type_obj) < 0) {
      throw python_error();
    }
    if (PySet_Add(tensor_classes.get(), type_obj) < 0) {
      throw python_error();
    }
  }
}

} // namespace tensors
} // namespace torch_mlu
