#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <structmember.h>
#include "python/Event.h"

PyObject* THMPEventClass = nullptr;

static PyObject* THMPEvent_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  unsigned char enable_timing = 0;
  unsigned char blocking = 0;
  unsigned char interprocess = 0;

  static char* kwlist[] = {
      "enable_timing", "blocking", "interprocess", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|bbb",
          const_cast<char**>(kwlist),
          &enable_timing,
          &blocking,
          &interprocess)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THMPEvent* self = (THMPEvent*)ptr.get();
  unsigned int flags =
      (blocking ? CNRT_NOTIFIER_SYNC_WAIT : CNRT_NOTIFIER_DEFAULT) |
      (enable_timing ? CNRT_NOTIFIER_DEFAULT
                     : CNRT_NOTIFIER_DISABLE_TIMING_ALL) |
      (interprocess
           ? CNRT_NOTIFIER_INTERPROCESS | CNRT_NOTIFIER_DISABLE_TIMING_ALL
           : CNRT_NOTIFIER_DEFAULT);

  new (&self->mlu_event) torch_mlu::MLUEvent(flags);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_from_ipc_handle(
    PyObject* _type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto type = (PyTypeObject*)_type;

  static torch::PythonArgParser parser({
      "from_ipc_handle(Device device, std::string ipc_handle)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Device device = r.device(0);
  std::string handle_string = r.string(1);

  TORCH_CHECK(
      handle_string.size() == sizeof(cnrtIpcNotifierHandle),
      "cnrtIpcNotifierHandle expects byte-like object of size ",
      sizeof(cnrtIpcNotifierHandle),
      ", but got ",
      handle_string.size());
  TORCH_CHECK(
      device.type() == at::kPrivateUse1,
      "MLUEvent can only be created on "
      "MLU devices, but got device type ",
      device.type())

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THMPEvent* self = (THMPEvent*)ptr.get();

  cnrtIpcNotifierHandle handle;
  std::memcpy(&handle, handle_string.c_str(), handle_string.size());
  new (&self->mlu_event) torch_mlu::MLUEvent(device.index(), &handle);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THMPEvent_dealloc(THMPEvent* self) {
  self->mlu_event.~MLUEvent();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THMPEvent_get_mlu_event(THMPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->mlu_event.event());
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_get_device(THMPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  at::optional<at::Device> device = self->mlu_event.device();
  if (!device) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_record(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS
  auto self = (THMPEvent*)_self;
  auto stream = (THMPStream*)_stream;
  self->mlu_event.place(stream->mlu_stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_wait(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS {
    auto self = (THMPEvent*)_self;
    auto stream = (THMPStream*)_stream;
    pybind11::gil_scoped_release no_gil;
    self->mlu_event.wait(stream->mlu_stream);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THMPEvent*)_self;
  return PyBool_FromLong(self->mlu_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_elapsed_time(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto self = (THMPEvent*)_self;
  auto other = (THMPEvent*)_other;
  return PyFloat_FromDouble(self->mlu_event.elapsed_time(other->mlu_event));
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_hardware_time(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto self = (THMPEvent*)_self;
  auto other = (THMPEvent*)_other;
  return PyFloat_FromDouble(self->mlu_event.hardware_time(other->mlu_event));
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    auto self = (THMPEvent*)_self;
    pybind11::gil_scoped_release no_gil;
    self->mlu_event.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_ipc_handle(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THMPEvent*)_self;

  cnrtIpcNotifierHandle handle;
  self->mlu_event.ipc_handle(&handle);
  return PyBytes_FromStringAndSize((const char*)&handle, sizeof(handle));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THMPEvent_properties[] = {
    {"device", (getter)THMPEvent_get_device, nullptr, nullptr, nullptr},
    {"mlu_event", (getter)THMPEvent_get_mlu_event, nullptr, nullptr, nullptr},
    {nullptr}};

static PyMethodDef THMPEvent_methods[] = {
    {(char*)"from_ipc_handle",
     castPyCFunctionWithKeywords(THMPEvent_from_ipc_handle),
     METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {(char*)"record", THMPEvent_record, METH_O, nullptr},
    {(char*)"wait", THMPEvent_wait, METH_O, nullptr},
    {(char*)"query", THMPEvent_query, METH_NOARGS, nullptr},
    {(char*)"elapsed_time", THMPEvent_elapsed_time, METH_O, nullptr},
    {(char*)"hardware_time", THMPEvent_hardware_time, METH_O, nullptr},
    {(char*)"synchronize", THMPEvent_synchronize, METH_NOARGS, nullptr},
    {(char*)"ipc_handle", THMPEvent_ipc_handle, METH_NOARGS, nullptr},
    {nullptr}};

PyTypeObject THMPEventType = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0) "torch_mlu._MLUC._MLUEventBase", /* tp_name */
    sizeof(THMPEvent), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THMPEvent_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    0, /* tp_getattr */
    0, /* tp_setattr */
    0, /* tp_reserved */
    0, /* tp_repr */
    0, /* tp_as_number */
    0, /* tp_as_sequence */
    0, /* tp_as_mapping */
    0, /* tp_hash  */
    0, /* tp_call */
    0, /* tp_str */
    0, /* tp_getattro */
    0, /* tp_setattro */
    0, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    THMPEvent_methods, /* tp_methods */
    0, /* tp_members */
    THMPEvent_properties, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    0, /* tp_init */
    0, /* tp_alloc */
    THMPEvent_pynew, /* tp_new */
};

void THMPEvent_init(PyObject* module) {
  THMPEventClass = (PyObject*)&THMPEventType;
  if (PyType_Ready(&THMPEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THMPEventType);
  if (PyModule_AddObject(module, "_MLUEventBase", (PyObject*)&THMPEventType) <
      0) {
    throw python_error();
  }
}
