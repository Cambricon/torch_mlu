import functools
import hashlib
import json
import textwrap
from typing import Any, Dict

import torch

from torch._inductor.codecache import CacheBase


@staticmethod
@functools.lru_cache(None)
def get_system():
    try:
        import triton

        triton_version = triton.__version__
    except ModuleNotFoundError:
        triton_version = None

    try:
        system: Dict[str, Any] = {
            "device": {
                "name": torch.mlu.get_device_properties(
                    torch.mlu.current_device()
                ).name,
            },
            "version": {
                "mlu": torch.version.mlu,
                "triton": triton_version,
            },
        }
    except (AssertionError, RuntimeError):
        # If mlu is not installed, none of the above config is relevant.
        system = {}

    system["hash"] = hashlib.sha256(
        json.dumps(system, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return system


torch._inductor.codecache.CacheBase.get_system = get_system


def CacheBase__init__(self):
    if not torch.mlu.is_available():
        return

    self.system = get_system()

    self.local_cache_path = CacheBase.get_local_cache_path()
    self.global_cache_path = CacheBase.get_global_cache_path()


torch._inductor.codecache.CacheBase.__init__ = CacheBase__init__


suffix_template = textwrap.dedent(
    """
    // Python bindings to call %s():
    #define PY_SSIZE_T_CLEAN
    #include <Python.h>
    #include <sstream>
    #include <cstdlib>

    #ifndef _MSC_VER
    #if __cplusplus < 202002L
    // C++20 (earlier) code
    // https://en.cppreference.com/w/cpp/language/attributes/likely
    #define likely(x)       __builtin_expect(!!(x), 1)
    #define unlikely(x)     __builtin_expect(!!(x), 0)
    #endif
    #else
    #define likely(x) (x)
    #define unlikely(x) (x)
    #endif

    // This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
    // We manually link it below to workaround issues with fbcode build.
    static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

    template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
        static_assert(std::is_pointer<T>::value, "arg type must be pointer or long");
        return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
    }
    template <> inline long parse_arg<long>(PyObject* args, size_t n) {
        auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
        if(unlikely(result == -1 && PyErr_Occurred()))
            throw std::runtime_error("expected int arg");
        return result;
    }

    %s

    static PyObject* %s_py(PyObject* self, PyObject* args) {
        try {
            if(unlikely(!PyTuple_CheckExact(args)))
                throw std::runtime_error("tuple args required");
            if(unlikely(PyTuple_GET_SIZE(args) != %s))
                throw std::runtime_error("requires %s args");
            %s
        } catch(std::exception const& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return nullptr;
        } catch(...) {
            PyErr_SetString(PyExc_RuntimeError, "unhandled error");
            return nullptr;
        }
    }

    static PyMethodDef py_methods[] = {
        {"%s", %s_py, METH_VARARGS, ""},
        {NULL, NULL, 0, NULL}};

    static struct PyModuleDef py_module =
        {PyModuleDef_HEAD_INIT, "%s", NULL, -1, py_methods};

    PyMODINIT_FUNC PyInit_%s(void) {
        const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
        if(!str_addr) {
            PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
            return nullptr;
        }
        std::istringstream iss(str_addr);
        uintptr_t addr = 0;
        iss >> addr;
        _torchinductor_pyobject_tensor_data_ptr =
            reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
        return PyModule_Create(&py_module);
    }
    """
)

torch._inductor.codecache.CppPythonBindingsCodeCache.suffix_template = suffix_template
