#include <ATen/ATen.h>
#include <ATen/MapAllocator.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/python_headers.h>
#include <structmember.h>
#include <c10/core/CPUAllocator.h>
#include <libshm.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include "python/THMP.h"
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/copy_utils.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/native/Resize.h>

#include "aten/utils/tensor_util.h"
#include "python/Storage.h"
#include "python/StorageMethods.h"
#include "framework/core/caching_allocator.h"

#define LSEEK lseek

void resize_bytes_mlu(c10::StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(
      storage->resizable(), "Trying to resize storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(
      allocator != nullptr, "Trying to resize storage without an allocator");
  int device;
  TORCH_CNRT_CHECK(cnrtGetDevice(&device));
  if (size_bytes == 0) {
    storage->set_data_ptr(
        at::DataPtr(nullptr, at::Device(at::DeviceType::PrivateUse1, device)));
    storage->set_nbytes(0);
  } else {
    at::DataPtr data = allocator->allocate(size_bytes);
    if (storage->data_ptr()) {
      // TODO(mengpenghui): currently, do not support memcpy across devices
      // Enable p2p access when the memcpy is across devices
      // get_p2p_access(state, device, THMStorage_getDevice(state, self));
      auto stream = torch_mlu::getCurrentMLUStream();
      TORCH_CNRT_CHECK(cnrtMemcpyAsync_V2(
          data.get(),
          const_cast<void*>(storage->data()),
          std::min(storage->nbytes(), size_bytes),
          stream.stream(),
          cnrtMemcpyDevToDev));
    }
    storage->set_data_ptr(std::move(data));
    storage->set_nbytes(size_bytes);
  }
}

static PyObject* THMPStorage_nbytes(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return py::cast(THMPStorage_Unpack(self).sym_nbytes()).release().ptr();
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_dataPtr(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self_ = THMPStorage_Unpack(self);
  auto invalid = self_.data() == nullptr &&
      self_.device_type() != c10::DeviceType::Meta && self_.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid,
      "Attempted to access the data pointer on an invalid python storage.")
  return PyLong_FromVoidPtr(self_.mutable_data());
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_copy_(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  at::Storage self_ = torch::createStorage(self);
  static torch::PythonArgParser parser({
      "copy_(Storage src, bool? non_blocking=None)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Storage src = r.storage(0);
  bool non_blocking = r.toBoolOptional(1).value_or(false);

  auto invalid = src.data() == nullptr &&
      src.device_type() != c10::DeviceType::Meta && src.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call copy_() on an invalid python storage.")

  TORCH_CHECK(self_.nbytes() == src.nbytes(), "size does not match");
  at::storage_copy(self_, src, non_blocking);

  Py_INCREF(self);
  return self;

  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_elementSize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(sizeof(uint8_t));
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_new(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::Allocator* allocator = THMPStorage_Unpack(self).allocator();
  auto new_storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      0,
      allocator,
      /*resizable=*/true);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  return THMPStorage_New(std::move(new_storage));
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_resize_(PyObject* self, PyObject* number_arg) {
  HANDLE_TH_ERRORS
  const auto& storage = THMPStorage_Unpack(self);
  auto invalid = storage.data() == nullptr &&
      storage.device_type() != c10::DeviceType::Meta &&
      storage.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call resize_() on an invalid python storage.")
  THPUtils_assert(
      THPUtils_checkLong(number_arg),
      "resize_ expects an int, "
      "but got %s",
      THPUtils_typename(number_arg));
  int64_t newsize = THPUtils_unpackLong(number_arg);
  c10::DeviceType device_type = storage.device_type();
  if (device_type == at::kCPU) {
    at::native::resize_bytes_cpu(storage.unsafeGetStorageImpl(), newsize);
  } else if (device_type == at::kPrivateUse1) {
    ptrdiff_t size_bytes_i = newsize;
    TORCH_CHECK(
        !c10::overflows<size_t>(size_bytes_i),
        "Requested storage size (",
        size_bytes_i,
        ") cannot be represented as a size_t");
    const auto size_bytes = static_cast<size_t>(size_bytes_i);
    resize_bytes_mlu(storage.unsafeGetStorageImpl(), size_bytes);
  } else {
    TORCH_CHECK(
        false,
        "UntypedStorage.resize_: got unexpected device type ",
        device_type);
  }
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_fill_(PyObject* self, PyObject* number_arg) {
  HANDLE_TH_ERRORS
  const auto& storage = THMPStorage_Unpack(self);
  auto invalid = storage.data() == nullptr &&
      storage.device_type() != c10::DeviceType::Meta &&
      storage.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call fill_() on an invalid python storage.")
  THPUtils_assert(
      THPByteUtils_checkReal(number_arg),
      "fill_ expects int, "
      "but got %s",
      THPUtils_typename(number_arg));
  storage_fill(storage, THPByteUtils_unpackReal(number_arg));
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_fromBuffer(
    PyObject* _unused,
    PyObject* args,
    PyObject* keywds) {
  HANDLE_TH_ERRORS
  PyObject* obj = nullptr;
  const char* byte_order_str = nullptr;
  Py_ssize_t count = -1, offset = 0;
  PyObject* dtype_obj = nullptr;
  c10::ScalarType scalar_type = at::kByte;
  Py_buffer buffer = {};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,clang-diagnostic-writable-strings)
  static char* kwlist[] = {
      "buffer", "byte_order", "count", "offset", "dtype", nullptr};
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const char* argtypes;
  argtypes = "O|snnO";

  if (!PyArg_ParseTupleAndKeywords(
          args,
          keywds,
          argtypes,
          kwlist,
          &obj,
          &byte_order_str,
          &count,
          &offset,
          &dtype_obj)) {
    return nullptr;
  }
  TORCH_CHECK(dtype_obj != nullptr, "argument 'dtype' cannot be None");
  TORCH_CHECK(
      THPDtype_Check(dtype_obj),
      "argument 'dtype' must be of type torch.dtype");
  auto dtype = reinterpret_cast<THPDtype*>(dtype_obj);
  scalar_type = dtype->scalar_type;

  const bool is_endian_independent = (scalar_type == at::kByte) ||
      (scalar_type == at::kChar) || (scalar_type == at::kFloat8_e5m2) ||
      (scalar_type == at::kFloat8_e4m3fn);

  TORCH_CHECK(
      is_endian_independent || (byte_order_str != nullptr),
      "function missing required argument 'byte_order' (pos 2)");
  size_t element_size = c10::elementSize(scalar_type);

  bool do_byte_swap = false;
  torch::utils::THPByteOrder byte_order;
  if (!is_endian_independent) {
    if (strcmp(byte_order_str, "native") == 0) {
      do_byte_swap = false;
    } else if (strcmp(byte_order_str, "big") == 0) {
      do_byte_swap =
          (torch::utils::THP_LITTLE_ENDIAN ==
           torch::utils::THP_nativeByteOrder());
    } else if (strcmp(byte_order_str, "little") == 0) {
      do_byte_swap =
          (torch::utils::THP_BIG_ENDIAN == torch::utils::THP_nativeByteOrder());
    } else {
      PyErr_Format(
          PyExc_ValueError,
          "invalid byte_order '%s' (expected 'big', 'little', or 'native')",
          byte_order_str);
      return nullptr;
    }
  }

  if (PyObject_GetBuffer(obj, &buffer, PyBUF_SIMPLE) < 0)
    return nullptr;

  if (offset < 0 || offset > buffer.len) {
    std::string msg =
        "offset must be non-negative and no greater than buffer length (" +
        std::to_string(offset) + ") , but got " + std::to_string(buffer.len);
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    PyBuffer_Release(&buffer);
    return nullptr;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t size_bytes = 0;
  if (count < 0) {
    if ((buffer.len - offset) % element_size != 0) {
      std::string msg = "buffer size ( " + std::to_string(buffer.len) +
          ") must be a multiple of element size (" +
          std::to_string(element_size) + ")";
      PyErr_SetString(PyExc_ValueError, msg.c_str());
      PyBuffer_Release(&buffer);
      return nullptr;
    }
    size_bytes = buffer.len - offset;
    count = static_cast<Py_ssize_t>(size_bytes / element_size);
  } else {
    size_bytes = count * element_size;
  }
  if (offset + (count * (Py_ssize_t)element_size) > buffer.len) {
    std::string msg = "buffer has only " + std::to_string(buffer.len - offset) +
        " elements after offset " + std::to_string(offset) +
        ", but specified a size of " + std::to_string(count);
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    PyBuffer_Release(&buffer);
    return nullptr;
  }

  uint8_t* src = (uint8_t*)buffer.buf;
  auto storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      c10::GetDefaultCPUAllocator(),
      /*resizable=*/true);

  if (is_endian_independent) {
    memcpy(storage->mutable_data(), src + offset, count);
  } else if (scalar_type == at::kBool) {
    // Because of ASAN checks, that are failing whenever
    // we are trying to get a value which is not 0 or 1, we have to manually
    // convert original values to boolean ones.
    torch::utils::THP_decodeBoolBuffer(
        static_cast<bool*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kShort) {
    torch::utils::THP_decodeInt16Buffer(
        static_cast<int16_t*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kInt) {
    torch::utils::THP_decodeInt32Buffer(
        static_cast<int32_t*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kLong) {
    torch::utils::THP_decodeInt64Buffer(
        static_cast<int64_t*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kHalf) {
    torch::utils::THP_decodeHalfBuffer(
        static_cast<c10::Half*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kBFloat16) {
    torch::utils::THP_decodeBFloat16Buffer(
        static_cast<c10::BFloat16*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kFloat) {
    torch::utils::THP_decodeFloatBuffer(
        static_cast<float*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kDouble) {
    torch::utils::THP_decodeDoubleBuffer(
        static_cast<double*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kComplexFloat) {
    torch::utils::THP_decodeComplexFloatBuffer(
        static_cast<c10::complex<float>*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kComplexDouble) {
    torch::utils::THP_decodeComplexDoubleBuffer(
        static_cast<c10::complex<double>*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else {
    TORCH_CHECK(false, "Unknown type: ", scalar_type);
  }
  PyBuffer_Release(&buffer);
  return (PyObject*)THMPStorage_New(storage);
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_fromFile(
    PyObject* _unused,
    PyObject* args,
    PyObject* keywds) {
  HANDLE_TH_ERRORS
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const char* filename;
  Py_ssize_t nbytes = 0;
  int shared = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,clang-diagnostic-writable-strings)
  static char* kwlist[] = {"filename", "shared", "nbytes", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args, keywds, "s|in", kwlist, &filename, &shared, &nbytes)) {
    return nullptr;
  }
  if (shared)
    shared = at::ALLOCATOR_MAPPED_SHARED;

  size_t actual_nbytes = -1;
  auto storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,
      at::MapAllocator::makeDataPtr(filename, shared, nbytes, &actual_nbytes),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  if (nbytes <= 0) {
    storage->set_nbytes(actual_nbytes);
  }

  return (PyObject*)THMPStorage_New(std::move(storage));
  END_HANDLE_TH_ERRORS
}

PyObject* THMPStorage_writeFile(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  const auto& storage = THMPStorage_Unpack(self);
  // See Note [Invalid Python Storages]
  auto invalid = storage.data() == nullptr &&
      storage.device_type() != c10::DeviceType::Meta &&
      storage.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call _write_file() on an invalid python storage.")
  PyObject* file = PyTuple_GetItem(args, 0);
  bool is_real_file = PyTuple_GetItem(args, 1) == Py_True;
  bool save_size = PyTuple_GetItem(args, 2) == Py_True;
  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 3);

  THPUtils_assert(
      element_size_obj != Py_None, "_write_file: need to specify element size");
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  if (!is_real_file) {
    if (storage.device_type() == at::kPrivateUse1) {
      THMPStorage_writeFileRaw<PyObject*>(
          storage.unsafeGetStorageImpl(), file, save_size, element_size);
    } else {
      THPStorage_writeFileRaw<PyObject*>(
          storage.unsafeGetStorageImpl(), file, save_size, element_size);
    }
    Py_RETURN_NONE;
  }

  int fd = PyObject_AsFileDescriptor(file);
  THPUtils_assert(
      fd != -1,
      "_write_file couldn't retrieve a file descriptor "
      "from given object");
  if (storage.device_type() == at::kPrivateUse1) {
    THMPStorage_writeFileRaw(
        storage.unsafeGetStorageImpl(), fd, save_size, element_size);
  } else {
    THPStorage_writeFileRaw(
        storage.unsafeGetStorageImpl(), fd, save_size, element_size);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPStorage_newWithFile(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyTuple_Size(args) == 2, "_new_with_file takes exactly two arguments");
  int fd = PyObject_AsFileDescriptor(PyTuple_GetItem(args, 0));
  THPUtils_assert(
      fd != -1,
      "_new_with_file couldn't retrieve a file "
      "descriptor from given object");
  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 1);
  THPUtils_assert(
      element_size_obj != Py_None,
      "_new_with_file: need to specify element size");
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);
  auto storage = THMPStorage_readFileRaw<int>(fd, {}, element_size);
  if (!storage.defined())
    return nullptr;
  return THMPStorage_New(std::move(storage));
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorage_setFromFile(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  const auto& storage = THMPStorage_Unpack(self);
  PyObject* file = PyTuple_GET_ITEM(args, 0);
  PyObject* offset = PyTuple_GET_ITEM(args, 1);
  bool is_real_file = PyTuple_GET_ITEM(args, 2) == Py_True;

  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 3);

  THPUtils_assert(
      element_size_obj != Py_None,
      "_set_from_file: need to specify element size");
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  if (!is_real_file) {
    // offset can be implemented with a call to the Python object's seek()
    // but it is currently unnecessary to support this.
    THPUtils_assert(
        offset == Py_None,
        "_set_from_file: offset is NYI for filelike objects");

    auto self_storage_impl = c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(
        storage.unsafeGetStorageImpl());
    auto storage_impl = storage.device_type() == at::kPrivateUse1
        ? THMPStorage_readFileRaw<PyObject*>(
              file, std::move(self_storage_impl), element_size)
        : THPStorage_readFileRaw<PyObject*>(
              file, std::move(self_storage_impl), element_size);
    if (!storage_impl.defined()) {
      return nullptr;
    }
    Py_INCREF(self);
    return (PyObject*)self;
  }

  // file is backed by a fd
  const int fd = PyObject_AsFileDescriptor(file);
  const auto fd_original_pos = LSEEK(fd, 0, SEEK_CUR);
  if (offset != Py_None) {
    LSEEK(fd, THPUtils_unpackLong(offset), SEEK_SET);
  }
  THPUtils_assert(
      fd != -1,
      "_set_from_file couldn't retrieve a file "
      "descriptor from given object");
  auto self_storage_impl = c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(
      storage.unsafeGetStorageImpl());
  auto storage_impl = storage.device_type() == at::kPrivateUse1
      ? THMPStorage_readFileRaw<int>(fd, self_storage_impl, element_size)
      : THPStorage_readFileRaw<int>(fd, self_storage_impl, element_size);
  if (!storage_impl.defined())
    return nullptr;
  Py_INCREF(self);

  // the file descriptor is returned to original position and
  // the file handle at python call-site needs updating to the
  // advanced position
  const auto fd_current_pos = LSEEK(fd, 0, SEEK_CUR);
  LSEEK(fd, fd_original_pos, SEEK_SET);
  const auto seek_return =
      PyObject_CallMethod(file, "seek", "Li", (long long)fd_current_pos, 0);
  if (seek_return == nullptr) {
    return nullptr;
  }
  Py_DECREF(seek_return);

  return self;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPStorage__setCdata(PyObject* _self, PyObject* new_cdata) {
  HANDLE_TH_ERRORS
  auto self = (THMPStorage*)_self;
  THPUtils_assert(
      THPUtils_checkLong(new_cdata),
      "given an invalid argument to "
      "_set_cdata - expected an int or long, but got %s",
      THPUtils_typename(new_cdata));
  c10::StorageImpl* ptr = (c10::StorageImpl*)PyLong_AsVoidPtr(new_cdata);
  self->cdata.~MaybeOwned<c10::Storage>();
  self->cdata = c10::MaybeOwned<c10::Storage>::owned(
      c10::Storage(c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(ptr)));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

PyObject* THMPStorage_byteswap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 1, "tuple of 1 item expected");
  PyObject* _elem_size = PyTuple_GET_ITEM(args, 0);
  THPUtils_assert(
      THPUtils_checkLong(_elem_size), "_byteswap(): arg must be an 'int'");
  auto elem_size = THPUtils_unpackLong(_elem_size);
  THPUtils_assert(
      elem_size == 1 || elem_size == 2 || elem_size == 4 || elem_size == 8,
      "elem_size must be 1, 2, 4, or 8");

  const auto& storage = THMPStorage_Unpack(self);
  const auto nbytes = static_cast<uint64_t>(storage.nbytes());
  const uint64_t count = nbytes / elem_size;

  if (elem_size == 1) {
    Py_RETURN_NONE;
  }
  THPUtils_assert(
      nbytes % elem_size == 0,
      "the length of data is not a multiple of %ld",
      elem_size);

  if (elem_size == 2) {
    auto buffer = static_cast<uint16_t*>(storage.mutable_data());
    for (uint64_t i = 0; i < count; i++, buffer++) {
      *buffer = thp_bswap16(*buffer);
    }
  } else if (elem_size == 4) {
    auto buffer = static_cast<uint32_t*>(storage.mutable_data());
    for (uint64_t i = 0; i < count; i++, buffer++) {
      *buffer = thp_bswap32(*buffer);
    }
  } else if (elem_size == 8) {
    auto buffer = static_cast<uint64_t*>(storage.mutable_data());
    for (uint64_t i = 0; i < count; i++, buffer++) {
      *buffer = thp_bswap64(*buffer);
    }
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef THMPStorage_methods[] = {
    {"copy_",
     castPyCFunctionWithKeywords(THMPStorage_copy_),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"element_size", THMPStorage_elementSize, METH_NOARGS, nullptr},
    {"fill_", THMPStorage_fill_, METH_O, nullptr},
    {"new", THMPStorage_new, METH_NOARGS, nullptr},
    {"resize_", THMPStorage_resize_, METH_O, nullptr},
    {"nbytes", THMPStorage_nbytes, METH_NOARGS, nullptr},
    {"data_ptr", THMPStorage_dataPtr, METH_NOARGS, nullptr},
    {"_write_file", THMPStorage_writeFile, METH_VARARGS, nullptr},
    {"_new_with_file",
     THMPStorage_newWithFile,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_set_from_file", THMPStorage_setFromFile, METH_VARARGS, nullptr},
    {"from_buffer",
     castPyCFunctionWithKeywords(THMPStorage_fromBuffer),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"from_file",
     castPyCFunctionWithKeywords(THMPStorage_fromFile),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"_set_cdata", THMPStorage__setCdata, METH_O, nullptr},
    {"_byteswap", THMPStorage_byteswap, METH_VARARGS, nullptr},
    {nullptr}};

PyMethodDef* THMPStorage_getMethods() {
  return THMPStorage_methods;
}
