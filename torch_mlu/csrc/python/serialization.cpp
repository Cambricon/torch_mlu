#include "framework/core/mlu_guard.h"
#include "python/THMP.h"
#include <c10/core/CPUAllocator.h>
#include "framework/core/caching_allocator.h"

// save_save is necessary since the old eager format saved storages as
// [size + data], but the v1.5 eager format removes this since size is saved in
// the filesize.
template <class io>
void THMPStorage_writeFileRaw(
    c10::StorageImpl* self,
    io fd,
    bool save_size,
    uint64_t element_size) {
  torch_mlu::mlu::MLUGuard guard(self->device());
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables
  uint8_t* data;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::unique_ptr<char[]> cpu_data;
  int64_t size_bytes = self->nbytes();
  int64_t numel = size_bytes / element_size;
  if (self->device_type() == at::kPrivateUse1) {
    cpu_data = std::unique_ptr<char[]>(new char[size_bytes]);
    data = (uint8_t*)cpu_data.get();
    TORCH_CNRT_CHECK(cnrtMemcpy(
        data, (uint8_t*)self->data(), size_bytes, cnrtMemcpyDevToHost));
  } else {
    TORCH_CHECK(
        false, "writeFileRaw: Device not recognized: ", self->device_type());
  }

  if (save_size) {
    if (torch::utils::THP_nativeByteOrder() ==
        torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
      doWrite(fd, &numel, sizeof(int64_t));
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t nsize; // convert big endian cpu to little endian storage
      torch::utils::THP_encodeInt64Buffer(
          (uint8_t*)&nsize,
          (const int64_t*)&numel,
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
          1);
      doWrite(fd, &nsize, sizeof(int64_t));
    }
  }

  // fast track for bytes and little endian
  if (element_size == 1 ||
      torch::utils::THP_nativeByteOrder() ==
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    doWrite(fd, data, size_bytes);
  } else {
    int64_t buffer_size = std::min(numel, (int64_t)5000);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    std::unique_ptr<uint8_t[]> le_buffer(
        new uint8_t[buffer_size * element_size]);
    for (int64_t i = 0; i < numel; i += buffer_size) {
      size_t to_convert = std::min(numel - i, buffer_size);
      if (element_size == 2) {
        torch::utils::THP_encodeInt16Buffer(
            (uint8_t*)le_buffer.get(),
            (const int16_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (element_size == 4) {
        torch::utils::THP_encodeInt32Buffer(
            (uint8_t*)le_buffer.get(),
            (const int32_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (element_size == 8) {
        torch::utils::THP_encodeInt64Buffer(
            (uint8_t*)le_buffer.get(),
            (const int64_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (element_size == 16) {
        // the native Pytorch has bug here
        // we need consider the case of complex<double> and complex<long>
        torch::utils::THP_encodeInt64Buffer(
            (uint8_t*)le_buffer.get(),
            (const int64_t*)data + i * 2,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert * 2);
      }
      doWrite(fd, le_buffer.get(), to_convert * element_size);
    }
  }
}

template void THMPStorage_writeFileRaw<int>(
    c10::StorageImpl* self,
    int fd,
    bool save_size,
    uint64_t element_size);
template void THMPStorage_writeFileRaw<PyObject*>(
    c10::StorageImpl* self,
    PyObject* fd,
    bool save_size,
    uint64_t element_size);

template <class io>
c10::intrusive_ptr<c10::StorageImpl> THMPStorage_readFileRaw(
    io file,
    c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size) {
  torch_mlu::mlu::OptionalMLUGuard guard;
  if (storage.defined()) {
    guard.set_device(storage->device());
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint8_t* data;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t size;
  doRead(file, &size, sizeof(int64_t));
  int64_t nbytes = element_size * size;
  if (torch::utils::THP_nativeByteOrder() ==
      torch::utils::THPByteOrder::THP_BIG_ENDIAN) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t nsize; // convert little endian storage to big endian cpu
    nsize = nbytes;
    torch::utils::THP_decodeInt64Buffer(
        &nbytes,
        (const uint8_t*)&nsize,
        torch::utils::THP_nativeByteOrder(),
        1);
  }
  if (!storage.defined()) {
    storage = c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        torch_mlu::MLUCachingAllocator::get(),
        /*resizable=*/true);
  } else {
    int64_t _storage_nbytes = storage->nbytes();
    TORCH_CHECK(
        _storage_nbytes == nbytes,
        "storage has wrong byte size: expected %ld got %ld",
        nbytes,
        _storage_nbytes);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::unique_ptr<char[]> cpu_data = std::unique_ptr<char[]>(new char[nbytes]);
  data = (uint8_t*)cpu_data.get();

  // fast track for bytes and little endian
  if (element_size == 1 ||
      torch::utils::THP_nativeByteOrder() ==
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    doRead(file, data, storage->nbytes());
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::unique_ptr<uint8_t[]> le_buffer(
        new uint8_t[buffer_size * element_size]);

    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      doRead(file, le_buffer.get(), element_size * to_convert);

      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (element_size == 2) {
        torch::utils::THP_decodeInt16Buffer(
            (int16_t*)data + i,
            le_buffer.get(),
            torch::utils::THP_nativeByteOrder(),
            to_convert);
      } else if (element_size == 4) {
        torch::utils::THP_decodeInt32Buffer(
            (int32_t*)data + i,
            le_buffer.get(),
            torch::utils::THP_nativeByteOrder(),
            to_convert);
      } else if (element_size == 8) {
        torch::utils::THP_decodeInt64Buffer(
            (int64_t*)data + i,
            le_buffer.get(),
            torch::utils::THP_nativeByteOrder(),
            to_convert);
      } else if (element_size == 16) {
        // the native Pytorch has bug here
        // we need consider the case of complex<double> and complex<long>
        to_convert *= 2;
        torch::utils::THP_decodeInt64Buffer(
            (int64_t*)data + i * 2,
            le_buffer.get(),
            torch::utils::THP_nativeByteOrder(),
            to_convert);
      }
    }
  }
  // TODO(mengpenghui): currently do not support memory copy of int64_t.
  TORCH_CNRT_CHECK(
      cnrtMemcpy((uint8_t*)storage->data(), data, nbytes, cnrtMemcpyHostToDev));
  return storage;
}
template c10::intrusive_ptr<c10::StorageImpl> THMPStorage_readFileRaw<int>(
    int fd,
    c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size);
template c10::intrusive_ptr<c10::StorageImpl> THMPStorage_readFileRaw<
    PyObject*>(
    PyObject* fd,
    c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size);
