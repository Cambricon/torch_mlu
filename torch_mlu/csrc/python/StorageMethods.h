#ifndef THMP_STORAGE_METHODS_INC
#define THMP_STORAGE_METHODS_INC

#include <Python.h>
void resize_bytes_mlu(c10::StorageImpl* storage, size_t size_bytes);

PyMethodDef* THMPStorage_getMethods();

#endif
