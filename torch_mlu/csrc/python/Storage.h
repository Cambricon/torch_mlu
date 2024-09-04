#ifndef THMP_STORAGE_INC
#define THMP_STORAGE_INC

#include <torch/csrc/Types.h>
#include "utils/Export.h"
#define THMPStorageStr "torch.mlu.UntypedStorage"
#define THMPStorageBaseStr "StorageBase"
#define THMPStoragePtr THPStoragePtr
#define THMPStorage THPStorage
#define THMPStorage_Unpack THPStorage_Unpack

PyObject* THMPStorage_New(c10::Storage storage);
extern PyObject* THMPStorageClass;

TORCH_MLU_API bool THMPStorage_init(PyObject* module);
TORCH_MLU_API void THMPStorage_postInit(PyObject* module);

extern PyTypeObject THMPStorageType;

#endif
