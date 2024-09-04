#include <Python.h>

extern PyObject* initMLUModule(void);

extern "C" __attribute__((visibility("default"))) PyObject* PyInit__MLUC(void);

PyMODINIT_FUNC PyInit__MLUC(void) {
  return initMLUModule();
}
