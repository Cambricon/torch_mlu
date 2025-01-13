/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/pytorch/pytorch/graphs/contributors Redistribution and use in
source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef TORCH_MLU_CSRC_PYTHON_MODULE_H_
#define TORCH_MLU_CSRC_PYTHON_MODULE_H_

void THMPModule_setDevice(int device);
void THMPModule_initModule();
int THMPModule_getDeviceCount_wrap();
PyObject* THMPModule_setDevice_wrap(PyObject* self, PyObject* arg);
PyObject* THMPModule_exchangeDevice(PyObject* self, PyObject* arg);
PyObject* THMPModule_maybeExchangeDevice(PyObject* self, PyObject* arg);
PyObject* THMPModule_mluSynchronize(PyObject* _unused, PyObject* noargs);
PyObject* THMPModule_mluSleep(PyObject* _unused, PyObject* cycles);
void THMPModule_methods(PyObject* module);
void registerMLUDeviceProperties(PyObject* module);
void registerMluAllocator(PyObject* module);
void registerMluPluggableAllocator(PyObject* module);
PyObject* THMPModule_getDevice_wrap(PyObject* self, PyObject* noargs);
#endif // TORCH_MLU_CSRC_PYTHON_MODULE_H_
