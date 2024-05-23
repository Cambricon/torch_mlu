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

#pragma once

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <vector>
#include "aten/utils/exceptions.h"
#include "aten/utils/tensor_util.h"
#include "cnnl.h" //  NOLINT

namespace torch_mlu {

template <typename T, cnnlStatus_t (*dtor)(T*)>
struct CnnlHeuristicResultDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CNNL_CHECK(dtor(x));
    }
  }
};

template <typename T, cnnlStatus_t (*ctor)(T**), cnnlStatus_t (*dtor)(T*)>
class CnnlHeuristicResult {
 public:
  // Use hr() to access the underlying HeuristicResult pointer in
  // a read-only fashion.  Most client code should use this.
  // If the HeuristicResult was never initialized, this will return
  // nullptr.
  T* hr() const {
    return hr_.get();
  }
  T* hr() {
    return hr_.get();
  }

  // Use mut_hr() to access the underlying HeuristicResult pointer
  // if you intend to modify what it points to This will ensure
  // that the HeuristicResult is initialized.
  // Code in this file will use this function.
  T* mut_hr() {
    init();
    return hr_.get();
  }

 protected:
  void init() {
    if (hr_ == nullptr) {
      T* raw_hr;
      TORCH_CNNL_CHECK(ctor(&raw_hr));
      hr_.reset(raw_hr);
    }
  }

 private:
  std::unique_ptr<T, CnnlHeuristicResultDeleter<T, dtor>> hr_;
};

class CnnlMatmulHeuristicResult : public CnnlHeuristicResult<
                                      cnnlMatMulHeuristicResult,
                                      &cnnlCreateMatMulHeuristicResult,
                                      &cnnlDestroyMatMulHeuristicResult> {
 public:
  CnnlMatmulHeuristicResult() {}
  void get(
      cnnlHandle_t handle,
      cnnlMatMulDescriptor_t matmul_desc,
      cnnlTensorDescriptor_t self_desc,
      cnnlTensorDescriptor_t other_desc,
      cnnlTensorDescriptor_t output_desc,
      cnnlTensorDescriptor_t out_of_place_desc,
      cnnlMatMulPrefer_t preference,
      int requested_algo_count,
      int* return_algo_count);
};

class CnnlMatmulExHeuristicResult : public CnnlHeuristicResult<
                                        cnnlMatMulExHeuristicResult,
                                        &cnnlCreateMatMulExHeuristicResult,
                                        &cnnlDestroyMatMulExHeuristicResult> {
 public:
  CnnlMatmulExHeuristicResult() {}
  void get(
      cnnlHandle_t handle,
      cnnlMatMulExDescriptor_t matmul_desc,
      cnnlTensorDescriptor_t self_desc,
      cnnlTensorDescriptor_t other_desc,
      cnnlTensorDescriptor_t output_desc,
      cnnlTensorDescriptor_t out_of_place_desc,
      cnnlMatMulExPrefer_t preference,
      int requested_algo_count,
      int* return_algo_count);
};

class CnnlStrideBatchMatmulHeuristicResult
    : public CnnlHeuristicResult<
          cnnlStrideBatchMatMulHeuristicResult,
          &cnnlCreateStrideBatchMatMulHeuristicResult,
          &cnnlDestroyStrideBatchMatMulHeuristicResult> {
 public:
  CnnlStrideBatchMatmulHeuristicResult() {}
  void get(
      cnnlHandle_t handle,
      bool transa,
      bool transb,
      float alpha,
      float beta,
      int m,
      int n,
      int k,
      int lda,
      int ldb,
      int ldc,
      int ldd,
      int64_t stride_a,
      int64_t stride_b,
      int64_t stride_c,
      int64_t stride_d,
      int* bs_ptr,
      cnnlStrideBatchMatMulDescriptor_t bmm_desc,
      cnnlTensorDescriptor_t self_desc,
      cnnlTensorDescriptor_t other_desc,
      cnnlTensorDescriptor_t output_desc,
      cnnlTensorDescriptor_t out_of_place_desc,
      cnnlMatMulPrefer_t preference,
      int requested_algo_count,
      int* return_algo_count);
};

} //  namespace torch_mlu
