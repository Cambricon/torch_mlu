/*************************************************************************
 * Copyright (C) [2025] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef KERNELS_SPARSE_ADD_SPARSE_ADD_H_
#define KERNELS_SPARSE_ADD_SPARSE_ADD_H_
#include <mlu_op.h>
#include "kernels/kernel.h"
#include "bangc_helper_dtype.h"
#include "bangc_kernels.h"
#include "cnnl.h"

NAMESPACE_BANGC_KERNELS_BEGIN

// Group: sparse_add
/*!
 * @brief compute dense_tensor add sparse_tensor
 *
 * @param[in] handle
 * Handle to a Cambricon CNNL context that is used to manage MLU devices and
 * queues in the lgamma operation.
 * @param[in] sparse_indices
 * Pointer to the MLU memory that stores the indices of sparse_tensor.
 * @param[in] sparse_values
 * Pointer to the MLU memory that stores the values of sparse_tensor.
 * @param[in] dense_tensor
 * Pointer to the MLU memory that stores the dense_tensor.
 * @param[in] alpha
 * A parameter that multiplier for sparse_tensor.
 * @param[in] nnz
 * A parameter that nnz of sparse_tensor.
 * @param[in] tensor_shape
 * Pointer to the array that represented the shape of dense_tensor.
 * @param[in] sparse_dim
 * A parameter that sparse_dim of sparse_tensor.
 * @param[in] dense_dim
 * A parameter that dense_dim of sparse_tensor.
 * @param[in] output_num
 * A parameter that number of output_num.
 * @param[out] output
 * Pointer to the MLU memory that stores the output.
 *
 * @par Return
 * - ::BANGC_KERNELS_STATUS_SUCCESS
 *
 * @par Data Type
 * - The supported data types of input and output tensor are as follows:
 *   - sparse_indices: int32, int64
 *   - sparse_values: float, half, bloat16
 *   - dense_tensor: float, half, bfloat16
 *   - output: float, half, bfloat16
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * https://github.com/pytorch/pytorch/blob/v2.5.0/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu
 */
template <typename IndiceType, typename ValueType>
bangcKernelsStatus_t BANGC_KERNELS_WIN_API
mluSparseAddDense(const cnnlHandle_t handle,
                  IndiceType *sparse_indices,
                  ValueType *sparse_values,
                  ValueType *dense_tensor,
                  const float alpha,
                  const size_t nnz,
                  const int64_t *tensor_shape,
                  const int sparse_dim,
                  const int dense_dim,
                  const size_t output_num,
                  ValueType *output);

NAMESPACE_BANGC_KERNELS_END
#endif  // KERNELS_SPARSE_ADD_SPARSE_ADD_H_
