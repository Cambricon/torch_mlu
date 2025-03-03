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
#ifndef KERNELS_SPARSE_ADD_SPARSE_ADD_KERNEL_H_
#define KERNELS_SPARSE_ADD_SPARSE_ADD_KERNEL_H_

#if defined(__BANG__)
template <typename IndiceType, typename ValueType>
__mlu_global__ void unionApplySparseAddDense(IndiceType *indices,
                                             ValueType *values,
                                             ValueType *output,
                                             const float alpha,
                                             const size_t nnz,
                                             const int sparse_dim,
                                             const size_t dense_size,
                                             const size_t gap_size);
#endif

#endif  // KERNELS_SPARSE_ADD_SPARSE_ADD_KERNEL_H_
