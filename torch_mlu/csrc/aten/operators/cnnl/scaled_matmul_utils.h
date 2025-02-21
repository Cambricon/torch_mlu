#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "c10/core/ScalarType.h"

namespace torch_mlu {
namespace ops {

c10::MaybeOwned<Tensor> resolve_conj_if_indicated(const Tensor& tensor, bool resolve_conj);
c10::MaybeOwned<Tensor> prepare_matrix_for_cnnl(const Tensor& tensor, bool& transpose_tensor, bool transpose_result);
c10::MaybeOwned<Tensor> prepare_matrix_for_cnnl(const Tensor& tensor, bool& transpose_tensor);

struct cnCommonArgs {
  cnCommonArgs(const Tensor& mat1, const Tensor& mat2, Tensor& c) {
    bool transpose_result = false, transpose_mat1 = false,
         transpose_mat2 = false;
    out = prepare_matrix_for_cnnl(c, transpose_result);
    mata = prepare_matrix_for_cnnl(
        transpose_result ? mat2 : mat1, transpose_mat1, transpose_result);
    matb = prepare_matrix_for_cnnl(
        transpose_result ? mat1 : mat2, transpose_mat2, transpose_result);
    auto mat1_sizes = mat1.sizes();
    auto mat2_sizes = mat2.sizes();
    if (transpose_result) {
      transpose_mat1 = !transpose_mat1;
      transpose_mat2 = !transpose_mat2;
      mat1_sizes = mata->sizes();
      mat2_sizes = matb->sizes();
    }

    m = mat1_sizes[transpose_result ? 1 : 0];
    k = mat1_sizes[transpose_result ? 0 : 1];
    n = mat2_sizes[transpose_result ? 0 : 1];
    lda = mata->stride((transpose_mat1 == transpose_result) ? 0 : 1);
    ldb = matb->stride((transpose_mat2 == transpose_result) ? 0 : 1);
    result_ld = out->stride(transpose_result ? 1 : 0);
    transa = transpose_mat1 ? 1 : 0;
    transb = transpose_mat2 ? 1 : 0;
    trans_result = transpose_result;
  }
  bool transa, transb, trans_result;
  int64_t m, n, k;
  int64_t lda, ldb, result_ld;
  c10::MaybeOwned<Tensor> mata, matb, out;
};

} // namespace ops
} // namespace torch_mlu 
