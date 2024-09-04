#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_eye_out(int64_t n, at::Tensor& out) {
  // the default value of `m` equals to `n`
  return cnnl_eye_out(n, n, out);
}

at::Tensor& cnnl_eye_out(int64_t n, int64_t m, at::Tensor& out) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

  out.resize_({n, m});
  out.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  int64_t stride = out.stride(0) + out.stride(1);

  Tensor diag = out.as_strided({sz}, {stride});
  diag.fill_(1);
  return out;
}

} // namespace ops
} // namespace torch_mlu
