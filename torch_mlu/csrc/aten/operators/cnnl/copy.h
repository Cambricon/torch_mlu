#pragma once

#include <ATen/ATen.h>
#include "aten/utils/cnnl_util.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_copy_impl(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking = true);

} // namespace ops
} // namespace torch_mlu
