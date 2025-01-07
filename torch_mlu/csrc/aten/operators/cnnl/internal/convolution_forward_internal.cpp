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

#include "aten/operators/cnnl/internal/convolution_internal_utils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

BenchmarkCache<cnnlConvolutionFwdAlgoPerf_t> fwd_algos;

template <>
struct algorithm_search<cnnlConvolutionFwdAlgoPerf_t> {
  using perf_t = cnnlConvolutionFwdAlgoPerf_t;
  using algo_t = cnnlConvolutionForwardAlgo_t;

  static constexpr auto DEFAULT_ALGO = CNNL_CONVOLUTION_FWD_ALGO_DIRECT;

  static BenchmarkCache<perf_t>& cache() {
    return fwd_algos;
  }

  static std::vector<perf_t> findAlgorithms(
      const ConvolutionArgs& args,
      bool benchmark) {
    // TODO(CNNLCORE-23840): need cnnl give CUDNN_CONVOLUTION_FWD_ALGO_COUNT
    static constexpr int num_algos = 23;
    int returned_algo_count;

    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    if (!benchmark) {
      returned_algo_count = 1;
      // TODO(CNNLCORE-23840): When deterministic==false && benchmark==false,
      // non-deterministic training cannot be guaranteed.
      perf_results[0].algo = CNNL_CONVOLUTION_FWD_ALGO_DIRECT;
      getWorkspaceSize(args, perf_results[0].algo, &(perf_results[0].memory));
    } else {
      // Sets the algorithm search strategy
      // This mode guarantees to return global optimal algorithm, but might be
      // time-consuming.
      cnnlConvolutionFwdAlgoSearchMode_t search_mode =
          CNNL_CONVOLUTION_FWD_ALGO_SEARCH_EXHAUSTIVE;
      TORCH_CNNL_CHECK(cnnlSetConvolutionDescriptorAlgoSearchMode(
          args.cdesc.desc(), search_mode));

      // Use the algo with the largest mlu memory
      // CNNL_CONVOLUTION_FWD_ALGO_DIRECT to ensure that there is enough
      // memory during the search for the optimal algorithm.
      size_t workspace_size = 0;
      cnnlConvolutionForwardAlgo_t algo_max_space =
          CNNL_CONVOLUTION_FWD_ALGO_DIRECT;
      getWorkspaceSize(args, algo_max_space, &workspace_size);
      auto workspace_ptr =
          torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

      // cnnl will go through all possible kernels and find the algo with the
      // shortest hardware time.
      TORCH_CNNL_CHECK(cnnlFindConvolutionForwardAlgo(
          /* handle               */ args.handle,
          /* args.cdesc           */ args.cdesc.desc(),
          /* alpha                */ nullptr,
          /* x_desc               */ args.idesc.get(),
          /* x                    */ args.input_ptr,
          /* w_desc               */ args.wdesc.get(),
          /* w                    */ args.weight_ptr,
          /* bias_desc            */ args.bdesc.get(),
          /* bias                 */ args.bias_ptr,
          /* beta                 */ nullptr,
          /* y_desc               */ args.odesc.get(),
          /* y                    */ args.output_ptr,
          /* requested_algo_count */ num_algos,
          /* returned_algo_count  */ &returned_algo_count,
          /* perf_results         */ perf_results.get(),
          /* workspace            */ workspace_ptr.get(),
          /* workspace_size       */ workspace_size));

      // Free the cached blocks in our caching allocator. They are
      // needed here because the above benchmarking uses a huge amount of
      // memory, e.g. a few GBs.
      torch_mlu::MLUCachingAllocator::emptyCache();
    }
    return getValidAlgorithms<perf_t>(
        perf_results.get(), args, returned_algo_count);
  }

  // get algo.memory when not call cnnlFindConvolutionForwardAlgo to get algo
  static void getWorkspaceSize(
      const ConvolutionArgs& args,
      algo_t algo,
      size_t* workspace_size) {
    TORCH_CNNL_CHECK(cnnlGetConvolutionForwardWorkspaceSize(
        args.handle,
        args.idesc.get(),
        args.wdesc.get(),
        args.odesc.get(),
        args.bdesc.get(),
        args.cdesc.desc(),
        algo,
        workspace_size));
  }
};

at::Tensor& cnnl_convolution_forward_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    bool is_depth_wise_conv) {
  auto input_impl = getMluTensorImpl(input);
  auto weight_impl = getMluTensorImpl(weight);
  auto output_impl = getMluTensorImpl(output);

  ConvolutionArgs args{
      mlu_data_ptr(input_impl),
      mlu_data_ptr(output_impl),
      mlu_data_ptr(weight_impl)};

  // get current args.handle
  args.handle = getCurrentHandle();

  // prepare desc
  const int64_t input_dim = input.dim();
  cnnlTensorLayout_t layout =
      input_dim > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
  auto output_cnnl_type = getCnnlDataType(output.scalar_type());
  auto input_scalar_type = input.scalar_type();
  auto input_cnnl_type = getCnnlDataType(input_scalar_type);
  auto weight_cnnl_type = getCnnlDataType(weight.scalar_type());
  const bool promote_compute_dtype =
      (output_cnnl_type == CNNL_DTYPE_HALF ||
       output_cnnl_type == CNNL_DTYPE_BFLOAT16);
  auto compute_dtype =
      promote_compute_dtype ? CNNL_DTYPE_FLOAT : output_cnnl_type;
  // Modify allow_tf32 based on input tensor scalar type
  if (input_scalar_type != at::kFloat)
    allow_tf32 = false;
  const auto& weight_size = weight.sizes();

  if (is_can_coalesce_second_dim(
          weight_size, input_dim, padding, stride, dilation)) {
    constexpr int64_t fixed_dim = 4;
    std::vector<int64_t> tensor_shape;
    tensor_shape.resize(fixed_dim);
    coalesce_conv_second_dim(input, input_cnnl_type, args.idesc, tensor_shape);
    coalesce_conv_second_dim(
        weight, weight_cnnl_type, args.wdesc, tensor_shape);
    coalesce_conv_second_dim(
        output, output_cnnl_type, args.odesc, tensor_shape);
    int64_t padding_t[2] = {padding[1], padding[2]};
    int64_t stride_t[2] = {stride[1], stride[2]};
    int64_t dilation_t[2] = {dilation[1], dilation[2]};
    args.cdesc.set(
        fixed_dim,
        stride_t,
        padding_t,
        dilation_t,
        groups,
        compute_dtype,
        allow_tf32);
  } else if (is_can_coalesce_last_dim(
                 weight_size, input_dim, padding, stride, dilation)) {
    constexpr int64_t fixed_dim = 4;
    std::vector<int64_t> tensor_shape;
    tensor_shape.resize(fixed_dim);
    coalesce_conv_last_dim(input, input_cnnl_type, args.idesc, tensor_shape);
    coalesce_conv_last_dim(weight, weight_cnnl_type, args.wdesc, tensor_shape);
    coalesce_conv_last_dim(output, output_cnnl_type, args.odesc, tensor_shape);
    int64_t padding_t[2] = {padding[0], 0};
    int64_t stride_t[2] = {stride[0], 1};
    int64_t dilation_t[2] = {dilation[0], 1};
    args.cdesc.set(
        fixed_dim,
        stride_t,
        padding_t,
        dilation_t,
        groups,
        compute_dtype,
        allow_tf32);
  } else {
    args.idesc = getTensorDesc(input_impl, input_cnnl_type, layout);
    // depth wise only support 4 dimension.
    if (is_depth_wise_conv) {
      args.wdesc =
          getTensorDesc(weight_impl, weight_cnnl_type, CNNL_LAYOUT_HWCN);
    } else {
      args.wdesc = getTensorDesc(weight_impl, weight_cnnl_type, layout);
    }
    args.odesc = getTensorDesc(output_impl, output_cnnl_type, layout);
    args.cdesc.set(
        input_dim,
        stride.data(),
        padding.data(),
        dilation.data(),
        groups,
        compute_dtype,
        allow_tf32);
  }

  // prepare bias
  args.bias_ptr = nullptr;
  if (bias.defined() && bias.dim() != 0 && bias.numel() != 0) {
    TORCH_CHECK(
        bias.dim() == 1,
        "currently only support 1-dim bias in "
        "cnnl_float_convolution_internal when bias.dim() != 0, but got ",
        bias.dim(),
        " dim.");
    auto bias_impl = getMluTensorImpl(bias);
    args.bdesc = getTensorDesc(bias_impl, CNNL_LAYOUT_ARRAY);
    args.bias_ptr = mlu_data_ptr(bias_impl);
  }

  // set conv params for search.
  setConvolutionParams(
      &args.params,
      input,
      weight,
      // the condition for triggering cnnlFindConvolutionForwardAlgo must also
      // check for bias.
      bias,
      padding,
      stride,
      dilation,
      groups,
      deterministic,
      allow_tf32,
      layout);

  AlgoIterator<cnnlConvolutionFwdAlgoPerf_t>(args, benchmark)
      .try_all([&](const cnnlConvolutionFwdAlgoPerf_t& fwdAlgPerf) {
        auto workspace_ptr =
            torch_mlu::MLUCachingAllocator::get()->allocate(fwdAlgPerf.memory);

        TORCH_CNNL_CHECK(cnnlConvolutionForward(
            /* handle         */ args.handle,
            /* conv_desc      */ args.cdesc.desc(),
            /* algo           */ fwdAlgPerf.algo,
            /* alpha          */ nullptr,
            /* x_desc         */ args.idesc.get(),
            /* x_ptr          */ args.input_ptr,
            /* w_desc         */ args.wdesc.get(),
            /* w_ptr          */ args.weight_ptr,
            /* bias_desc      */ args.bdesc.get(),
            /* bias_ptr       */ args.bias_ptr,
            /* workspace      */ workspace_ptr.get(),
            /* workspace_size */ fwdAlgPerf.memory,
            /* beta           */ nullptr,
            /* y_desc         */ args.odesc.get(),
            /* y_ptr          */ args.output_ptr));
      });

  return output;
}

} // namespace ops
} // namespace torch_mlu
