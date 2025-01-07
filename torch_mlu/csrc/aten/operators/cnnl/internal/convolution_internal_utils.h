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

#include <vector>
#include <tuple>
#include <cstring>
#include <c10/util/irange.h>
#include <ATen/native/utils/ParamsHash.h>
#include "ATen/Tensor.h"
#include "aten/cnnl/cnnlOpDescriptors.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {

using std::memcmp;

inline bool is_can_coalesce_second_dim(
    const at::IntArrayRef& weight_size,
    const int input_dim,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& dilation) {
  return input_dim == 5 && weight_size[2] == 1 && stride[0] == 1 &&
      padding[0] == 0 && dilation[0] == 1;
}

// Dont add more mlu check for this function, so keep it force inline
// to where it be called. This is only used when conv3d convert to conv2d,
// and which is satisfied func is_can_coalesce_second_dim.
inline void coalesce_conv_second_dim(
    const at::Tensor& self,
    cnnlDataType_t data_type,
    tensorDescPtr_t& desc,
    std::vector<int64_t>& shape) {
  auto combine_second_dims = [](const at::IntArrayRef& sizes,
                                std::vector<int64_t>& output) -> void {
    output[0] = sizes[0] * sizes[2];
    output[1] = sizes[1];
    output[2] = sizes[3];
    output[3] = sizes[4];
  };
  combine_second_dims(self.sizes(), shape);
  auto stride = std::move(get_channels_last_strides(shape));
  desc = getTensorDesc(shape, stride, data_type, CNNL_LAYOUT_NHWC);
}

inline bool is_can_coalesce_last_dim(
    const at::IntArrayRef& weight_size,
    const int input_dim,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& dilation) {
  return input_dim == 5 && weight_size[3] == 1 && weight_size[4] == 1 &&
      stride[1] == 1 && stride[2] == 1 && padding[1] == 0 && padding[2] == 0 &&
      dilation[1] == 1 && dilation[2] == 1;
}

// Dont add more mlu check for this function, so keep it force inline
// to where it be called. This is only used when conv3d convert to conv2d,
// and which is satisfied func is_can_coalesce_second_dim.
inline void coalesce_conv_last_dim(
    const at::Tensor& self,
    cnnlDataType_t data_type,
    tensorDescPtr_t& desc,
    std::vector<int64_t>& shape) {
  auto combine_last_dims = [](const at::IntArrayRef& sizes,
                              std::vector<int64_t>& output) -> void {
    output[0] = sizes[0];
    output[1] = sizes[1];
    output[2] = sizes[2];
    output[3] = sizes[3] * sizes[4];
  };
  combine_last_dims(self.sizes(), shape);
  auto stride = std::move(get_channels_last_strides(shape));
  desc = getTensorDesc(shape, stride, data_type, CNNL_LAYOUT_NHWC);
}

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;

struct ConvolutionParams {
  c10::DeviceIndex device_id;
  cnnlDataType_t dataType;
  int input_size[2 + max_dim];
  uint8_t input_dim;
  cnnlTensorLayout_t layout;
  int weight_size[2 + max_dim];
  int64_t padding[max_dim];
  int64_t stride[max_dim];
  int64_t dilation[max_dim];
  int64_t groups;
  bool deterministic;
  bool allow_tf32;
  bool has_bias;
};

inline void setConvolutionParams(
    ConvolutionParams* params,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const std::vector<int64_t> padding,
    const std::vector<int64_t> stride,
    const std::vector<int64_t> dilation,
    int64_t groups,
    bool deterministic,
    bool allow_tf32,
    cnnlTensorLayout_t layout) {
  auto dataType = getCnnlDataType(input.scalar_type());
  memset(params, 0, sizeof(ConvolutionParams));
  params->device_id = torch_mlu::current_device();
  params->dataType = dataType;
  params->input_dim = input.dim();
  params->layout = layout;
  for (int i = 0; i != params->input_dim; ++i) {
    params->input_size[i] = (int)input.sizes()[i];
    params->weight_size[i] = (int)weight.sizes()[i];
  }
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  params->groups = groups;
  params->deterministic = deterministic;
  params->allow_tf32 = allow_tf32;
  params->has_bias =
      static_cast<bool>(bias.defined() && bias.dim() != 0 && bias.numel() != 0);
}

// Convenience struct for passing around descriptors and datapointers
struct ConvolutionArgs {
  cnnlHandle_t handle;
  ConvolutionParams params;
  tensorDescPtr_t idesc;
  tensorDescPtr_t odesc;
  tensorDescPtr_t wdesc;
  tensorDescPtr_t bdesc;
  void* input_ptr;
  void* output_ptr;
  void* weight_ptr;
  void* bias_ptr;
  CnnlConvolutionDescriptor cdesc;

  ConvolutionArgs(void* input_ptr, void* output_ptr, void* weight_ptr)
      : input_ptr(input_ptr), output_ptr(output_ptr), weight_ptr(weight_ptr) {}
};

template <typename T>
struct BenchmarkCache {
  std::mutex mutex;
  std::unordered_map<
      ConvolutionParams,
      T,
      at::native::ParamsHash<ConvolutionParams>,
      at::native::ParamsEqual<ConvolutionParams>>
      map;

  bool find(const ConvolutionParams& params, T* results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *results = it->second;
    return true;
  }

  void insert(const ConvolutionParams& params, const T& results) {
    std::lock_guard<std::mutex> guard(mutex);
    map[params] = results;
  }
};

template <typename perf_t>
std::vector<perf_t> getValidAlgorithms(
    perf_t* perfResults,
    const ConvolutionArgs& args,
    int n_algo) {
  std::vector<perf_t> result;
  result.reserve(n_algo);

  for (const auto i : c10::irange(n_algo)) {
    perf_t perf = perfResults[i];

    // Double check documentation for cudnnFindConvolutionForwardAlgorithmEx
    if (!args.params.deterministic || perf.determinism == CNNL_DETERMINISTIC) {
      result.push_back(perf);
    }

    // TODO(CNNLCORE-23840): cnnl need modify algo.status when group conv,
    // now is CNNL_STATUS_NOT_SUPPORTED
    // if (perf.status == CNNL_STATUS_SUCCESS) {
    //   if (!args.params.deterministic ||
    //       perf.determinism == CNNL_DETERMINISTIC) {
    //     result.push_back(perf);
    //   }
    // }
  }
  TORCH_CHECK(
      result.size() > 0, "no valid convolution algorithms available in CNNL");
  return result;
}

template <typename perf_t>
struct algorithm_search {};

template <typename perf_t>
class AlgoIterator {
  using search = algorithm_search<perf_t>;
  const ConvolutionArgs& args;
  bool benchmark;

 public:
  AlgoIterator(const ConvolutionArgs& args, bool benchmark)
      : args(args), benchmark(benchmark) {}

  static std::vector<perf_t> onlyDefaultAlgorithm(const ConvolutionArgs& args) {
    std::vector<perf_t> perfResults(1);
    perfResults[0].algo = search::DEFAULT_ALGO;
    search::getWorkspaceSize(
        args, perfResults[0].algo, &(perfResults[0].memory));
    return perfResults;
  }

  void try_all(std::function<void(const perf_t& perf)> f) {
    bool only_use_default = args.params.deterministic && !benchmark;

    auto& cache = search::cache();
    perf_t algoPerf;
    if (!only_use_default && cache.find(args.params, &algoPerf)) {
      try {
        f(algoPerf);
        return;
      } catch (c10::OutOfMemoryError& e) {
        cnrtGetLastError(); // clear error
      }
    }

    auto perfResults = only_use_default
        ? onlyDefaultAlgorithm(args)
        : search::findAlgorithms(args, benchmark);
    for (auto& algoPerf : perfResults) {
      try {
        f(algoPerf);
        cache.insert(args.params, algoPerf);
        return;
      } catch (c10::OutOfMemoryError& e) {
        cnrtGetLastError(); // clearerror
      } catch (c10::Error& e) {
        cnrtGetLastError(); // error
      }
    }

    TORCH_CHECK(
        false, "Unable to find a valid CNNL algorithm to run convolution");
  }
};

} // namespace ops
} // namespace torch_mlu
