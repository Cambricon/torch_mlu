#pragma once

#include "aten/cnnl/cnnlHandle.h"
#include "aten/cnnl/cnnlTensorDesc.h"
#include "aten/utils/tensor_util.h"
#include "framework/core/mlu_guard.h"
#include <ATen/native/utils/ParamsHash.h>

#include <list>

namespace torch_mlu {
namespace ops {
namespace detail {

constexpr int kMaxRank =
    1; // currently only support 1-d RFFT because of CNNL limitation

#define CNNL_FFT_L_LIMIT 4096

// Enum representing the FFT type
enum class CnFFTTransformType : int8_t {
  C2C, // Complex-to-complex
  R2C, // Real-to-complex
  C2R, // Complex-to-real
};

using CnFFTDimVector = c10::SmallVector<int, at::kDimVectorStaticSize>;

// This struct is used to let us easily compute hashes of the
// parameters.
// It will be the **key** to the plan cache.
struct CnFFTParams {
  int64_t signal_ndim_; // between 1 and kMaxRank
  // These include additional batch dimension as well.
  int64_t sizes_[kMaxRank + 1];
  int64_t input_strides_[kMaxRank + 1];
  int64_t output_strides_[kMaxRank + 1];
  CnFFTTransformType fft_type_;
  at::ScalarType value_type_;

  CnFFTParams() = default;

  CnFFTParams(
      c10::IntArrayRef in_strides,
      c10::IntArrayRef out_strides,
      c10::IntArrayRef signal_sizes,
      CnFFTTransformType fft_type,
      at::ScalarType value_type) {
    // Padding bits must be zeroed for hashing
    memset(this, 0, sizeof(*this));
    signal_ndim_ = signal_sizes.size() - 1;
    fft_type_ = fft_type;
    value_type_ = value_type;

    TORCH_INTERNAL_ASSERT(in_strides.size() == signal_sizes.size());
    TORCH_INTERNAL_ASSERT(out_strides.size() == signal_sizes.size());
    TORCH_INTERNAL_ASSERT(1 <= signal_ndim_ && signal_ndim_ <= kMaxRank);

    std::copy(signal_sizes.cbegin(), signal_sizes.cend(), sizes_);
    std::copy(in_strides.cbegin(), in_strides.cend(), input_strides_);
    std::copy(out_strides.cbegin(), out_strides.cend(), output_strides_);
  }
};

static_assert(std::is_trivial<CnFFTParams>::value, "");

// Create transform type enum from bools representing if input and output are
// complex
inline CnFFTTransformType get_cnfft_transformtype(
    bool complex_input,
    bool complex_output) {
  if (complex_input && complex_output) {
    return CnFFTTransformType::C2C;
  } else if (complex_input && !complex_output) {
    return CnFFTTransformType::C2R;
  } else if (!complex_input && complex_output) {
    return CnFFTTransformType::R2C;
  }
  TORCH_INTERNAL_ASSERT(false, "Real to real FFTs are not supported");
}

class CnFFTHandle {
  ::cnnlFFTPlan_t plan_;

 public:
  CnFFTHandle() {
    TORCH_CNNL_CHECK(cnnlCreateFFTPlan(&plan_));
  }

  ::cnnlFFTPlan_t& get() {
    return plan_;
  }
  const ::cnnlFFTPlan_t& get() const {
    return plan_;
  }

  ~CnFFTHandle() {
    cnnlDestroyFFTPlan(plan_);
  }
};

static bool is_pow_of_two(int64_t x) {
  return (x & (x - 1)) == 0;
}

static int64_t cal_cnnl_fft_length_factorization(const int64_t n) {
  auto l = n;
  while (!(l & 1)) {
    l >>= 1;
  }
  return l;
}

// This class contains all the information needed to execute a cnFFT plan:
//   1. the plan
//   2. the workspace size needed
//   3. the reservespace needed
//
// This class will be the **value** in the plan cache.
class CnFFTConfig {
 public:
  // Only move semantics is enought for this class. Although we already use
  // unique_ptr for the plan, still remove copy constructor and assignment op so
  // we don't accidentally copy and take perf hit.
  CnFFTConfig(const CnFFTConfig&) = delete;
  CnFFTConfig& operator=(CnFFTConfig const&) = delete;

  explicit CnFFTConfig(
      const CnFFTParams& params,
      const at::Tensor& input,
      const at::Tensor& output)
      : CnFFTConfig(
            c10::IntArrayRef(params.sizes_, params.signal_ndim_ + 1),
            input,
            output,
            params.value_type_) {}

  // sizes are for the full signal, including batch size and always two-sided
  CnFFTConfig(
      c10::IntArrayRef sizes,
      const at::Tensor& input,
      const at::Tensor& output,
      at::ScalarType value_type) {
    // signal sizes (excluding batch dim)
    CnFFTDimVector signal_sizes(sizes.begin() + 1, sizes.end());

    const int signal_ndim = static_cast<int>(sizes.size() - 1);

    if (value_type == at::ScalarType::Half) { // half or complex_half
      for (int64_t i = 0; i < signal_ndim; i++) {
        TORCH_CHECK(
            is_pow_of_two(sizes[i + 1]),
            "CNNL FFT only supports dimensions whose sizes are powers of two "
            "when"
            " computing in half precision, but got a signal size of",
            sizes.slice(1));
      }
    } else { // float or complex_float
      for (int64_t i = 0; i < signal_ndim; i++) {
        auto signal_size = sizes[i + 1];
        auto l = cal_cnnl_fft_length_factorization(signal_size);
        TORCH_CHECK(
            l <= CNNL_FFT_L_LIMIT,
            "when the length of FFT > 4096, "
            "the length must can be factorized into 2 ^ m * l, and l <= 4096"
            ", but got a signal size of ",
            sizes.slice(1));
      }
    }

    auto input_impl = getMluTensorImpl(input);
    auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
    auto onchip_dtype =
        value_type == at::ScalarType::Half ? CNNL_DTYPE_HALF : CNNL_DTYPE_FLOAT;
    TORCH_CNNL_CHECK(
        cnnlSetTensorDescriptorOnchipDataType(input_desc.get(), onchip_dtype));

    auto output_impl = getMluTensorImpl(output);
    auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);

    size_t ws_size_t, rs_size_t;
    TORCH_CNNL_CHECK(cnnlMakeFFTPlanMany(
        getCurrentHandle(),
        plan(),
        input_desc.get(),
        output_desc.get(),
        signal_ndim,
        signal_sizes.data(),
        &rs_size_t,
        &ws_size_t));
    ws_size = static_cast<int64_t>(ws_size_t);

    // malloc memory without caching_allocator to prevent dependency between
    // global variable plan_caches and caching_allocator
    TORCH_CNRT_CHECK(cnrtMalloc(&reservespace_ptr, rs_size_t));
    device_id = torch_mlu::current_device();
  }

  ~CnFFTConfig() {
    try {
      if (reservespace_ptr) {
        torch_mlu::mlu::MLUGuard guard(device_id);
        TORCH_CNRT_CHECK(cnrtSyncDevice());
        TORCH_CNRT_CHECK(cnrtFree(reservespace_ptr));
      }
    } catch (...) { /* No throw */
    }
  }

  const cnnlFFTPlan_t& plan() const {
    return plan_ptr.get();
  }

  int64_t workspace_size() const {
    return ws_size;
  }

  void* get_reservespace_ptr() const {
    return reservespace_ptr;
  }

 private:
  CnFFTHandle plan_ptr;
  int64_t ws_size;
  void* reservespace_ptr;
  c10::DeviceIndex device_id;
};

// This cache assumes that the mapping from key to value never changes.
// This is **NOT** thread-safe. Please use a mutex when using it **AND** the
// value returned from try_emplace_value.
// The contract of using this cache is that try_emplace_value should only be
// used when the max_size is positive.
class CnFFTParamsLRUCache {
 public:
  using kv_t = typename std::pair<CnFFTParams, CnFFTConfig>;
  using map_t = typename std::unordered_map<
      std::reference_wrapper<CnFFTParams>,
      typename std::list<kv_t>::iterator,
      at::native::ParamsHash<CnFFTParams>,
      at::native::ParamsEqual<CnFFTParams>>;
  using map_kkv_iter_t = typename map_t::iterator;

  // Different from cuFFT, CNNL FFT currently do not have the max plan number
  // limitation.
  CnFFTParamsLRUCache()
      : CnFFTParamsLRUCache(std::numeric_limits<int64_t>::max()) {}

  CnFFTParamsLRUCache(int64_t max_size) {
    _set_max_size(max_size);
  }

  CnFFTParamsLRUCache(CnFFTParamsLRUCache&& other) noexcept
      : _usage_list(std::move(other._usage_list)),
        _cache_map(std::move(other._cache_map)),
        _max_size(other._max_size) {}

  CnFFTParamsLRUCache& operator=(CnFFTParamsLRUCache&& other) noexcept {
    _usage_list = std::move(other._usage_list);
    _cache_map = std::move(other._cache_map);
    _max_size = other._max_size;
    return *this;
  }

  // If key is in this cache, return the cached config. Otherwise, emplace the
  // config in this cache and return it.
  // Return const reference because CnFFTConfig shouldn't be tampered with once
  // created.
  const CnFFTConfig& lookup(
      CnFFTParams params,
      const at::Tensor& input,
      const at::Tensor& output) {
    AT_ASSERT(_max_size > 0);

    map_kkv_iter_t map_it = _cache_map.find(params);
    // Hit, put to list front
    if (map_it != _cache_map.end()) {
      _usage_list.splice(_usage_list.begin(), _usage_list, map_it->second);
      return map_it->second->second;
    }

    // Miss, remove if needed
    if (_usage_list.size() >= _max_size) {
      auto last = _usage_list.end();
      last--;
      _cache_map.erase(last->first);
      _usage_list.pop_back();
    }

    // construct new plan at list front, then insert into _cache_map
    _usage_list.emplace_front(
        std::piecewise_construct,
        std::forward_as_tuple(params),
        std::forward_as_tuple(params, input, output));
    auto kv_it = _usage_list.begin();
    _cache_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(kv_it->first),
        std::forward_as_tuple(kv_it));
    return kv_it->second;
  }

  void clear() {
    _cache_map.clear();
    _usage_list.clear();
  }

  void resize(int64_t new_size) {
    _set_max_size(new_size);
    auto cur_size = _usage_list.size();
    if (cur_size > _max_size) {
      auto delete_it = _usage_list.end();
      for (size_t i = 0; i < cur_size - _max_size; i++) {
        delete_it--;
        _cache_map.erase(delete_it->first);
      }
      _usage_list.erase(delete_it, _usage_list.end());
    }
  }

  size_t size() const {
    return _cache_map.size();
  }

  size_t max_size() const noexcept {
    return _max_size;
  }

  std::mutex mutex;

 private:
  // Only sets size and does value check. Does not resize the data structures.
  void _set_max_size(int64_t new_size) {
    TORCH_CHECK(
        new_size >= 0,
        "cnFFT plan cache size must be non-negative, but got ",
        new_size);
    _max_size = static_cast<size_t>(new_size);
  }

  std::list<kv_t> _usage_list;
  map_t _cache_map;
  size_t _max_size;
};

// Since MLU build is separated from CPU build, we need a way to call
// these functions only when MLU is loaded. We use MLU hooks for this purpose
// (at aten/util/MLUHooks.cpp), and call the hooked functions
// from the actual function counterparts (at native/SpectralOps.cpp), i.e.,
// _cufft_get_plan_cache_max_size, _cufft_set_plan_cache_max_size
// _cufft_get_plan_cache_size, and _cufft_clear_plan_cache.
int64_t cnfft_get_plan_cache_max_size_impl(int64_t device_index);
void cnfft_set_plan_cache_max_size_impl(int64_t device_index, int64_t max_size);
int64_t cnfft_get_plan_cache_size_impl(int64_t device_index);
void cnfft_clear_plan_cache_impl(int64_t device_index);

} // namespace detail
} // namespace ops
} // namespace torch_mlu
