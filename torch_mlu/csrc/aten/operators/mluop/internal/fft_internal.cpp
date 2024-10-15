#include "aten/operators/mluop/internal/mluop_internal.h"
#include "aten/operators/mluop/internal/cnfft_plan_cache.h"

namespace torch_mlu {
namespace ops {

#define CNFFT_FORWARD 0 // Forward FFT
#define CNFFT_INVERSE 1 // Inverse FFT

// The cnFFT plan cache
// unique_ptr for nullability and to avoid reference invalidation on vector
// resize
static std::vector<std::unique_ptr<detail::CnFFTParamsLRUCache>> plan_caches;
static std::mutex plan_caches_mutex;

static inline detail::CnFFTParamsLRUCache& cnfft_get_plan_cache(
    at::DeviceIndex device_index) {
  std::lock_guard<std::mutex> guard(plan_caches_mutex);

  AT_ASSERT(device_index >= 0);

  if (device_index >= static_cast<int64_t>(plan_caches.size())) {
    plan_caches.resize(device_index + 1);
  }

  if (!plan_caches[device_index]) {
    plan_caches[device_index] = std::make_unique<detail::CnFFTParamsLRUCache>();
  }

  return *plan_caches[device_index];
}

namespace detail {

int64_t cnfft_get_plan_cache_max_size_impl(int64_t device_index) {
  TORCH_CHECK(
      0 <= device_index && device_index < device_count(),
      "cnfft_get_plan_cache_max_size: expected 0 <= device_index < ",
      device_count(),
      "], but got device_index=",
      device_index);
  return cnfft_get_plan_cache(device_index).max_size();
}

void cnfft_set_plan_cache_max_size_impl(
    int64_t device_index,
    int64_t max_size) {
  TORCH_CHECK(
      0 <= device_index && device_index < device_count(),
      "cnfft_set_plan_cache_max_size: expected 0 <= device_index < ",
      device_count(),
      "], but got device_index=",
      device_index);
  return cnfft_get_plan_cache(device_index).resize(max_size);
}

int64_t cnfft_get_plan_cache_size_impl(int64_t device_index) {
  TORCH_CHECK(
      0 <= device_index && device_index < device_count(),
      "cnfft_get_plan_cache_size: expected 0 <= device_index < ",
      device_count(),
      "], but got device_index=",
      device_index);
  return cnfft_get_plan_cache(device_index).size();
}

void cnfft_clear_plan_cache_impl(int64_t device_index) {
  TORCH_CHECK(
      0 <= device_index && device_index < device_count(),
      "cnfft_clear_plan_cache: expected 0 <= device_index < ",
      device_count(),
      "], but got device_index=",
      device_index);
  return cnfft_get_plan_cache(device_index).clear();
}

} // namespace detail

// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r),
// Note: CNNL FFT currently only support r2c.
// CNNL FFT is different from cuFFT mainly in two ways currently:
// 1. CNNL FFT needs invokers to malloc and manage reservespace for them.
// 2. CNNL FFT fuses scaling up into cnnlExecFFT.
const at::Tensor& mluop_fft_internal(
    at::Tensor& out,
    const at::Tensor& self,
    c10::IntArrayRef out_sizes,
    c10::IntArrayRef dim,
    bool forward,
    const float scale_factor) {
  const auto ndim = self.dim();
  const int64_t signal_ndim = dim.size();
  const auto batch_dims = ndim - signal_ndim;

  // Permute dimensions so batch dimensions come first, and in stride order
  // This maximizes data locality when collapsing to a single batch dimension
  at::DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

  c10::SmallVector<bool, at::kDimVectorStaticSize> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }
  auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(), [&](int64_t d) {
        return !is_transformed_dim[d];
      });
  auto self_strides = self.strides();
  std::sort(dim_permute.begin(), batch_end, [&](int64_t a, int64_t b) {
    return self_strides[a] > self_strides[b];
  });
  std::copy(dim.cbegin(), dim.cend(), batch_end);
  auto input = self.permute(dim_permute);

  // Collapse batch dimensions into a single dimension
  at::DimVector batched_sizes(signal_ndim + 1);
  batched_sizes[0] = -1;
  std::copy(
      input.sizes().cbegin() + batch_dims,
      input.sizes().cend(),
      batched_sizes.begin() + 1);
  input = input.reshape(batched_sizes);

  const auto batch_size = input.sizes()[0];
  at::DimVector signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;
  for (int64_t i = 0; i < signal_ndim; ++i) {
    auto in_size = input.sizes()[i + 1];
    auto out_size = out_sizes[dim[i]];
    signal_size[i + 1] = std::max(in_size, out_size);
    TORCH_INTERNAL_ASSERT(
        in_size == signal_size[i + 1] ||
        in_size == (signal_size[i + 1] / 2) + 1);
    TORCH_INTERNAL_ASSERT(
        out_size == signal_size[i + 1] ||
        out_size == (signal_size[i + 1] / 2) + 1);
  }

  batched_sizes[0] = batch_size;
  at::DimVector batched_out_sizes(batched_sizes.begin(), batched_sizes.end());
  for (size_t i = 0; i < dim.size(); ++i) {
    batched_out_sizes[i + 1] = out_sizes[dim[i]];
  }
  out.resize_(batched_out_sizes, c10::MemoryFormat::Contiguous);

  // Create the transform plan (either from cache or locally)
  auto value_type = c10::toRealValueType(input.scalar_type());
  if (value_type == at::ScalarType::Double) {
    value_type = at::ScalarType::Float; // MLU do not support 64 bit dtype
  }
  auto fft_type =
      detail::get_cnfft_transformtype(input.is_complex(), out.is_complex());
  detail::CnFFTParams params(
      input.strides(), out.strides(), signal_size, fft_type, value_type);
  detail::CnFFTParamsLRUCache& plan_cache =
      cnfft_get_plan_cache(input.device().index());
  std::unique_lock<std::mutex> guard(plan_cache.mutex, std::defer_lock);
  std::optional<detail::CnFFTConfig> uncached_plan;
  const detail::CnFFTConfig* config = nullptr;

  if (plan_cache.max_size() > 0) {
    guard.lock();
    if (plan_cache.max_size() > 0) { // check again after acquiring the lock
      config = &plan_cache.lookup(params, input, out);
    }
  }
  if (config == nullptr) {
    uncached_plan.emplace(params, input, out);
    config = &uncached_plan.value();
  }

  // Set reserve area for plan
  auto handle = getCurrentMluOpHandle();
  auto& plan = config->plan();
  TORCH_MLUOP_CHECK(
      mluOpSetFFTReserveArea(handle, plan, config->get_reservespace_ptr()));

  // run
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(config->workspace_size());
  TORCH_MLUOP_CHECK(mluOpExecFFT(
      /* handle       */ handle,
      /* fft_plan     */ plan,
      /* input        */ mlu_data_ptr(getMluTensorImpl(input)),
      /* scale_factor */ scale_factor,
      /* workspace    */ workspace_ptr.get(),
      /* output       */ mlu_data_ptr(getMluTensorImpl(out)),
      /* direction    */ forward ? CNFFT_FORWARD : CNFFT_INVERSE));

  // Inplace reshaping to original batch shape and inverting the dimension
  // permutation
  at::DimVector out_strides(ndim);
  int64_t batch_numel = 1;
  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    out_strides[dim_permute[i]] = batch_numel * out.strides()[0];
    batch_numel *= out_sizes[dim_permute[i]];
  }
  for (int64_t i = batch_dims; i < ndim; ++i) {
    out_strides[dim_permute[i]] = out.strides()[1 + (i - batch_dims)];
  }
  return out.as_strided_(out_sizes, out_strides, out.storage_offset());
}

} // namespace ops

TORCH_LIBRARY_FRAGMENT(torch_mlu, m) {
  const std::vector<at::Tag> tags_0 = {at::Tag::pt2_compliant_tag};
  m.def("_cnfft_get_plan_cache_size(DeviceIndex device_index) -> int", tags_0);
  m.def(
      "_cnfft_get_plan_cache_max_size(DeviceIndex device_index) -> int",
      tags_0);
  m.def(
      "_cnfft_set_plan_cache_max_size(DeviceIndex device_index, int max_size) -> ()",
      tags_0);
  m.def("_cnfft_clear_plan_cache(DeviceIndex device_index) -> ()", tags_0);
}

TORCH_LIBRARY_IMPL(torch_mlu, CompositeImplicitAutograd, m) {
  m.impl(
      "_cnfft_get_plan_cache_size",
      TORCH_FN(ops::detail::cnfft_get_plan_cache_size_impl));
  m.impl(
      "_cnfft_get_plan_cache_max_size",
      TORCH_FN(ops::detail::cnfft_get_plan_cache_max_size_impl));
  m.impl(
      "_cnfft_set_plan_cache_max_size",
      TORCH_FN(ops::detail::cnfft_set_plan_cache_max_size_impl));
  m.impl(
      "_cnfft_clear_plan_cache",
      TORCH_FN(ops::detail::cnfft_clear_plan_cache_impl));
}

namespace mlu {

int64_t _cnfft_get_plan_cache_size(at::DeviceIndex device_index) {
  return ops::detail::cnfft_get_plan_cache_size_impl(device_index);
}

int64_t _cnfft_get_plan_cache_max_size(at::DeviceIndex device_index) {
  return ops::detail::cnfft_get_plan_cache_max_size_impl(device_index);
}

void _cnfft_set_plan_cache_max_size(
    at::DeviceIndex device_index,
    int64_t max_size) {
  return ops::detail::cnfft_set_plan_cache_max_size_impl(
      device_index, max_size);
}

void _cnfft_clear_plan_cache(at::DeviceIndex device_index) {
  return ops::detail::cnfft_clear_plan_cache_impl(device_index);
}

} // namespace mlu

} // namespace torch_mlu
