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

#include <framework/generator/generator_impl.h>
#include <framework/core/tensor_impl.h>
#include <aten/utils/tensor_util.h>
#include "aten/utils/utils.h"
#include <ATen/Utils.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include "framework/core/device.h"
#include "framework/core/guard_impl.h"
#include "framework/core/mlu_guard.h"
#include "framework/graphs/MLUGraphUtils.h"
#include "framework/graphs/MLUGraph.h"

namespace torch_mlu {

// Ensures we only call mluGetDeviceCount only once
static std::once_flag num_mlu_init_flag;

// Total number of mlus in the system.
static int64_t num_mlus;

// Ensures default_gens_mlu is initialized once.
static std::deque<std::once_flag> mlu_gens_init_flag;

// Default, global MLU generators, one per MLU.
static std::vector<at::Generator> default_gens_mlu;

/*
 * Populates the global variables related to MLU generators
 * Warning: this function must only be called once!
 */
static void initMLUGenVector() {
  num_mlus = device_count();
  mlu_gens_init_flag.resize(num_mlus);
  default_gens_mlu.resize(num_mlus);
}

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultMLUGenerator gets the default generator for a particular
 * mlu device.
 */
const at::Generator& getDefaultMLUGenerator(at::DeviceIndex device_index) {
  std::call_once(num_mlu_init_flag, initMLUGenVector);
  at::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_mlus);
  }
  std::call_once(mlu_gens_init_flag[idx], [&] {
    default_gens_mlu[idx] = at::make_generator<MLUGeneratorImpl>(idx);
    default_gens_mlu[idx].seed();
  });
  return default_gens_mlu[idx];
}

/**
 * Utility to create a MLUGeneratorImpl. Returns a shared_ptr
 */
at::Generator createMLUGenerator(at::DeviceIndex device_index) {
  std::call_once(num_mlu_init_flag, initMLUGenVector);
  at::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_mlus, "The device_index is invalid.");
  auto generator = at::make_generator<MLUGeneratorImpl>(idx);
  auto gen_impl = at::check_generator<MLUGeneratorImpl>(generator);
  gen_impl->set_current_seed(c10::default_rng_seed_val);
  gen_impl->set_philox_offset_per_thread(0);
  return generator;
}

/**
 * Creates a clone of this MLU Generator State.
 */
c10::intrusive_ptr<MLUGeneratorState> MLUGeneratorState::clone() {
  return c10::make_intrusive<MLUGeneratorState>(
      seed_, philox_offset_per_thread_, offset_intragraph_);
}

/**
 * Function to increase the internal offset based on the specified increment.
 */
void MLUGeneratorState::increase(uint64_t increment) {
  // Rounds increment up to the nearest multiple of 4 to meet alignment
  // requirements.
  // see Note [Why enforce RNG offset % 4 == 0?]
  increment = ((increment + 3) / 4) * 4;
  // Handling different behaviors based on whether capturing is active.
  if (torch_mlu::currentStreamCaptureStatus() !=
      torch_mlu::CaptureStatus::None) {
    // Ensures that the state is actually capturing.
    TORCH_CHECK(
        capturing_,
        "Attempt to increase offset for a MLU generator not in capture mode.");
    // Ensures the offset is a multiple of 4
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(
        offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
    // Ensures the increment does not cause overflow.
    TORCH_INTERNAL_ASSERT(
        offset_intragraph_ <= std::numeric_limits<uint32_t>::max() - increment,
        "Increment causes overflow in the offset value.");
    offset_intragraph_ += increment;
  } else {
    // Checks that the increment is expected outside graph capturing.
    TORCH_CHECK(
        !capturing_,
        "Offset increment outside graph capture encountered unexpectedly.");
    // Ensures the offset is a multiple of 4
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(
        philox_offset_per_thread_ % 4 == 0,
        "RNG offset must be a multiple of 4.");
    philox_offset_per_thread_ += increment;
  }
}

/**
 * Registers this state to a MLU graph to manage within the graph.
 */
void MLUGeneratorState::register_graph(torch_mlu::MLUGraph* graph) {
  // Ensures that the RNG state is not currently being captured.
  torch_mlu::assertNotCapturing(
      "Cannot register the state during capturing stage.");

  // If this is the first graph to be registered, allocate memory for the seed
  // and offset on the MLU.
  if (registered_graphs_.empty()) {
    auto options =
        at::TensorOptions().device(at::kPrivateUse1).dtype(at::kLong);
    seed_extragraph_ = at::empty({1}, options);
    offset_extragraph_ = at::empty({1}, options);
  }

  // Insert the graph into the set of registered graphs if it's not already
  // registered.
  if (registered_graphs_.find(graph) == registered_graphs_.end()) {
    registered_graphs_.insert(graph);
  }
}

/**
 * Unregisters a MLU graph from the RNG state.
 */
void MLUGeneratorState::unregister_graph(torch_mlu::MLUGraph* graph) {
  // Verify the graph was previously registered.
  TORCH_CHECK(
      registered_graphs_.find(graph) != registered_graphs_.end(),
      "The graph should be registered to the state");

  // Remove the graph from the set of registered graphs.
  registered_graphs_.erase(graph);

  // If no more graphs are registered, deallocate the GPU memory for the seed
  // and offset.
  if (registered_graphs_.empty()) {
    seed_extragraph_.reset();
    offset_extragraph_.reset();
  }
}

/**
 * Performs the prologue steps for capturing a MLU graph state.
 * This method is intended to reset graph-related state variables before
 * capturing begins.
 */
void MLUGeneratorState::capture_prologue() {
  capturing_ = true;
  offset_intragraph_ = 0;
  seed_extragraph_.fill_(int64_t(seed_));
  offset_extragraph_.fill_(int64_t(0));
}

/**
 * Ends the capturing phase and resets related variables, returning the whole
 * graph increment.
 */
uint64_t MLUGeneratorState::capture_epilogue() {
  capturing_ = false;
  return offset_intragraph_;
}

/**
 * Prepares the state for replay by setting initial state tensors and applying
 * total increment.
 */
void MLUGeneratorState::replay_prologue(uint64_t wholegraph_increment) {
  // Ensures the generator is not in capturing mode.
  torch_mlu::assertNotCapturing(
      "Cannot prepare for replay during capturing stage.");
  seed_extragraph_.fill_(int64_t(seed_));
  offset_extragraph_.fill_(int64_t(philox_offset_per_thread_));
  // Applies the total increment achieved during previous captures to update the
  // offset.
  increase(wholegraph_increment);
}

/**
 * MLUGeneratorImpl class implementation
 */
MLUGeneratorImpl::MLUGeneratorImpl(at::DeviceIndex device_index)
    : c10::GeneratorImpl{
          at::Device(at::DeviceType::PrivateUse1, device_index),
          at::DispatchKeySet(c10::DispatchKey::PrivateUse1)} {
  torch_mlu::assertNotCapturing("Cannot construct a new MLUGeneratorImpl");
  state_ = c10::make_intrusive<MLUGeneratorState>();
}

MLUGeneratorImpl::MLUGeneratorImpl(
    at::DeviceIndex device_index,
    c10::intrusive_ptr<MLUGeneratorState> state)
    : c10::
          GeneratorImpl{at::Device(at::DeviceType::PrivateUse1, device_index), at::DispatchKeySet(c10::DispatchKey::PrivateUse1)},
      state_(std::move(state)) {}

/**
 * Sets the seed to be used by cnnlRandGenerator_t
 * Resets the philox_offset_per_thread_ to 0
 */
void MLUGeneratorImpl::set_current_seed(uint64_t seed) {
  torch_mlu::assertNotCapturing(
      "Cannot call MLUGeneratorImpl::set_current_seed");
  state_->seed_ = seed;
  state_->philox_offset_per_thread_ = 0;
}

/**
 * Sets the offset to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void MLUGeneratorImpl::set_offset(uint64_t offset) {
  torch_mlu::assertNotCapturing("Cannot call MLUGeneratorImpl::set_offset");
  // the set function checks if the offset is a multiple of 4.
  set_philox_offset_per_thread(offset);
  // no_reset_rnn_state_.clear();
}

/**
 * Gets the current offset of MLUGeneratorImpl.
 */
uint64_t MLUGeneratorImpl::get_offset() const {
  torch_mlu::assertNotCapturing("Cannot call MLUGeneratorImpl::get_offset");
  return state_->philox_offset_per_thread_;
}

/**
 * Gets the current seed of MLUGeneratorImpl.
 */
uint64_t MLUGeneratorImpl::current_seed() const {
  torch_mlu::assertNotCapturing("Cannot call MLUGeneratorImpl::current_seed");
  return state_->seed_;
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 */
uint64_t MLUGeneratorImpl::seed() {
  torch_mlu::assertNotCapturing("Cannot call MLUGeneratorImpl::seed");
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/**
 * Gets the current internal state of MLUGeneratorImpl. The internal
 * state is returned as a CPU byte tensor.
 */
c10::intrusive_ptr<c10::TensorImpl> MLUGeneratorImpl::get_state() const {
  // The RNG state comprises the seed, and an offset used for Philox.
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  auto state_tensor = at::detail::empty_cpu(
      {(int64_t)total_size},
      at::ScalarType::Byte,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt);
  auto rng_state = state_tensor.data_ptr<uint8_t>();
  auto current_seed = this->current_seed();
  auto offset = static_cast<int64_t>(
      this->philox_offset_per_thread()); // Note that old THCGeneratorState had
                                         // offset as std::atomic<int64_t>
  memcpy(rng_state, &current_seed, seed_size);
  memcpy(rng_state + seed_size, &offset, offset_size);

  return state_tensor.getIntrusivePtr();
}

/**
 * Sets the internal state of MLUGeneratorImpl. The new internal state
 * must be a strided CPU byte tensor and have appropriate size. See
 * comments of MLUGeneratorImpl::state for information about the layout
 * and size of the internal state.
 */
void MLUGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  torch_mlu::assertNotCapturing(
      "Please ensure to utilize the MLUGeneratorImpl::graphsafe_set_state method during capturing.");
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  at::detail::check_rng_state(new_state);

  bool no_philox_seed = false;
  auto new_state_size = new_state.numel();
  if (new_state_size == total_size - offset_size) {
    no_philox_seed = true;
  } else {
    TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");
  }

  uint64_t input_seed;
  auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&input_seed, new_rng_state, seed_size);
  this->set_current_seed(input_seed);
  int64_t philox_offset = 0;
  if (!no_philox_seed) {
    memcpy(&philox_offset, new_rng_state + seed_size, offset_size);
  }
  this->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}

/**
 * Sets the generator's current state to
 * This function allows switching between different registered states of
 * the generator.
 */
void MLUGeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<c10::GeneratorImpl>& gen) {
  c10::intrusive_ptr<MLUGeneratorImpl> mlu_gen =
      c10::dynamic_intrusive_pointer_cast<MLUGeneratorImpl>(gen);
  TORCH_CHECK(mlu_gen, "Expected a MLU Generator");
  state_ = mlu_gen->state_;
}

/**
 * Get the GeneratorImpl that point to current state_
 */
c10::intrusive_ptr<c10::GeneratorImpl> MLUGeneratorImpl::graphsafe_get_state()
    const {
  auto gen = c10::make_intrusive<MLUGeneratorImpl>(device().index(), state_);
  return gen;
}

/**
 * Sets the philox_offset_per_thread_ to be used by cnnlRandGenerator_t
 */
void MLUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  state_->philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of MLUGeneratorImpl.
 */
uint64_t MLUGeneratorImpl::philox_offset_per_thread() const {
  return state_->philox_offset_per_thread_;
}

/**
 * Registers this state to a MLU graph to manage within the graph.
 */
void MLUGeneratorImpl::register_graph(torch_mlu::MLUGraph* graph) {
  graph->register_generator_state(state_);
  state_->register_graph(graph);
}

/**
 * Unregisters a MLU graph from the RNG state.
 */
void MLUGeneratorImpl::unregister_graph(torch_mlu::MLUGraph* graph) {
  state_->unregister_graph(graph);
}

PhiloxMLUState MLUGeneratorImpl::philox_mlu_state(uint64_t increment) {
  if (torch_mlu::currentStreamCaptureStatus() !=
      torch_mlu::CaptureStatus::None) {
    uint32_t offset = state_->offset_intragraph_;
    state_->increase(increment);
    return PhiloxMLUState(
        state_->seed_extragraph_.data_ptr<int64_t>(),
        state_->offset_extragraph_.data_ptr<int64_t>(),
        offset);
  } else {
    uint64_t offset = state_->philox_offset_per_thread_;
    state_->increase(increment);
    return PhiloxMLUState(state_->seed_, offset);
  }
}

std::pair<uint64_t, uint64_t> MLUGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  torch_mlu::assertNotCapturing(
      "Refactor this op to use MLUGeneratorImpl::philox_mlu_state. Cannot call MLUGeneratorImpl::philox_engine_inputs");
  uint64_t offset = state_->philox_offset_per_thread_;
  state_->increase(increment);
  return std::make_pair(state_->seed_, offset);
}

/*
 * Gets the DeviceType of MLUGeneratorImpl.
 * Used for type checking during run time.
 */
DeviceType MLUGeneratorImpl::device_type() {
  return at::DeviceType::PrivateUse1;
}

/**
 * Public clone method implementation
 */
std::shared_ptr<MLUGeneratorImpl> MLUGeneratorImpl::clone() const {
  return std::shared_ptr<MLUGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 */
MLUGeneratorImpl* MLUGeneratorImpl::clone_impl() const {
  torch_mlu::assertNotCapturing("Cannot call MLUGeneratorImpl::clone_impl");
  auto gen = new MLUGeneratorImpl(this->device().index(), state_->clone());
  return gen;
}

// register MLU generator
at::Generator make_mlu_generator(c10::DeviceIndex device_index) {
  return at::make_generator<MLUGeneratorImpl>(device_index);
}

REGISTER_GENERATOR_PRIVATEUSE1(make_mlu_generator)

} // namespace torch_mlu
