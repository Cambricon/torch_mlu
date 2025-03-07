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

#include <unordered_set>
#include <c10/core/GeneratorImpl.h>
#include <ATen/core/Generator.h>
#include "framework/core/device.h"
#include "aten/cnnl/cnnlHandle.h"

namespace torch_mlu {

struct MLUGraph;

struct PhiloxMLUState {
  PhiloxMLUState() = default;
  // Called if graph capture is not underway
  PhiloxMLUState(uint64_t seed, uint64_t offset) {
    seed_.val = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxMLUState(
      int64_t* seed,
      int64_t* offset_extragraph,
      uint32_t offset_intragraph) {
    seed_.ptr = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  Payload seed_;
  Payload offset_;
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

struct TORCH_MLU_API MLUGeneratorState : public c10::intrusive_ptr_target {
  uint64_t seed_;
  uint64_t philox_offset_per_thread_;
  uint32_t offset_intragraph_;
  bool capturing_{};
  std::unordered_set<torch_mlu::MLUGraph*> registered_graphs_;
  at::TensorBase seed_extragraph_{};
  at::TensorBase offset_extragraph_{};

  MLUGeneratorState(
      uint64_t seed = c10::default_rng_seed_val,
      uint64_t philox_offset_per_thread = 0,
      uint32_t offset_intragraph = 0)
      : seed_(seed),
        philox_offset_per_thread_(philox_offset_per_thread),
        offset_intragraph_(offset_intragraph) {}

  void increase(uint64_t increment);

  void register_graph(torch_mlu::MLUGraph* graph);
  void unregister_graph(torch_mlu::MLUGraph* graph);

  void capture_prologue();
  // capture_epilogue returns the wholegraph_increment
  uint64_t capture_epilogue();
  void replay_prologue(uint64_t wholegraph_increment);
  c10::intrusive_ptr<MLUGeneratorState> clone();
};

struct TORCH_MLU_API MLUGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  MLUGeneratorImpl(at::DeviceIndex device_index = -1);
  MLUGeneratorImpl(
      at::DeviceIndex device_index,
      c10::intrusive_ptr<MLUGeneratorState> state_);
  ~MLUGeneratorImpl() override = default;

  // MLUGeneratorImpl methods
  std::shared_ptr<MLUGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  void graphsafe_set_state(
      const c10::intrusive_ptr<c10::GeneratorImpl>& state) override;
  c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override;

  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread() const;

  void register_graph(torch_mlu::MLUGraph* graph);
  void unregister_graph(torch_mlu::MLUGraph* graph);

  PhiloxMLUState philox_mlu_state(uint64_t increment);

  // Temporarily accommodates call sites that use philox_engine_inputs.
  // Allows incremental refactor of call sites to use philox_cuda_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);

  static at::DeviceType device_type();

 private:
  MLUGeneratorImpl* clone_impl() const override;
  c10::intrusive_ptr<MLUGeneratorState> state_;
};

TORCH_MLU_API const at::Generator& getDefaultMLUGenerator(
    at::DeviceIndex device_index = -1);
TORCH_MLU_API at::Generator createMLUGenerator(
    at::DeviceIndex device_index = -1);

} // namespace torch_mlu
