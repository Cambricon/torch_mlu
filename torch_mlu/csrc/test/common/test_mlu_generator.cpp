#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <thread>
#include <limits>
#include <random>

#include "framework/generator/generator_impl.h"
#include "utils/assert_tensor.h"

namespace torch_mlu {

TEST(MLUGeneratorImpl, TestGeneratorDynamicCast) {
  // Test Description: Check dynamic cast for MLU
  auto foo = torch_mlu::createMLUGenerator();
  auto result = at::check_generator<MLUGeneratorImpl>(foo);
  ASSERT_EQ(
      typeid(torch_mlu::MLUGeneratorImpl*).hash_code(),
      typeid(result).hash_code());
}

TEST(MLUGeneratorImpl, TestDefaultGenerator) {
  // Test Description:
  // Check if default generator is created only once
  // address of generator should be same in all calls
  auto foo = torch_mlu::getDefaultMLUGenerator();
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto bar = torch_mlu::getDefaultMLUGenerator();
  ASSERT_EQ(foo, bar);
}

TEST(MLUGeneratorImpl, TestClonImpl) {
  // Test Description:
  // Check cloning of new generators.
  // Note that we don't allow cloning of other
  // generator states into default generators.
  at::Tensor input =
      at::ones({2, 4}).to(at::Device(at::Device::Type::PrivateUse1));
  auto gen1 = torch_mlu::createMLUGenerator();
  auto gen2 = torch_mlu::createMLUGenerator();
  gen2 = gen1.clone();
  auto output1 = at::native::bernoulli(input, 0.5 /*p*/, gen1);
  auto output2 = at::native::bernoulli(input, 0.5 /*p*/, gen2);
  assertTensorsEqual(output1.cpu(), output2.cpu(), 0.0, true, false, false);
}

TEST(MLUGeneratorImpl, TestClone) {
  // Test Description:
  // Check cloning of new generators.
  // Note that we don't allow cloning of other
  // generator states into default generators.
  at::Tensor input =
      at::ones({2, 4}).to(at::Device(at::Device::Type::PrivateUse1));
  auto gen1 = torch_mlu::createMLUGenerator();
  auto mlu_gen1 = at::check_generator<MLUGeneratorImpl>(gen1);
  auto gen2 = torch_mlu::createMLUGenerator();
  auto mlu_gen2 = at::check_generator<MLUGeneratorImpl>(gen2);
  mlu_gen2 = mlu_gen1->clone().get();
  auto output1 = at::native::bernoulli(input, 0.5 /*p*/, gen1);
  auto output2 = at::native::bernoulli(input, 0.5 /*p*/, gen2);
  assertTensorsEqual(output1.cpu(), output2.cpu(), 0.0, true, false, false);
}

TEST(MLUGeneratorImpl, TestCapturePrologue) {
  auto gen = torch_mlu::createMLUGenerator();
  auto mlu_gen = at::check_generator<MLUGeneratorImpl>(gen);
  try {
    mlu_gen->capture_prologue(0 /*seed_extragraph*/, 0 /*offset_extragraph*/);
  } catch (const c10::Error& e) {
    ASSERT_EQ("MLUGeneratorImpl::capture_prologue() not implement", e.msg());
  }
}

TEST(MLUGeneratorImpl, TestCaptureEpilogue) {
  auto gen = torch_mlu::createMLUGenerator();
  auto mlu_gen = at::check_generator<MLUGeneratorImpl>(gen);
  try {
    mlu_gen->capture_epilogue();
  } catch (const c10::Error& e) {
    ASSERT_EQ("MLUGeneratorImpl::capture_epilogue() not implement", e.msg());
  }
}

TEST(MLUGeneratorImpl, TestGetSetCurrentSeed) {
  // Test Description:
  // Test current seed getter and setter
  // See Note [Acquire lock when using random generators]
  auto foo = torch_mlu::getDefaultMLUGenerator();
  std::lock_guard<std::mutex> lock(foo.mutex());
  foo.set_current_seed(123);
  auto current_seed = foo.current_seed();
  ASSERT_EQ(current_seed, 123);
}

TEST(MLUGeneratorImpl, TestPhiloxEngineInputs) {
  auto gen = torch_mlu::createMLUGenerator();
  auto mlu_gen = at::check_generator<MLUGeneratorImpl>(gen);
  uint64_t offset0 = mlu_gen->philox_offset_per_thread();
  uint64_t seed0 = mlu_gen->current_seed();
  mlu_gen->philox_engine_inputs(4 /*increment*/);
  uint64_t offset1 = mlu_gen->philox_offset_per_thread();
  uint64_t seed1 = mlu_gen->current_seed();
  ASSERT_EQ(seed0, seed1);
  ASSERT_EQ(offset0 + 4, offset1);
}

} // namespace torch_mlu
