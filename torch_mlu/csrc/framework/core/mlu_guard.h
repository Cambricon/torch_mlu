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

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include "framework/core/guard_impl.h"

namespace torch_mlu {
namespace mlu {

// This code is kind of boilerplatey.

/// It is a variant of DeviceGuard specialized for MLU. Integer is accepted
/// and would be interprated as MLU DeviceIndex
struct TORCH_MLU_API MLUGuard {
  MLUGuard() = delete;

  /// Set the current MLU device to the passed device index.
  explicit MLUGuard(at::DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current MLU device to the passed device.  Errors occur when the
  /// passed device is not a MLU device.
  explicit MLUGuard(at::Device device) : guard_(device) {}

  // Copy is not allowed
  MLUGuard(const MLUGuard&) = delete;
  MLUGuard& operator=(const MLUGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  MLUGuard(MLUGuard&& other) = delete;
  MLUGuard& operator=(MLUGuard&& other) = delete;

  /// Sets the MLU device to the given device.  Errors if the given device
  /// is not a MLU device.
  void set_device(at::Device device) {
    guard_.set_device(device);
  }

  /// Sets the MLU device to the given device.  Errors if the given device
  /// is not a MLU device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(at::Device device) {
    guard_.reset_device(device);
  }

  /// Sets the MLU device to the given device index.
  void set_index(at::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  at::Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  at::Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<MLUGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for MLU.  See
/// MLUGuard for when you can use this.
struct OptionalMLUGuard {
  /// Create an uninitialized OptionalMLUGuard.
  explicit OptionalMLUGuard() : guard_() {}

  /// Set the current MLU device to the passed Device, if it is not nullopt.
  explicit OptionalMLUGuard(std::optional<at::Device> device_opt)
      : guard_(device_opt) {}

  /// Set the current MLU device to the passed device index, if it is not
  /// nullopt
  explicit OptionalMLUGuard(std::optional<at::DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalMLUGuard(const OptionalMLUGuard&) = delete;
  OptionalMLUGuard& operator=(const OptionalMLUGuard&) = delete;

  OptionalMLUGuard(OptionalMLUGuard&& other) = delete;

  OptionalMLUGuard& operator=(OptionalMLUGuard&& other) = delete;

  /// Sets the MLU device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a MLU
  /// device.
  void set_device(at::Device device) {
    guard_.set_device(device);
  }

  /// Sets the MLU device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a MLU device.
  void reset_device(at::Device device) {
    guard_.reset_device(device);
  }

  /// Sets the MLU device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(at::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  std::optional<at::Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  std::optional<at::Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original MLU device, resetting this guard to uninitialized
  /// state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<MLUGuardImpl> guard_;
};

} // namespace mlu
} // namespace torch_mlu
