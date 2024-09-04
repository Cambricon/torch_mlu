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

#include <memory>
#include <mutex>

#include "cnrt.h" // NOLINT
#include "cndev.h" // NOLINT

#include "aten/utils/exceptions.h"
#include "utils/Export.h"

namespace torch_mlu {

#define SINGLETON(CLASS)                   \
 public:                                   \
  CLASS(const CLASS&) = delete;            \
  CLASS& operator=(const CLASS&) = delete; \
  static CLASS& instance() {               \
    static CLASS instance;                 \
    return instance;                       \
  }                                        \
                                           \
 private:                                  \
  CLASS();                                 \
  ~CLASS()

// Humanity will defeat COVID-19 after all!

// A singleton class to hold common Catch stuff
class TORCH_MLU_API Global {
  SINGLETON(Global);

 public:
  cndevNameEnum_t getDeviceName() {
    return device_name_;
  }
  bool isUsingFloatingDevice() {
    return is_running_fp32_;
  }
  void setFP32RunningMode(bool run_fp32) {
    is_running_fp32_ = run_fp32;
  }

  // TF32 mode management
  bool allowCNNLTF32() const {
    return allow_tf32_cnnl_;
  }
  void setAllowCNNLTF32(bool b) {
    allow_tf32_cnnl_ = b;
  }
  bool allowMLUCustomTF32() const {
    return allow_tf32_custom_;
  }
  void setAllowMLUCustomTF32(bool b) {
    allow_tf32_custom_ = b;
  }

  // Fusion op management
  bool allowOpFusion() const {
    return enabled_fusion_;
  }
  void setAllowOpFusion(bool b) {
    enabled_fusion_ = b;
  }

 private:
  cndevNameEnum_t device_name_;
  bool is_running_fp32_;
  bool allow_tf32_cnnl_ =
      true; // vs `torch.backends.cudnn.allow_tf32` currently only affect conv
  bool allow_tf32_custom_ =
      false; // control wether to allow TF32 on the rest MLU ops
  bool enabled_fusion_ = true; // control wether torch.nn.LSTM use fusion op.
};

} // namespace torch_mlu
