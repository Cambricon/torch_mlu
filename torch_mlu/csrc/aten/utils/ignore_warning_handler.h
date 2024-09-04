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
#include <c10/util/Exception.h>

class IgnoreWarningHandler : public c10::WarningHandler {
 public:
  void process(const c10::Warning& warning) override {
    ;
  }
};

static c10::WarningHandler* getIgnoreHandler() {
  static IgnoreWarningHandler handler_ = IgnoreWarningHandler();
  return &handler_;
};

// use to ignore the warning info when overriding operator for CPU-implement
#define WITH_IGNORE_WARNING_OVERRIDE_OPERATOR(enable, KERNEL_REGISTER) \
  int enter_warning() {                                                \
    if (enable) {                                                      \
      c10::WarningUtils::set_warning_handler(getIgnoreHandler());      \
    }                                                                  \
    return 1;                                                          \
  }                                                                    \
  static int _temp_enter_warning = enter_warning();                    \
  KERNEL_REGISTER                                                      \
  int exit_warning() {                                                 \
    if (enable) {                                                      \
      c10::WarningUtils::set_warning_handler(nullptr);                 \
    }                                                                  \
    return 1;                                                          \
  }                                                                    \
  static int _temp_exit_warning = exit_warning();
