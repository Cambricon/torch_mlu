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

#include <ATen/Tensor.h>
#include <ATen/core/List.h>
#include "utils/Export.h"

const int FATAL = 3;
const int ERROR = 2;
const int WARNING = 1;
const int INFO = 0;

// DEBUG=-1, INFO=0, WARNING=1, ERROR=2, FATAL=3
const std::vector<std::string> LOG_LEVEL =
    {"DEBUG", "INFO", "WARNING", "ERROR", "FATAL"};

namespace torch_mlu {
int64_t MinCNLogLevelFromEnv();
int64_t MinCNLogLevel();
int64_t LogLevelStrToInt(const char* log_level_ptr);

class TORCH_MLU_API CNLogMessage {
 public:
  CNLogMessage(const char* file, int line, const char* func, int severity);
  ~CNLogMessage();

  std::stringstream& stream() {
    return stream_;
  }

  static int64_t MinCNLogLevel();

 private:
  std::stringstream stream_;
  int severity_;
};

} // namespace torch_mlu

#define CNLOG_IS_ON(lvl) ((lvl) >= torch_mlu::CNLogMessage::MinCNLogLevel())

#define CNLOG(lvl)      \
  if (CNLOG_IS_ON(lvl)) \
  torch_mlu::CNLogMessage(__FILE__, __LINE__, __FUNCTION__, lvl).stream()
