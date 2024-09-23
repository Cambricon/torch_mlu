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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <mutex>
#include <thread>
#include "utils/cnlog.h"

namespace torch_mlu {

int64_t MinCNLogLevelFromEnv() {
  const char* th_env_var_val = getenv("TORCH_MIN_CNLOG_LEVEL");
  return LogLevelStrToInt(th_env_var_val);
}

int64_t LogLevelStrToInt(const char* th_env_var_val) {
  // Set default level=1(WARNING)
  if (th_env_var_val == nullptr) {
    return 1;
  }
  std::string min_log_level(th_env_var_val);
  std::istringstream ss(min_log_level);
  int64_t level = 1;
  if (!(ss >> level)) {
    // We allow value of TORCH_MIN_CNLOG_LEVEL to be valid string,
    // like export TORCH_MIN_CNLOG_LEVEL=INFO
    for (size_t i = 0; i < LOG_LEVEL.size(); ++i) {
      if (min_log_level == LOG_LEVEL[i]) {
        level = i - 1;
        break;
      }
      // Set default level=1(WARNING)
      level = 1;
    }
  }
  return level;
}

int64_t CNLogMessage::MinCNLogLevel() {
  static const int64_t min_vlog_level = MinCNLogLevelFromEnv();
  return min_vlog_level;
}

CNLogMessage::CNLogMessage(
    const char* file,
    int line,
    const char* func,
    int severity)
    : severity_(severity) {
  stream_ << "[" << LOG_LEVEL[severity + 1] << "]"
          << "[" << file << ":" << line << "]"
          << "[" << func << "]"
          << "[process:" << getpid() << "]"
          << "[thread:" << std::this_thread::get_id() << "]"
          << ": ";
}

CNLogMessage::~CNLogMessage() {
  stream_ << "\n";
  std::cerr << stream_.str();
  // Keeping same with the glog/caffe2 log default behavior:
  // if the severity is above INFO, flush the stream so that
  // the output appears immediately on std::cerr.
  if (severity_ > INFO) {
    std::cerr << std::flush;
  }
  if (severity_ == FATAL) {
    abort();
  }
}

} // namespace torch_mlu
