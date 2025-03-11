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

#include <iterator>
#include <string>
#include <map>
#include <cstdlib>
#include <utility>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "common.h"
#include <c10/util/Exception.h>

namespace {

const std::unordered_map<std::string, torch_mlu::OpPrecisionMode>
    DEFAULT_OP_PRECISION_MAP = {
        {"silu", torch_mlu::OpPrecisionMode::LOW},
        {"silu_backward", torch_mlu::OpPrecisionMode::LOW},
        {"sigmoid", torch_mlu::OpPrecisionMode::LOW},
        {"sigmoid_backward", torch_mlu::OpPrecisionMode::LOW},
        {"sqrt", torch_mlu::OpPrecisionMode::LOW},
        {"pow", torch_mlu::OpPrecisionMode::LOW},
        {"reduce", torch_mlu::OpPrecisionMode::LOW},
        {"foreach_unary", torch_mlu::OpPrecisionMode::LOW},
        {"normal", torch_mlu::OpPrecisionMode::LOW},
        {"div", torch_mlu::OpPrecisionMode::HIGH},
        {"adam", torch_mlu::OpPrecisionMode::LOW},
        {"custom_fused_adam", torch_mlu::OpPrecisionMode::LOW},
        {"custom_fused_l2_norm", torch_mlu::OpPrecisionMode::LOW},
        {"custom_fused_lamb", torch_mlu::OpPrecisionMode::LOW},
};

const std::vector<std::string> SUPPORTED_PRECISION_OP_LIST = []() {
  std::vector<std::string> res;
  std::transform(
      DEFAULT_OP_PRECISION_MAP.begin(),
      DEFAULT_OP_PRECISION_MAP.end(),
      std::back_inserter(res),
      [](const auto& pair) { return pair.first; });
  return res;
}();

const std::map<std::string, torch_mlu::OpPrecisionMode> OP_PRECISION_MODE_TABLE{
    {"low", torch_mlu::OpPrecisionMode::LOW},
    {"high", torch_mlu::OpPrecisionMode::HIGH},
};

std::string trimSpace(const std::string& str) {
  size_t first = str.find_first_not_of(" \t");
  if (first == std::string::npos)
    return "";
  size_t last = str.find_last_not_of(" \t");
  return str.substr(first, last - first + 1);
}

std::unordered_map<std::string, std::string> getOpPrecisionEnvMap() {
  std::unordered_map<std::string, std::string> result;
  // format: "silu: high, all_op: low"
  // "all_op", represent all ops in SUPPORTED_PRECISION_OP_LIST.
  const auto env = std::getenv("TORCH_OP_PRECISION_CONFIG");
  if (env) {
    std::string env_str = env;
    std::transform(
        env_str.begin(), env_str.end(), env_str.begin(), [](unsigned char c) {
          return std::tolower(c);
        });
    size_t start = 0, end;
    while ((end = env_str.find(',', start)) != std::string::npos) {
      auto sub_s = trimSpace(env_str.substr(start, end - start));
      size_t first = sub_s.find(':', 0);
      result[trimSpace(sub_s.substr(0, first))] =
          trimSpace(sub_s.substr(first + 1));
      start = end + 1;
    }
    // process the last substring
    auto sub_s = trimSpace(env_str.substr(start));
    size_t first = sub_s.find(':', 0);
    result[trimSpace(sub_s.substr(0, first))] =
        trimSpace(sub_s.substr(first + 1));
  }
  return result;
}

} // namespace

namespace torch_mlu {
const std::map<std::string, cndevNameEnum_t> device_name_table{
    {"MLU100", MLU100},
    {"MLU270", MLU270},
    {"MLU220_M2", MLU220_M2},
    {"MLU220_EDGE", MLU220_EDGE},
    {"MLU220_EVB", MLU220_EVB},
    {"MLU220_M2i", MLU220_M2i},
    {"MLU290", MLU290},
    {"MLU590", MLU590},
    {"MLU370", MLU370},
    {"MLU580", MLU580}};

Global::Global() {
  is_running_fp32_ = true;

  cndevCardInfo_t card_num_info;
  card_num_info.version = 5; // CNDEV use API version 5 as default
  card_num_info.number = 0;
  cndevCardName_t card_name;
  card_name.version = 5; // CNDEV use API version 5 as default
  // Auto get device name from cndev
  TORCH_CNDEV_CHECK(cndevInit(0));
  TORCH_CNDEV_CHECK(cndevGetDeviceCount(&card_num_info));
  TORCH_MLU_CHECK(
      card_num_info.number != 0,
      "Cannot find any visiale MLU decvice to specify device name");
  TORCH_CNDEV_CHECK(cndevGetCardName(&card_name, 0));
  device_name_ = card_name.id;

  // default map as initial value
  op_precision_map_ = DEFAULT_OP_PRECISION_MAP;
  // read from environ-variable
  // [op_name -> precision_mode]
  std::unordered_map<std::string, std::string> op_precision_env_map =
      getOpPrecisionEnvMap();
  if (auto s = op_precision_env_map.find("all_op"); // NOSONAR
      s != op_precision_env_map.end()) {
    // need set precision mode for "all_op" first!
    this->setPrecisionMode(s->second, "all_op");
    op_precision_env_map.erase(s);
  }
  for (const std::pair<std::string, std::string>& word : op_precision_env_map) {
    this->setPrecisionMode(word.second, word.first);
  }

  return;
}

Global::~Global() {}

std::vector<std::string> Global::getPrecisionSupportedOpList() const {
  return SUPPORTED_PRECISION_OP_LIST;
}

OpPrecisionMode Global::getPrecisionMode(const std::string& op) const {
  std::string op_str = op;
  std::transform(
      op_str.begin(), op_str.end(), op_str.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
  if (auto s = op_precision_map_.find(op_str); // NOSONAR
      s == op_precision_map_.end()) {
    return torch_mlu::OpPrecisionMode::HIGH;
  }
  return op_precision_map_.at(op_str);
}

void Global::setPrecisionMode(const std::string& mode, const std::string& op) {
  if (op.empty())
    return;
  std::string op_str = op;
  std::transform(
      op_str.begin(), op_str.end(), op_str.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
  if (auto s = DEFAULT_OP_PRECISION_MAP.find(op_str);
      s == DEFAULT_OP_PRECISION_MAP.end() && op_str != "all_op") {
    TORCH_WARN(
        "The operator '",
        op,
        "' is not in supported precision op list; ",
        "the current setPrecisionMode call has no effect.");
    return;
  }

  std::string mode_str = mode;
  std::transform(
      mode_str.begin(), mode_str.end(), mode_str.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
  if (auto s = OP_PRECISION_MODE_TABLE.find(mode_str);
      s == OP_PRECISION_MODE_TABLE.end()) {
    TORCH_WARN(
        "The mode '",
        mode,
        "' is not one of 'low' or 'high'; ",
        "the current setPrecisionMode call on operator '",
        op,
        "' has no effect.");
    return;
  }

  // processing case for "all_op"
  if (op_str == "all_op")
    for (auto& p : SUPPORTED_PRECISION_OP_LIST)
      op_precision_map_[p] = OP_PRECISION_MODE_TABLE.at(mode_str);
  else
    op_precision_map_[op_str] = OP_PRECISION_MODE_TABLE.at(mode_str);
}

} // namespace torch_mlu
