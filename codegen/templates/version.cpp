/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
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

// ${generated_comment}

#include "utils/version.h"

namespace torch_mlu {

${version_info}

struct MLULibVer {
  explicit MLULibVer(const std::string& version) : version_(version) {
    parse(version);
  }

  MLULibVer(int x, int y, int z)
      : x_(x),
        y_(y),
        z_(z),
        version_(
            std::to_string(x) + "." + std::to_string(y) + "." +
            std::to_string(z)) {}

  std::string str() const {
    return version_;
  }

  // parsing the string read from build.property
  void parse(const std::string& str) {
    // The format of version will be x.y,z-a, e.g.: 1.3.0-1, 1.2.0-torch1.9
    size_t ver_count = std::count(str.begin(), str.end(), '.');
    if (ver_count >= 2) {
      std::sscanf(str.c_str(), "%d.%d.%d", &x_, &y_, &z_);
    } else {
      CNLOG(ERROR) << "Unknown library version string :" << str;
    }
  }

  int x_ = 0; // major
  int y_ = 0; // minor
  int z_ = 0; // bugfix
  std::string version_;
};

std::string getDriverVersion() {
  CNdev mlu_dev;
  const int stub_dev_ordinal = 0;
  std::string driver_str;
  if (cnDeviceGet(&mlu_dev, stub_dev_ordinal)) {
    CNLOG(WARNING) << "Failed to get MLU device!!";
  }
  int major = 0, minor = 0, patch = 0;
  CNresult result = cnGetDriverVersion(&major, &minor, &patch);

  if (result) {
    CNLOG(WARNING) << "Failed to get driver version!!";
  }

  driver_str = std::to_string(major) + "." + std::to_string(minor) + "." +
      std::to_string(patch) + "-1";
  return driver_str;
}

// get torch_mlu version
std::string getVersion() {
  return MLULibVer(torch_mlu).str();
}

// check the versions of compiled and runtime
void checkRequirements() {
  using GetLibFunc = std::function<void(int*, int*, int*)>;
  auto get_lib_ver = [](const GetLibFunc& func) {
    int x, y, z;
    func(&x, &y, &z);
    MLULibVer lib_ver(x, y, z);
    return lib_ver;
  };

  // xxx_runtime is the current version of neuware,
  // xxx_required is the required version of neuware
  auto cnnl_runtime = get_lib_ver(cnnlGetLibVersion);
#ifdef USE_CNCL
  auto cncl_runtime = get_lib_ver(cnclGetLibVersion);
  auto cncl_required = MLULibVer(cncl_required_str);
#endif
  auto driver_runtime = MLULibVer(getDriverVersion());
  auto cnnl_required = MLULibVer(cnnl_required_str);
  auto driver_required = MLULibVer(driver_required_str);

  auto version_check = [](const std::string& lib_name,
                          MLULibVer runtime,
                          MLULibVer required) {
    if (runtime.x_ < required.x_ ||
        (runtime.x_ == required.x_ && runtime.y_ < required.y_) ||
        (runtime.x_ == required.x_ && runtime.y_ == required.y_ &&
         runtime.z_ < required.z_)) {
      CNLOG(WARNING)
          << "Cambricon NEUWARE minimum version requirements not met! Require "
          << lib_name << " minimum verion is " << required.str()
          << ", but current version is " << runtime.str();
    }
  };

  version_check("CNNL", cnnl_runtime, cnnl_required);
#ifdef USE_CNCL
  version_check("CNCL", cncl_runtime, cncl_required);
#endif
  version_check("DRIVER", driver_runtime, driver_required);
}

} // namespace torch_mlu
