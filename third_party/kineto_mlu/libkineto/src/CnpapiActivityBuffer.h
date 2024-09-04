// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <stdlib.h>
#include <assert.h>
#include <cstdint>
#include <map>
#include <memory>
#include <sys/types.h>
#include <vector>

#include "ITraceActivity.h"

namespace KINETO_NAMESPACE {

class CnpapiActivityBuffer {
 public:
  explicit CnpapiActivityBuffer(size_t size) : size_(size) {
    buf_.reserve(size);
  }
  CnpapiActivityBuffer() = delete;
  CnpapiActivityBuffer& operator=(const CnpapiActivityBuffer&) = delete;
  CnpapiActivityBuffer(CnpapiActivityBuffer&&) = default;
  CnpapiActivityBuffer& operator=(CnpapiActivityBuffer&&) = default;

  size_t size() const {
    return size_;
  }

  void setSize(size_t size) {
    assert(size <= buf_.capacity());
    size_ = size;
  }

  uint64_t* data() {
    return buf_.data();
  }

 private:

  std::vector<uint64_t> buf_;
  size_t size_;

  std::vector<std::unique_ptr<const libkineto::ITraceActivity>> wrappers_;
};

using CnpapiActivityBufferMap =
    std::map<uint64_t*, std::unique_ptr<CnpapiActivityBuffer>>;

} // namespace KINETO_NAMESPACE
