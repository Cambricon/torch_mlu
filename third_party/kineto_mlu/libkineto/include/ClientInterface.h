// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

namespace libkineto {

class ClientInterface {
 public:
  virtual ~ClientInterface() {}
  virtual void init() = 0;
  virtual void prepare(bool, bool, bool, bool, bool) = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
};

} // namespace libkineto
