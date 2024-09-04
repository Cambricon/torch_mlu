// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "GenericTraceActivity.h"
#include "output_base.h"

namespace libkineto {
  using namespace KINETO_NAMESPACE;
  void GenericTraceActivity::log(ActivityLogger& logger) const {
    logger.handleGenericActivity(*this);
  }
} // namespace libkineto
