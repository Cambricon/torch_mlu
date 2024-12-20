#include "aten/utils/sleep.h"
#include "framework/core/MLUStream.h"

namespace torch_mlu {

void sleep(int64_t cycles) {
  auto stream = getCurMLUStream();
  sleep_mlu(cycles, stream);
}
} // namespace torch_mlu
