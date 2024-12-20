#pragma once
#include <cstdint>
#include "utils/Export.h"
#include "cnrt.h"

namespace torch_mlu {

// sleep for gievn cycles
TORCH_MLU_API void sleep(int64_t cycles);

void sleep_mlu(int64_t cycles, cnrtQueue_t queue);

} // namespace torch_mlu
