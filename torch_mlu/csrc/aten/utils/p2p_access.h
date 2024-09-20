#include <ATen/ATen.h>
#include "utils/Export.h"
#include <cstdint>

namespace torch_mlu {

void init_p2p_access_cache(int64_t num_devices);

TORCH_MLU_API bool get_p2p_access(int source_dev, int dest_dev);

} // namespace torch_mlu