import warnings
warnings.warn("`torch_mlu.utils.model_transfer` is deprecated. "
              "Please use `torch_mlu.utils.gpu_migration` instead.",
              FutureWarning)

# for backward compatibility
from torch_mlu.utils.gpu_migration import migration as transfer
