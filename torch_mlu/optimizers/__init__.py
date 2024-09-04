from .fused_adam import FusedAdam
from .fused_lamb import FusedLAMB, FusedLAMBAMP
from .fused_sgd import FusedSGD

from .optimizer import apply_optimizer_patch


def apply_optim_patch():
    apply_optimizer_patch()
