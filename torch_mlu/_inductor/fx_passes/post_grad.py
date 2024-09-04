from torch._inductor.fx_passes import post_grad


def should_prefer_unfused_addmm(match):
    # decomposing addmm into mm and add would result in degraded performance on MLU.
    return False


post_grad.should_prefer_unfused_addmm.__code__ = should_prefer_unfused_addmm.__code__
