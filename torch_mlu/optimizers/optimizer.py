import warnings

import torch


def _mlu_graph_capture_health_check(self) -> None:
    # Note [torch.compile x capturable] in pytorch/torch/optim/optimizer.py
    if not torch._utils.is_compiling() and torch.mlu.is_available():
        capturing = torch.mlu.is_current_stream_capturing()

        if capturing and not all(group["capturable"] for group in self.param_groups):
            raise RuntimeError(
                "Attempting MLU graph capture of step() for an instance of "
                + self.__class__.__name__
                + " but param_groups' capturable is False."
            )

        if (
            (not getattr(self, "_warned_capturable_if_run_uncaptured", False))
            and all(group["capturable"] for group in self.param_groups)
            and (not capturing)
        ):
            warnings.warn(
                "This instance was constructed with capturable=True or some of all the param_groups came with capturable=True, "
                "but step() is running without MLU graph capture. If you never intend to graph-capture this "
                "instance, capturable=True can impair performance, and you should set capturable=False."
            )
            self._warned_capturable_if_run_uncaptured = True


def apply_optimizer_patch():
    torch.optim.Optimizer._cuda_graph_capture_health_check.__code__ = (
        _mlu_graph_capture_health_check.__code__
    )
