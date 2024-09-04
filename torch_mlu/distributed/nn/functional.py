import torch
import torch.distributed as dist
from torch.autograd import Function

# The two imports below are not always available depending on the
# USE_DISTRIBUTED compile flag. Make sure they raise import error
# if we're trying to use them.
from torch.distributed import group, ReduceOp


class _AllGatherBase(Function):
    @staticmethod
    def forward(ctx, output_tensor, input_tensor, group):
        ctx.group = group
        dist._all_gather_base(output_tensor, input_tensor.contiguous(), group=group)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        if dist.get_backend(group=ctx.group) is dist.Backend.CNCL:
            world_size = dist.get_world_size(group=ctx.group)
            out_size = list(grad_output.size())
            if out_size[0] % world_size != 0:
                raise RuntimeError(
                    f"Tensor with dimensions: {out_size} does "
                    f"not have first dimension divisible by world_size: {world_size}"
                )
            out_size[0] = out_size[0] // dist.get_world_size(group=ctx.group)
            gx = torch.empty(
                out_size, device=grad_output.device, dtype=grad_output.dtype
            )
            dist._reduce_scatter_base(gx, grad_output, ReduceOp.SUM, ctx.group)
        else:
            raise RuntimeError("Backend not supported!")
        return (None, gx, None)


def apply_functional_patch():
    dist.nn.functional._AllGatherBase = _AllGatherBase
