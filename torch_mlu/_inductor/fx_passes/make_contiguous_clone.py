import torch
import torch.fx

from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional
from torch_mlu._inductor.lowering import mlu_lowerings

aten = torch.ops.aten


def is_contiguous(node: torch.fx.Node):
    val = node.meta.get("val")
    if isinstance(val, torch._subclasses.fake_tensor.FakeTensor):
        return val.is_contiguous()
    else:
        return True


def make_continuous_clone(graph: torch.fx.Graph):
    lowerings_list = mlu_lowerings
    for node in reversed(graph.nodes):
        contiguous_user_list = []
        if not is_contiguous(node):
            for user in node.users:
                if (
                    user.target in lowerings_list
                    and is_contiguous(user)
                    and user.target
                    not in [
                        aten.clone,
                        aten.clone.default,
                        triton_kernel_wrapper_functional,
                    ]
                ):
                    contiguous_user_list.append(user)

        if contiguous_user_list:
            with graph.inserting_after(node):
                clone_node = graph.call_function(
                    aten.clone.default,
                    (node,),
                    {"memory_format": torch.contiguous_format},
                )
            for user in contiguous_user_list:
                user.replace_input_with(node, clone_node)
