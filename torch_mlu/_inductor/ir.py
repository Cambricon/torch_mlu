from typing import Iterable, Optional

import torch
from torch._inductor.ir import (
    IRNode,
    TemplateBuffer,
    mark_node_as_mutating,
)


class TritonTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        debug_extra=None,
        mutated_inputs: Optional[Iterable[IRNode]] = None,
    ):
        """
        NOTE:[TritonTemplates with multiple outputs]
        We want the ability for TritonTemplates to output multiple tensors. Triton
        kernels have no notion of outputs and this is done by creating tensors that
        are then mutated by the kernel. Currenlty our STORE_OUTPUT codegen doesn't
        support creating multinode outputs for triton templates.
        We work around this by creating an extra input buffer during the lowering
        and we mark them as mutated inputs.
        """
        super().__init__(layout, inputs, make_kernel_render)
        self.debug_extra = debug_extra
        self.mutated_inputs = mutated_inputs
        if mutated_inputs is not None:
            mark_node_as_mutating(self, *mutated_inputs)


torch._inductor.ir.TritonTemplateBuffer = TritonTemplateBuffer
