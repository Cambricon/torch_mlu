import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.codegen.wrapper import WrapperCodeGen
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    register_backend_for_device,
)

from .codegen.wrapper import MluWrapperCodeGen


def init_backend_registration(self):
    if get_scheduling_for_device("cpu") is None:
        from torch._inductor.codegen.cpp import CppScheduling

        register_backend_for_device("cpu", CppScheduling, WrapperCodeGen)

    if get_scheduling_for_device("mlu") is None:
        from .codegen.triton import MluTritonScheduling

        register_backend_for_device("mlu", MluTritonScheduling, MluWrapperCodeGen)


torch._inductor.graph.GraphLowering.init_backend_registration = (
    init_backend_registration
)
