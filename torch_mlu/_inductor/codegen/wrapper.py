import dataclasses
from typing import Optional
from torch._inductor import codecache, config
from torch._inductor.codegen.common import IndentedBuffer, PythonPrinter
from torch._inductor.codegen.wrapper import WrapperLine, WrapperCodeGen
from torch._inductor.virtualized import V


@dataclasses.dataclass
class EnterMluDeviceContextManagerLine(WrapperLine):
    device_idx: int
    last_seen_device_guard_index: Optional[int]

    def codegen(self, code: IndentedBuffer) -> None:
        if V.graph.cpp_wrapper:
            # Modified by Cambricon: comment below code block.
            # Reason: we does not supoort cpp_wrapper.
            # code.writeline("\n")
            # if V.graph.aot_mode:
            #     # In AOT mode, we have a stream provided as a param. A stream is
            #     # associated with a device, so we never expect the device to change.
            #     # CUDAStreamGuard sets the stream and the device.
            #     if self.last_seen_device_guard_index is None:
            #         if config.abi_compatible:
            #             code.writeline(
            #                 "AOTICudaStreamGuard stream_guard(stream, this->device_idx_);"
            #             )
            #         else:
            #             code.writeline(
            #                 "at::cuda::CUDAStreamGuard stream_guard("
            #                 + "at::cuda::getStreamFromExternal(stream, this->device_idx_));"
            #             )
            #     else:
            #         assert (
            #             self.last_seen_device_guard_index == self.device_idx
            #         ), "AOTInductor only supports running on one CUDA device"
            # else:
            #     if self.last_seen_device_guard_index is None:
            #         code.writeline(
            #             f"AOTICudaGuard device_guard({self.device_idx});"
            #             if config.abi_compatible
            #             else f"at::cuda::CUDAGuard device_guard({self.device_idx});"
            #         )
            #     else:
            #         code.writeline(f"device_guard.set_index({self.device_idx});")
            raise NotImplementedError(f"Cpp_wrapper mode is not supported for now!")
        else:
            # Note _DeviceGuard has less overhead than device, but only accepts
            # integers
            code.writeline(f"with {V.graph.device_ops.device_guard(self.device_idx)}:")
            code.do_indent()
            code.writeline(V.graph.device_ops.set_device(self.device_idx))


class ExitMluDeviceContextManagerLine(WrapperLine):
    def codegen(self, code: IndentedBuffer) -> None:
        if not V.graph.cpp_wrapper:
            code.do_unindent()


class MluWrapperCodeGen(WrapperCodeGen):
    """
    Generate outer wrapper in Python that calls the kernels.
    """

    def __init__(self):
        super().__init__()

    def write_header(self) -> None:
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import torch_mlu
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.codegen.memory_planning import _align as align

                from torch import device, empty_strided
                from {codecache.__name__} import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels
                from torch._inductor.codegen.multi_kernel import MultiKernelCall

                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
                async_compile = AsyncCompile()

            """
        )

    def codegen_device_guard_enter(self, device_idx: int) -> None:
        self.writeline(
            EnterMluDeviceContextManagerLine(
                device_idx, self.last_seen_device_guard_index
            )
        )
        self.last_seen_device_guard_index = device_idx

    def codegen_device_guard_exit(self) -> None:
        self.writeline(ExitMluDeviceContextManagerLine())
