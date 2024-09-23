import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config
from torch._inductor.virtualized import V
from torch._inductor.scheduler import (
    log,
    FusedSchedulerNode,
    NopKernelSchedulerNode,
    SchedulerNode,
)


@dynamo_timed
def codegen(self):
    for node in self.nodes:
        try:
            log.debug(
                "Generating code for node %s with estimated runtime %f",
                node.get_name(),
                node.get_estimated_runtime(),
            )
        except Exception as e:
            log.debug(
                "Generating code for node %s with estimated runtime 0.0",
                node.get_name(),
            )

        self.enter_context(node)

        if not isinstance(node, NopKernelSchedulerNode):
            device = node.get_device()
            if device != self.current_device or node.is_extern() or node.is_template():
                self.flush()
            if device != self.current_device:
                if device.type == "mlu":
                    if self.current_device and self.current_device.type == "mlu":
                        V.graph.wrapper_code.codegen_device_guard_exit()
                    assert device.index is not None, "device should have an index"
                    V.graph.wrapper_code.codegen_device_guard_enter(device.index)
                elif self.current_device and self.current_device.type == "mlu":
                    V.graph.wrapper_code.codegen_device_guard_exit()
                self.current_device = device

        self.buffer_names_to_free.update(node.last_usage)

        if node.is_template():
            node, *epilogue = node.get_nodes()
            self.get_backend(device).codegen_template(node, epilogue)  # type: ignore[possibly-undefined]
        elif node.is_extern():
            self.codegen_extern_call(node)
        elif node.is_foreach():
            self.get_backend(device).codegen_foreach(node)  # type: ignore[possibly-undefined]
        elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
            self.get_backend(device).codegen_nodes(node.get_nodes())  # type: ignore[possibly-undefined]
        else:
            assert isinstance(node, NopKernelSchedulerNode)
            node.allocate()

        if config.debug_check_inf_and_nan:
            V.graph.wrapper_code.generate_inf_and_nan_checker(node)

        if config.triton.debug_sync_kernel:
            self.get_backend(device).codegen_sync()  # type: ignore[possibly-undefined]

        self.available_buffer_names.update(node.get_names())

        if not isinstance(node, NopKernelSchedulerNode):
            device = node.get_device()
            if self.get_backend(device).ready_to_flush():
                self.flush()

    if self.current_device and self.current_device.type == "mlu":
        # exit the outermost CUDA device guard. this is
        # important for nested indentation codegen-ing.
        V.graph.wrapper_code.codegen_device_guard_exit()

    self.flush()


torch._inductor.scheduler.Scheduler.codegen = codegen
