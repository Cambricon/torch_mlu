from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_device_op_overrides,
)


class MLUDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        # return f"from torch_mlu._MLUC import _mlu_getCurrentStream as {name}"
        return f"from torch.mlu import current_stream  as {name}"

    def set_device(self, device_idx):
        return f"torch.mlu.set_device({device_idx})"

    def synchronize(self):
        return "torch.mlu.synchronize()"

    def device_guard(self, device_idx):
        return f"torch.mlu._DeviceGuard({device_idx})"


register_device_op_overrides("mlu", MLUDeviceOpOverrides())
