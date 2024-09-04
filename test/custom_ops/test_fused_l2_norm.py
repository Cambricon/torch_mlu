from itertools import product
import logging
import sys
import os
import unittest

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import (
    testinfo,
    TEST_BFLOAT16,
    TestCase,
)  # pylint: disable=C0413,C0411


class TestFusedL2Norm(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFusedL2Norm, self).__init__(*args, **kwargs)
        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int).to(
            torch.device("mlu")
        )

    def fused_l2norm_dtype(self, op, test_type, shape, per_tensor=False, len=2):
        tensors_cpu = []
        tensors_mlu = []
        torch.manual_seed(9876)

        for _ in range(len):
            tensor = torch.randn(shape, dtype=torch.float)
            tensors_cpu.append(tensor)
            tensors_mlu.append(tensor.to("mlu").to(test_type))

        fused_out1, fused_out2 = op(self._dummy_overflow_buf, tensors_mlu, per_tensor)

        reference = torch.cat(tensors_cpu).norm().reshape(1)
        if per_tensor:
            referenceb = torch.cat([t.flatten().norm().reshape(1) for t in tensors_cpu])

        self.assertTensorsEqual(
            fused_out1.cpu().float(), reference, 0.003, use_MSE=True
        )
        if per_tensor:
            self.assertTensorsEqual(
                fused_out2.cpu().float(), referenceb, 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_fused_l2norm(self):
        shape_list = [
            (0),
            (1),
            (15, 20),
            (2, 3, 4),
            (8, 1, 2, 3),
            (2, 1, 2, 1, 4),
            (33333),
            (555),
            (2048 * 32 + 1),
        ]
        op_list = [
            torch.ops.torch_mlu.fused_l2_norm_amp,
            torch.ops.torch_mlu.fused_l2_norm,
        ]
        dtype_list = [torch.float, torch.half]
        per_tensor_list = [True, False]
        len_list = [1, 2, 17, 35, 49]
        loop_var = [shape_list, op_list, dtype_list, per_tensor_list, len_list]
        for shape, op, dtype, per_tensor, len in product(*loop_var):
            self.fused_l2norm_dtype(
                op=op, test_type=dtype, shape=shape, per_tensor=per_tensor, len=len
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_fused_l2norm_amp_nan(self):
        device = "mlu"
        length = 100
        dtype = torch.float
        overflow_buf = torch.zeros(1, device=device, dtype=torch.int)
        per_tensor = True
        tensors_cpu = []
        tensors_mlu = []
        for i in range(length):
            if i % 3 == 0:
                tensor = torch.randn((12, 23), dtype=torch.float)
            else:
                tensor = torch.randn((12, 23, 3), dtype=torch.float)
            if i == 25:
                tensor.view(-1)[3] = float("nan")
            tensors_cpu.append(tensor.to(dtype))
            tensors_mlu.append(tensor.to(device).to(dtype))
        norm, norm_per_tensor = torch.ops.torch_mlu.fused_l2_norm_amp(
            overflow_buf, tensors_mlu, per_tensor
        )
        norm_expect = torch.zeros_like(norm)
        norm_per_tensor_expect = torch.zeros_like(norm_per_tensor)
        self.assertTensorsEqual(norm.cpu(), norm_expect.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(
            norm_per_tensor.cpu(), norm_per_tensor_expect.cpu(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_fused_l2norm_bfloat16(self):
        shape_list = [
            (0),
            (1),
            (15, 20),
            (2, 3, 4),
            (8, 1, 2, 3),
            (2, 1, 2, 1, 4),
            (33333),
            (555),
            (2048 * 32 + 1),
        ]
        op_list = [
            torch.ops.torch_mlu.fused_l2_norm_amp,
            torch.ops.torch_mlu.fused_l2_norm,
        ]
        dtype_list = [torch.bfloat16]
        per_tensor_list = [True, False]
        len_list = [1, 2, 17, 35, 49]
        loop_var = [shape_list, op_list, dtype_list, per_tensor_list, len_list]
        for shape, op, dtype, per_tensor, len in product(*loop_var):
            self.fused_l2norm_dtype(
                op=op, test_type=dtype, shape=shape, per_tensor=per_tensor, len=len
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_fused_l2norm_exception(self):
        msg_ref_list = [
            "\"bang_fused_l2_norm\" not implemented for 'Long'",
            "\"bang_fused_l2_norm_amp\" not implemented for 'Long'",
        ]
        op_list = [
            torch.ops.torch_mlu.fused_l2_norm,
            torch.ops.torch_mlu.fused_l2_norm_amp,
        ]
        for msg_ref, op in zip(msg_ref_list, op_list):
            with self.assertRaisesRegex(RuntimeError, msg_ref):
                self.fused_l2norm_dtype(op=op, test_type=torch.long, shape=(1, 2))


if __name__ == "__main__":
    unittest.main()
