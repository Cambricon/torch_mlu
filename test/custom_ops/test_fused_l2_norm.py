from itertools import product
import logging
import sys
import os
import unittest

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import (
    testinfo,
    read_card_info,
    TestCase,
)  # pylint: disable=C0413,C0411
from common_utils import TEST_LARGETENSOR, largeTensorTest

TEST_BFLOAT16 = read_card_info()


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

        sizeA = shape[0]
        sizeB = shape[1]
        for _ in range(len):
            tensorA = torch.randn(sizeA, dtype=torch.float)
            tensorB = torch.randn(sizeB, dtype=torch.float)
            tensors_cpu.extend([tensorA, tensorB])
            tensors_mlu.extend(
                [tensorA.to("mlu").to(test_type), tensorB.to("mlu").to(test_type)]
            )

        fused_out1, fused_out2 = op(self._dummy_overflow_buf, tensors_mlu, per_tensor)

        tensors_cpu_flatten = [item.flatten() for item in tensors_cpu]
        reference = torch.cat(tensors_cpu_flatten).norm().reshape(1)
        if per_tensor:
            referenceb = torch.cat([t.flatten().norm().reshape(1) for t in tensors_cpu])

        self.assertTensorsEqual(
            fused_out1.cpu().float(), reference, 0.003, use_MSE=True
        )
        if per_tensor:
            self.assertTensorsEqual(
                fused_out2.cpu().float(), referenceb, 0.003, use_MSE=True
            )
        self.assertTrue(self._dummy_overflow_buf.item() == 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_fused_l2norm(self):
        shape_list = [
            ((0), (333 * 2)),
            ((1), (333 * 2)),
            ((15, 20), (333 * 2)),
            ((2, 3, 4), (333 * 2)),
            ((8, 1, 2, 3), (333 * 2)),
            ((2, 1, 2, 1, 4), (333 * 2)),
            ((33333), (333 * 2)),
            ((555), (333 * 2)),
            ((2048 * 32 + 1), (333 * 2)),
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
    def test_fused_l2norm_multi_tensors(self):
        shape_list = [
            ((0), (333 * 2)),
            ((2048 * 32 + 1), (333 * 2)),
        ]
        op_list = [
            torch.ops.torch_mlu.fused_l2_norm_amp,
            torch.ops.torch_mlu.fused_l2_norm,
        ]
        len = 500
        loop_var = [shape_list, op_list]
        for shape, op in product(*loop_var):
            self.fused_l2norm_dtype(
                op=op, test_type=torch.float, shape=shape, per_tensor=True, len=len
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
            ((0), (333 * 2)),
            ((1), (333 * 2)),
            ((15, 20), (333 * 2)),
            ((33333), (333 * 2)),
            ((555), (333 * 2)),
            ((2048 * 32 + 1), (333 * 2)),
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
        msg_ref = "\"bang_fused_l2_norm\" not implemented for 'Long'"
        op_list = [
            torch.ops.torch_mlu.fused_l2_norm,
            torch.ops.torch_mlu.fused_l2_norm_amp,
        ]
        for op in op_list:
            with self.assertRaisesRegex(RuntimeError, msg_ref):
                self.fused_l2norm_dtype(op=op, test_type=torch.long, shape=(1, 2))

    @unittest.skipIf(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("49GB")
    def test_fused_adam_large(self):
        # add custom cases
        size_list = [
            (11003, 23986),
            (2147483660),
        ]
        num_list = [24, 6]
        for size, num in zip(size_list, num_list):
            tensors = []
            for i in range(num):
                tensors.append(torch.rand(size, dtype=torch.float, device="mlu"))
            op_list = [
                torch.ops.torch_mlu.fused_l2_norm,
                torch.ops.torch_mlu.fused_l2_norm_amp,
            ]

            for op in op_list:
                fused_out1, fused_out2 = op(self._dummy_overflow_buf, tensors, True)
            torch.mlu.synchronize()


if __name__ == "__main__":
    unittest.main()
