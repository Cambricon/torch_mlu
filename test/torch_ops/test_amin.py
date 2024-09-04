from __future__ import print_function

import sys
import os
import unittest
import logging
import copy
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestMinOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_amin_dim_and_out(self):
        shape_list = [
            (2, 3, 4),
            (1, 3, 224),
            (1, 3, 1, 1, 1),
            (1, 3, 224, 224),
            (1, 1, 1, 2),
            (2, 3, 4, 5),
            (3, 4, 5, 6),
            (1, 2, 3, 4, 5),
        ]
        dim_list = [1, -1, 0, 2, 3, (2, 3), (-1, 1), (0, 2, 4)]
        type_list = [True, False, True, False, False, True, False, True, False]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., ::2]]
        for func in func_list:
            for i, _ in enumerate(shape_list):
                x = torch.randn(shape_list[i], dtype=torch.float)
                out_cpu = torch.amin(func(x), dim_list[i], keepdim=type_list[i])
                out_mlu = torch.amin(
                    func(self.to_device(x)), dim_list[i], keepdim=type_list[i]
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
                # amin sorting algorithm for mlu is different from cpu,
                # when value is the same the amin index may be different,
                # in this case, index test is not included for amin in unit test.

            # test amin_dim_out
            for i, _ in enumerate(shape_list):
                x = torch.randn(shape_list[i], dtype=torch.float)
                out_cpu = torch.amin(func(x), dim_list[i], keepdim=type_list[i])
                out_mlu_value = copy.deepcopy(out_cpu).mlu()
                torch.amin(
                    func(self.to_device(x)),
                    dim_list[i],
                    keepdim=type_list[i],
                    out=out_mlu_value,
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu_value.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_amin(self):
        shape_list = [
            (2, 3, 4, 113, 4, 2, 1),
            (64, 3, 4),
            (1, 32, 5, 12, 8),
            (2, 128, 10, 6),
            (2, 512, 8),
            (1, 100),
            (24,),
            (1, 1, 1, 73, 1, 411, 1, 1),
            (1, 1, 1, 2),
            (1, 1, 1, 1),
        ]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., ::2]]
        dtype_list = [
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.long,
            torch.half,
            torch.float,
            torch.double,
        ]
        for func in func_list:
            for shape in shape_list:
                for dtype in dtype_list:
                    x = torch.randn(shape).to(dtype)
                    out_cpu = torch.amin(func(x))
                    out_mlu = torch.amin(func(self.to_device(x)))
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.float().cpu(), 0.0, use_MSE=True
                    )
        # test scalar
        x = torch.randn(())
        out_cpu = torch.amin(x)
        out_mlu = torch.amin(self.to_device(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_amin_empty_tensor(self):
        input = torch.randn(2, 0, 4)
        out_cpu = torch.amin(input, dim=2)
        out_mlu = torch.amin(self.to_device(input), dim=2)
        self.assertEqual(out_cpu[0].numel(), 0)
        self.assertEqual(out_mlu[0].numel(), 0)
        self.assertEqual(out_cpu[0].shape, out_mlu[0].shape)

    # TODO(hyl): dependcy div.out op
    # @unittest.skip("not test")
    @testinfo()
    def test_amin_backward(self):
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for dim in range(-dim_len, dim_len):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x_mlu = self.to_device(x)

                out_cpu = torch.amin(x, dim)
                grad = torch.randn(out_cpu.shape)
                grad_mlu = copy.deepcopy(grad).to("mlu")
                out_cpu.backward(grad)
                x_grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()

                out_mlu = torch.amin(x_mlu, dim)
                out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    x_grad_cpu.float(), x.grad.float(), 0.003, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_amin_bfloat16(self):
        left = torch.testing.make_tensor(
            (2, 10, 24), dtype=torch.bfloat16, device="cpu"
        )
        left_cpu = torch.nn.Parameter(left)
        left_mlu = torch.nn.Parameter(left.mlu())
        out_cpu = torch.amin(left_cpu, 1, keepdim=True)
        out_mlu = torch.amin(left_mlu, 1, keepdim=True)
        # TODO(): backward not support bfloat16
        # grad = torch.randn(out_cpu.shape).bfloat16()
        # out_cpu.backward(grad)
        # out_mlu.backward(grad.mlu())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )
        # self.assertTensorsEqual(left_cpu.grad.float(),
        #                         left_mlu.grad.cpu().float(),
        #                         0.0,
        #                         use_MSE=True)
        left.requires_grad = False
        left_mlu.requires_grad = False
        output_cpu = torch.testing.make_tensor(
            out_cpu.shape, dtype=torch.bfloat16, device="cpu"
        )
        output_mlu = output_cpu.mlu()
        torch.amin(left, 1, keepdim=True, out=output_cpu)
        torch.amin(left_mlu, 1, keepdim=True, out=output_mlu)
        self.assertTensorsEqual(
            output_mlu.cpu().float(), output_cpu.float(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            output_mlu.cpu().float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    def test_amin_large(self):
        shape_list = [(48, 4096, 13725), (1, 4096 * 48 * 13725)]
        dtype_list = [torch.bool, torch.int8, torch.half]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape).to(dtype)
                out_cpu = torch.amin(x)
                out_mlu = torch.amin(self.to_device(x))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.float().cpu(), 0.0, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
