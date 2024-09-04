from __future__ import print_function

import sys
import logging
import os
import copy
import unittest

import torch
import torch_mlu

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


class TestFloorOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_floor(self):
        shape_list = [
            (1, 3, 224, 224),
            (2, 3, 4),
            (2, 2),
            (254, 254, 112, 1, 1, 3),
            (0, 2, 3),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.floor(x)
            out_mlu = torch.floor(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_scalar(self):
        x_0 = torch.tensor(-1.57)
        out_cpu = torch.floor(x_0)
        out_mlu = torch.floor(self.to_mlu(x_0))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_inplace(self):
        shape_list = [(1, 3, 224, 224), (2, 3, 4), (2, 2), (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = self.to_mlu(x)
            y_data = y.data_ptr()
            torch.floor_(x)
            torch.floor_(y)
            self.assertEqual(y_data, y.data_ptr())
            self.assertTensorsEqual(x, y.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_t(self):
        shape_list = [(1, 3, 224, 224), (2, 3, 4), (2, 2), (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = x.floor()
            out_mlu = self.to_mlu(x).floor()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_out(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(1, dtype=torch.float)
            y_mlu = copy.deepcopy(y).to(torch.device("mlu"))

            torch.floor(x, out=y)
            torch.floor(self.to_mlu(x), out=y_mlu)
            self.assertTensorsEqual(y, y_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_channelslast_and_nodense(self):
        def run_test(x):
            out_cpu = torch.floor(x)
            out_mlu = torch.floor(x.to("mlu"))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

        shape_list = [(64, 3, 6, 6), (2, 25, 64, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)

            # channels_last input
            run_test(x.to(memory_format=torch.channels_last))

            # not-dense input
            run_test(x[..., :2])

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = copy.deepcopy(x).mlu()
            out_mlu = copy.deepcopy(out).mlu()
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                permute_shape[i]
            )
            torch.floor(x, out=out)
            torch.floor(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype in dtype_list:
            x = torch.randn((2, 3, 4, 5, 6), dtype=torch.half)
            x_mlu = self.to_mlu_dtype(x, dtype)
            x = x.float()
            x.floor_()
            x_mlu.floor_()
            self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_backward(self):
        shape_list = [
            (66),
            (39, 48),
            (16, 27, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 13, 21),
            (6, 7, 8, 9, 10, 11),
            (11, 13, 16, 18, 20, 23),
        ]
        type_list = [torch.float]
        for shape in shape_list:
            for data_type in type_list:
                x_0 = torch.randn(shape, dtype=data_type)
                x_mlu = x_0.to("mlu")
                x_0.requires_grad_(True)
                x_mlu.requires_grad_(True)
                out_cpu = torch.floor(x_0)
                out_mlu = torch.floor(x_mlu)
                out_cpu.backward(torch.ones_like(out_cpu))
                out_mlu.backward(torch.ones_like(out_mlu))
                self.assertTensorsEqual(
                    x_0.grad, x_mlu.grad.cpu(), 0.003, allow_inf=True, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("46GB")
    def test_floor_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.floor(x)
            out_mlu = torch.floor(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_floor_bfloat16(self):
        shape_list = [
            (66),
            (39, 48),
            (16, 27, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 13, 21),
            (6, 7, 8, 9, 10, 11),
            (11, 13, 16, 18, 20, 23),
        ]
        for shape in shape_list:
            x_0 = torch.randn(shape, dtype=torch.bfloat16)
            x_mlu = x_0.to("mlu")
            x_0.requires_grad_(True)
            x_mlu.requires_grad_(True)
            out_cpu = torch.floor(x_0)
            out_mlu = torch.floor(x_mlu)
            out_cpu.backward(torch.ones_like(out_cpu))
            out_mlu.backward(torch.ones_like(out_mlu))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
            self.assertTensorsEqual(
                x_0.grad.float(),
                x_mlu.grad.cpu().float(),
                0.003,
                allow_inf=True,
                use_MSE=True,
            )


if __name__ == "__main__":
    run_tests()
