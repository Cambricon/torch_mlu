from __future__ import print_function

import sys
import os
import unittest
import logging
import copy
from itertools import product
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    read_card_info,
    largeTensorTest,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestMaxOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_max_dim_and_out(self):
        def create_input(shape, dtype):
            if dtype == torch.bool:
                low = 0
                high = 2
                return torch.randint(low, high, shape, dtype=dtype)
            else:
                return torch.randn(*shape, dtype=torch.float).to(dtype)

        shape_list = [
            (2, 3, 4),
            (1, 3, 224),
            (1, 3, 1, 1, 1),
            (1, 3, 224, 224),
            (1, 1, 1, 2),
        ]
        dim_list = [1, -1, 0, 2, 3]
        type_list = [True, False, True, False, False]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., ::2]]
        dtypes = [torch.double, torch.float, torch.long, torch.int, torch.bool]
        for func in func_list:
            for i, _ in enumerate(shape_list):
                for t in dtypes:
                    x = create_input(shape_list[i], t)
                    out_cpu = torch.max(func(x), dim_list[i], keepdim=type_list[i])
                    out_mlu = torch.max(
                        func(self.to_device(x)), dim_list[i], keepdim=type_list[i]
                    )
                    self.assertTensorsEqual(
                        out_cpu[0].float(), out_mlu[0].cpu().float(), 0.0, use_MSE=True
                    )
                    # max sorting algorithm for mlu is different from cpu,
                    # when value is the same the max index may be different,
                    # in this case, index test is not included for max in unit test.

            # test max_dim_out
            for i, _ in enumerate(shape_list):
                x = torch.randn(shape_list[i], dtype=torch.float)
                out_cpu = torch.max(func(x), dim_list[i], keepdim=type_list[i])
                out_mlu_value = copy.deepcopy(out_cpu[0]).mlu()
                out_mlu_index = copy.deepcopy(out_cpu[1]).mlu()
                torch.max(
                    func(self.to_device(x)),
                    dim_list[i],
                    keepdim=type_list[i],
                    out=(out_mlu_value, out_mlu_index),
                )
                out_mlu = torch.max(
                    func(self.to_device(x)), dim_list[i], keepdim=type_list[i]
                )
                self.assertTensorsEqual(
                    out_cpu[0].float(), out_mlu_value.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_max_other(self):
        type_list = [torch.float, torch.int, torch.long, torch.double]
        for t in type_list:
            for shape1, shape2 in [
                ((1, 1, 1024), (64, 1024, 1)),
                ((2, 2, 4, 2), (2)),
                ((2, 2, 4, 2), (1, 2)),
                ((1, 2), (2, 2, 4, 2)),
                ((2, 1, 2, 4), (1, 2, 4)),
                ((1, 2, 4), (2, 1, 2, 4)),
                ((1, 3, 1, 113, 1, 1, 1, 7), (13, 1, 17, 1, 31, 1, 1, 1)),
                ((255, 1, 5, 1, 1, 1, 1, 1), (1, 1, 1, 73, 1, 411, 1, 1)),
                ((257, 1, 1, 1, 1, 1, 1, 1), (1, 1, 13, 1, 1, 1, 1, 1)),
                ((), ()),
                ((), (1)),
                ((0), (0)),
            ]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_cpu = torch.max(x, y)
                out_mlu = torch.max(self.to_device(x), self.to_device(y))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_max_backward(self):
        for shape1, shape2 in [
            ((1, 1, 1024), (64, 1024, 1)),
            ((2, 2, 4, 2), (2)),
            ((2, 2, 4, 2), (1, 2)),
            ((1, 2), (2, 2, 4, 2)),
        ]:
            x = torch.randn(shape1, requires_grad=True)
            y = torch.randn(shape2, requires_grad=True)
            out_cpu = torch.max(x, y)
            out_mlu = torch.max(self.to_device(x), self.to_device(y))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )
            # test backward
            grad = torch.randn(out_cpu.shape)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            out_mlu.backward(grad.to("mlu"))
            grad_mlu = x.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_max(self):
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
            torch.int,
            torch.int64,
            torch.half,
            torch.float,
            torch.double,
        ]
        for func in func_list:
            for shape in shape_list:
                for dtype in dtype_list:
                    x = torch.randn(shape).to(dtype)
                    out_cpu = torch.max(func(x))
                    out_mlu = torch.max(func(self.to_device(x)))
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.float().cpu(), 0.0, use_MSE=True
                    )
        # test scalar
        x = torch.randn(())
        out_cpu = torch.max(x)
        out_mlu = torch.max(self.to_device(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_max_empty_tensor(self):
        input = torch.randn(2, 0, 4)
        out_cpu = torch.max(input, dim=2)
        out_mlu = torch.max(self.to_device(input), dim=2)
        self.assertEqual(out_cpu[0].numel(), 0)
        self.assertEqual(out_mlu[0].numel(), 0)
        self.assertEqual(out_cpu[0].shape, out_mlu[0].shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_max_out(self):
        for shape1, shape2 in [
            ((1, 1, 1024), (64, 1024, 1)),
            ((2, 1, 2, 4), (1, 2, 4)),
            ((1, 2, 4), (2, 1, 2, 4)),
            ((1, 3, 1, 113, 1, 1, 1, 7), (13, 1, 17, 1, 31, 1, 1, 1)),
            ((255, 1, 5, 1, 1, 1, 1, 1), (1, 1, 1, 73, 1, 411, 1, 1)),
            ((257, 1, 1, 1, 1, 1, 1, 1), (1, 1, 13, 1, 1, 1, 1, 1)),
        ]:
            x = torch.randn(shape1, dtype=torch.float)
            y = torch.randn(shape2, dtype=torch.float)
            out_cpu = torch.randn(1, dtype=torch.float)
            x_mlu = x.to("mlu")
            y_mlu = y.to("mlu")
            out_mlu = out_cpu.to("mlu")
            torch.max(x, y, out=out_cpu)
            torch.max(x_mlu, y_mlu, out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_max_out_not_contiguous(self):
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., 0:1]]
        list_list = [func_list, func_list]
        for xin_func, yin_func in product(*list_list):
            for shape1, shape2 in [
                ((2, 2, 2, 1, 2), (2, 2, 2, 2)),
                ((2, 1, 2, 4), (1, 2, 4)),
                ((1, 2, 4), (2, 1, 2, 4)),
                ((1, 3, 1, 1), (1, 3, 224, 224)),
                ((1, 3, 1, 224), (1, 3, 224, 224)),
            ]:
                x = torch.randn(shape1, dtype=torch.float)
                y = torch.randn(shape2, dtype=torch.float)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                out_cpu = torch.max(xin_func(x), yin_func(y))
                out_mlu = torch.max(xin_func(x_mlu), yin_func(y_mlu))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_max_exception(self):
        input = torch.randn((0)).to("mlu")
        ref_msg = r"max\(\): Expected reduction dim to be specified for input.numel\(\) == 0. "
        ref_msg = ref_msg + "Specify the reduction dim with the 'dim' argument."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.max(input)

    # @unittest.skip("not test")
    @testinfo()
    def test_max_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.randn(shape_list[i], dtype=torch.float)
            y = torch.randn(shape_list[i], dtype=torch.float)
            out = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            y_mlu = copy.deepcopy(y).to("mlu")
            out_mlu = copy.deepcopy(out).to("mlu")
            x, y, out = (
                x.permute(permute_shape[i]),
                y.permute(permute_shape[i]),
                out.permute(permute_shape[i]),
            )
            x_mlu, y_mlu, out_mlu = (
                x_mlu.permute(permute_shape[i]),
                y_mlu.permute(permute_shape[i]),
                out_mlu.permute(permute_shape[i]),
            )
            torch.max(x, y, out=out)
            torch.max(x_mlu, y_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_max_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype in dtype_list:
            x = torch.randn((2, 3, 4, 5, 6), dtype=torch.half)
            y = torch.randn((2, 3, 4, 5, 6), dtype=torch.half)
            x_mlu, y_mlu = self.to_mlu_dtype(x, dtype), self.to_mlu_dtype(y, dtype)
            x, y = x.float(), y.float()
            output = torch.max(x, y)
            output_mlu = torch.max(x_mlu, y_mlu)
            self.assertTensorsEqual(output, output_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_max_zero_dim_cpu_tensor(self):
        x = torch.randn((2, 3, 4, 5, 6), dtype=torch.float)
        zero_dim_cpu_tensor = torch.randn([])
        x_mlu = x.mlu()
        output = torch.max(x, zero_dim_cpu_tensor)
        output_mlu_1 = torch.max(x_mlu, zero_dim_cpu_tensor)
        output_mlu_2 = torch.max(zero_dim_cpu_tensor, x_mlu)
        self.assertTensorsEqual(output, output_mlu_1.cpu().float(), 0.0, use_MSE=True)
        self.assertTensorsEqual(output, output_mlu_2.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_max_big_number(self):
        big_num1 = torch.iinfo(torch.int).max - 1
        big_num2 = torch.iinfo(torch.int).max - 11
        big_num3 = torch.iinfo(torch.int).max - 5
        big_num4 = torch.iinfo(torch.int).max - 20
        big_num5 = torch.iinfo(torch.int).max - 10

        x = torch.tensor([[big_num1, big_num2], [big_num3, big_num4]]).int()
        zero_dim_cpu_tensor = torch.tensor(big_num5).int()
        x_mlu = x.mlu()
        output = torch.max(x, zero_dim_cpu_tensor)
        output_mlu = torch.max(x_mlu, zero_dim_cpu_tensor)
        diff = output - output_mlu.cpu()
        base_diff = torch.tensor([[0, 0], [0, 0]]).int()
        # TODO(xujian1): closed for banc command not support int type.
        # self.assertTrue(torch.allclose(diff, base_diff, atol=0.0, rtol=0.0))

    # @unittest.skip("not test")
    @testinfo()
    def test_max_no_contiguous_tensor(self):
        device = "mlu"
        y = torch.randn(5, 3, device=device)
        values = torch.randn(5, 3, device=device)
        indices = torch.zeros(5, 3, device=device).long() - 1
        torch.max(y, 1, keepdim=False, out=(values[:, 1], indices[:, 1]))
        values_expected, indices_expected = torch.max(y, 1, keepdim=False)
        self.assertTensorsEqual(
            values[:, 1].cpu(), values_expected.cpu(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            indices[:, 1].cpu(), indices_expected.cpu(), 0.0, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("58GB")
    def test_max_large(self):
        dtype_list = [torch.half]
        for dtype in dtype_list:
            for shape1, shape2 in [((5, 1024, 1024, 1024), (5, 1024, 1024, 1024))]:
                x = torch.randn(shape1).to(dtype)
                y = torch.randn(shape2).to(dtype)
                out_cpu = torch.max(x, y)
                out_mlu = torch.max(self.to_device(x), self.to_device(y))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_max_bfloat16(self):
        left = torch.testing.make_tensor(
            (1, 10, 24), dtype=torch.bfloat16, device="cpu"
        )
        right = torch.testing.make_tensor(
            (64, 10, 1), dtype=torch.bfloat16, device="cpu"
        )
        left_cpu = torch.nn.Parameter(left)
        right_cpu = torch.nn.Parameter(right)
        left_mlu = torch.nn.Parameter(left.mlu())
        right_mlu = torch.nn.Parameter(right.mlu())
        out_cpu = torch.max(left_cpu, right_cpu)
        out_mlu = torch.max(left_mlu, right_mlu)
        # TODO(CNNLCORE-14058): backward not support bfloat16
        # grad = torch.randn(out_cpu.shape)
        # grad_mlu = grad.mlu()
        # out_cpu.backward(grad)
        # out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )
        # self.assertTensorsEqual(left_cpu.grad.float(),
        #                         left_mlu.grad.cpu().float(),
        #                         0.0,
        #                         use_MSE=True)
        # self.assertTensorsEqual(right_cpu.grad.float(),
        #                         right_mlu.grad.cpu().float(),
        #                         0.0,
        #                         use_MSE=True)
        out_cpu, _ = torch.max(left_cpu, 1, keepdim=True)
        out_mlu, _ = torch.max(left_mlu, 1, keepdim=True)
        # left_cpu.grad.zero_()
        # left_mlu.grad.zero_()
        # grad = torch.randn(out_cpu.shape)
        # grad_mlu = grad.mlu()
        # out_cpu.backward(grad)
        # out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )
        # self.assertTensorsEqual(left_cpu.grad.float(),
        #                         left_mlu.grad.cpu().float(),
        #                         0.0,
        #                         use_MSE=True)


if __name__ == "__main__":
    run_tests()
