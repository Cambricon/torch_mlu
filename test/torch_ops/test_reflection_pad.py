from __future__ import print_function
import copy

import sys
import logging
import os
import unittest
from itertools import product
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0411,C0413

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestReflectionPadOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_reflection_pad2d(self):
        shape_list = [(2, 3, 4, 5), (3, 4, 5), (256, 32, 12, 12), (0, 2, 12, 12)]
        pad_list = [(1, 1, 2, 3), (0, 2, 1, 3), 1]
        type_list = [(torch.double, 0.0), (torch.float, 0.0), (torch.half, 0.003)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        loop_list = [shape_list, pad_list, type_list, func_list]
        for shape, pad, (dtype, err), func in product(*loop_list):
            m = torch.nn.ReflectionPad2d(pad)
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x)
            x.requires_grad = True
            x_mlu.requires_grad = True
            out_cpu = m(func(x))
            out_mlu = m(func(x_mlu.to(dtype).to("mlu")))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            grad = torch.randn(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad).to("mlu")
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(x.grad, x_mlu.grad.float(), err, use_MSE=True)

        # test nan/inf
        m = torch.nn.ReflectionPad2d(1)
        x = torch.randn((2, 2, 3, 4), dtype=torch.float)
        x[0, :, 2] = float("nan")
        x[1, :, 1] = float("inf")
        out_cpu = m(x)
        out_mlu = m(x.mlu())
        self.assertEqual(out_cpu, out_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_reflection_pad1d(self):
        shape_list = [(2, 3, 4), (32, 3, 224), (3, 4), (256, 32, 64), (0, 2, 3)]
        pad_list = [(1, 1), (0, 2), 1]
        type_list = [(torch.double, 0.0), (torch.float, 0.0), (torch.half, 0.003)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        loop_list = [shape_list, pad_list, type_list, func_list]
        for shape, pad, (dtype, err), func in product(*loop_list):
            m = torch.nn.ReflectionPad1d(pad)
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x)
            x.requires_grad = True
            x_mlu.requires_grad = True
            out_cpu = m(func(x))
            out_mlu = m(func(x_mlu.to(dtype).to("mlu")))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            grad = torch.randn(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad).to("mlu")
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(x.grad, x_mlu.grad.float(), err, use_MSE=True)

        # test nan/inf
        m = torch.nn.ReflectionPad1d(1)
        x = torch.randn((2, 3, 4), dtype=torch.float)
        x[0, :, 2] = float("nan")
        x[1, :, 1] = float("inf")
        out_cpu = m(x)
        out_mlu = m(x.mlu())
        self.assertEqual(out_cpu, out_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_reflection_pad1d_exception(self):
        with self.assertRaisesRegex(RuntimeError, "2D or 3D \(batch mode\)"):
            input = torch.randn((1, 0, 4), dtype=torch.float32, device="mlu")
            m = torch.nn.ReflectionPad1d(1)
            res_mlu = m(input)

        with self.assertRaisesRegex(RuntimeError, "Argument #4: Padding size"):
            input = torch.randn((1, 2, 4), dtype=torch.float32, device="mlu")
            m = torch.nn.ReflectionPad1d(5)
            res_mlu = m(input)

        with self.assertRaisesRegex(RuntimeError, "input \(W: "):
            input = torch.randn((1, 2, 2), dtype=torch.float32, device="mlu")
            m = torch.nn.ReflectionPad1d(-1)
            res_mlu = m(input)

        with self.assertRaisesRegex(RuntimeError, "not implemented for 'ComplexFloat'"):
            input = torch.randn((1, 2, 2), dtype=torch.complex64, device="mlu")
            m = torch.nn.ReflectionPad1d(1)
            res_mlu = m(input)

        with self.assertRaisesRegex(RuntimeError, "negative padding"):
            input = torch.randn(
                (1, 2, 4), dtype=torch.float32, device="mlu"
            ).requires_grad_()
            m = torch.nn.ReflectionPad1d(-1)
            res_mlu = m(input)
            res_mlu.sum().backward()

    # @unittest.skip("not test")
    @testinfo()
    def test_reflection_pad2d_exception(self):
        with self.assertRaisesRegex(RuntimeError, "3D or 4D \(batch mode\)"):
            input = torch.randn((1, 0, 4, 2), dtype=torch.float32, device="mlu")
            m = torch.nn.ReflectionPad2d(1)
            res_mlu = m(input)

        with self.assertRaisesRegex(
            RuntimeError,
            "Padding size should be less than the corresponding input dimension",
        ):
            input = torch.randn((1, 2, 4, 2), dtype=torch.float32, device="mlu")
            m = torch.nn.ReflectionPad2d((5, 5, 1, 1))
            res_mlu = m(input)

        with self.assertRaisesRegex(
            RuntimeError,
            "Padding size should be less than the corresponding input dimension",
        ):
            input = torch.randn((1, 2, 4, 2), dtype=torch.float32, device="mlu")
            m = torch.nn.ReflectionPad2d((1, 1, 5, 5))
            res_mlu = m(input)

        with self.assertRaisesRegex(RuntimeError, "input \(H: "):
            input = torch.randn((1, 1, 2, 2), dtype=torch.float32, device="mlu")
            m = torch.nn.ReflectionPad2d(-1)
            res_mlu = m(input)

        with self.assertRaisesRegex(RuntimeError, "not implemented for 'ComplexFloat'"):
            input = torch.randn((1, 2, 2, 2), dtype=torch.complex64, device="mlu")
            m = torch.nn.ReflectionPad2d(1)
            res_mlu = m(input)

        with self.assertRaisesRegex(RuntimeError, "negative padding"):
            input = torch.randn(
                (1, 2, 4, 2), dtype=torch.float32, device="mlu"
            ).requires_grad_()
            m = torch.nn.ReflectionPad2d(-1)
            res_mlu = m(input)
            res_mlu.sum().backward()

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_reflection_pad2d_bfloat16(self):
        torch.manual_seed(1234)
        shape_list = [
            (2, 3, 4, 5),
        ]
        pad_list = [
            (1, 1, 2, 3),
        ]
        type_list = [
            (torch.bfloat16, 0.003),
        ]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        loop_list = [shape_list, pad_list, type_list, func_list]
        for shape, pad, (dtype, err), func in product(*loop_list):
            m = torch.nn.ReflectionPad2d(pad)
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x)
            x.requires_grad = True
            x_mlu.requires_grad = True
            out_cpu = m(func(x))
            out_mlu = m(func(x_mlu.to(dtype).to("mlu")))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            grad = torch.randn(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad).to(dtype).to("mlu")
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(x.grad, x_mlu.grad.float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_pad_edge_case(self):
        input_cpu = torch.randn([4, 1, 65536]).as_strided(
            size=[4, 1, 65536], stride=[65536, 65536, 1]
        )
        input_mlu = input_cpu.mlu()
        n_pad = 2

        output_cpu = torch.nn.functional.pad(input_cpu, (0, n_pad), "reflect")
        output_mlu = torch.nn.functional.pad(input_mlu, (0, n_pad), "reflect")

        self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_reflection_pad1d_bfloat16(self):
        shape_list = [
            (2, 3, 4),
        ]
        pad_list = [
            (1, 1),
        ]
        type_list = [
            (torch.bfloat16, 0.003),
        ]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        loop_list = [shape_list, pad_list, type_list, func_list]
        for shape, pad, (dtype, err), func in product(*loop_list):
            m = torch.nn.ReflectionPad1d(pad)
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x)
            x.requires_grad = True
            x_mlu.requires_grad = True
            out_cpu = m(func(x))
            out_mlu = m(func(x_mlu.to(dtype).to("mlu")))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            grad = torch.randn(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad).to(dtype).to("mlu")
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(x.grad, x_mlu.grad.float(), err, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
