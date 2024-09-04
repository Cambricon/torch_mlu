from __future__ import print_function

import sys
import os
import copy
import unittest
from itertools import product
import logging

from torch._C import device, dtype  # pylint: disable=W0611

import torch
from torch.autograd import Variable

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestZerosOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_zeros_like(self):
        shape_list = [(99), (14, 15, 16, 17, 18), (98, 97, 94)]
        data_types = [torch.float, torch.half, torch.int64, torch.int32]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., ::2]]
        for shape, data_type, func in product(shape_list, data_types, func_list):
            # test zeros_like
            a = torch.rand(shape).to(data_type)
            result_cpu = torch.zeros_like(func(a))
            result_mlu = torch.zeros_like(func(self.to_mlu_dtype(a, data_type)))
            self.assertTensorsEqual(
                result_cpu.float(), result_mlu.cpu().float().contiguous(), 0
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_zeros_like_bound(self):
        shape = (2, 246625072)
        data_type = torch.half
        func = lambda x: x
        a = torch.rand(shape).to(data_type)
        result_cpu = torch.zeros_like(func(a))
        result_mlu = torch.zeros_like(func(self.to_mlu_dtype(a, data_type)))
        self.assertTensorsEqual(
            result_cpu.float(), result_mlu.cpu().float().contiguous(), 0.0
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_zeros_like_dtype(self):
        shape_list = [(99), (14, 15, 16, 17, 18), (98, 97, 94)]
        data_types = [
            torch.bool,
            torch.half,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.long,
            torch.uint8,
        ]
        for shape, data_type in product(shape_list, data_types):
            a = torch.rand(shape).to(data_type)
            result_cpu = torch.zeros_like(a)
            result_mlu = torch.zeros_like(self.to_mlu_dtype(a, data_type))
            self.assertTensorsEqual(result_cpu.float(), result_mlu.cpu().float(), 0)
        for data_type in [torch.bool, torch.int]:
            a = torch.randn(3)
            result_cpu = torch.zeros_like(a, dtype=data_type)
            result_mlu = torch.zeros_like(self.to_device(a), dtype=data_type)
            self.assertEqual(result_cpu.dtype, result_mlu.dtype)
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_zeros_like_grad(self):
        shape_list = [(99), (14, 15, 16, 17, 18), (98, 97, 94)]
        data_types = [torch.float, torch.half]
        channel_first = [True, False]
        for shape, data_type in product(shape_list, data_types):
            for channel in channel_first:
                a = torch.rand(shape).to(torch.float)
                result_cpu = torch.zeros_like(a, requires_grad=True)
                if channel is False:
                    a = self.convert_to_channel_last(a)
                result_mlu = torch.zeros_like(
                    self.to_mlu_dtype(a, data_type), requires_grad=True
                )
                b = result_cpu + 1
                b_mlu = result_mlu + 1
                grad_out = torch.ones(shape)
                b.backward(grad_out)
                if channel is False:
                    grad_out = self.convert_to_channel_last(grad_out)
                b_mlu.backward(self.to_mlu_dtype(grad_out, data_type))
                self.assertTensorsEqual(
                    result_cpu.float(), result_mlu.cpu().float().contiguous(), 0
                )
                self.assertTensorsEqual(
                    result_cpu.grad.float(),
                    result_mlu.grad.cpu().float().contiguous(),
                    0,
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_zeros(self):
        shape_pin_memory_list = [
            ((99), False),
            ((14, 15, 16, 17, 18), True),
            ((98, 97, 94), True),
        ]
        for shape, pin_memory in shape_pin_memory_list:
            # test zeros
            result_cpu = torch.zeros(shape, pin_memory=pin_memory)
            result_mlu = torch.zeros(shape, device="mlu")
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
            if pin_memory is True:
                self.assertTrue(
                    result_cpu.is_pinned(), "create cpu pin memory tensor failed."
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_zeros_dtype(self):
        shape_list = [(99), (14, 15, 16, 17, 18), (98, 97, 94)]
        data_types = [
            torch.bool,
            torch.float,
            torch.half,
            torch.int,
            torch.short,
            torch.int8,
            torch.long,
            torch.uint8,
        ]
        for shape, data_type in product(shape_list, data_types):
            x = torch.zeros(shape, dtype=data_type)
            x_mlu = torch.zeros(shape, device="mlu", dtype=data_type)
            self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0)

    # zeros.out can not support inplace now
    # @unittest.skip("not test")
    @testinfo()
    def test_zeros_out(self):
        shape_list = [(99), (14, 15, 16, 17, 18), (98, 97, 94)]
        for shape in shape_list:
            y = torch.randn(shape, dtype=torch.float)
            y_mlu = self.to_device(copy.deepcopy(y))
            torch.zeros(shape, out=y)
            torch.zeros(shape, out=y_mlu)
            self.assertTensorsEqual(y, y_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_zeros_grad(self):
        shape_list = [(99), (14, 15, 16, 17, 18), (98, 97, 94)]
        for shape in shape_list:
            result_cpu = Variable(torch.zeros(shape), requires_grad=True)
            result_mlu = Variable(torch.zeros(shape, device="mlu"), requires_grad=True)
            b = result_cpu + 1
            b_mlu = result_mlu + 1
            b.backward(torch.ones(shape))
            b_mlu.backward(self.to_device(torch.ones(shape)))
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
            self.assertTensorsEqual(result_cpu.grad, result_mlu.grad.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_zero_(self):
        shape_list = [(14, 15, 16, 17, 18), (98, 97, 94)]
        contiguous = [True, False]
        for shape in shape_list:
            for con in contiguous:
                a = torch.rand(shape, dtype=torch.float)
                if con is True:
                    a = self.get_not_contiguous_tensor(a)
                b = copy.deepcopy(a)
                result_cpu = a.zero_()
                b_mlu = self.to_device(b)
                y_orig_ptr = b_mlu.data_ptr()
                result_mlu = b_mlu.zero_()
                y_ptr = result_mlu.data_ptr()
                self.assertTensorsEqual(result_cpu, result_mlu.cpu().contiguous(), 0)
                self.assertEqual(y_orig_ptr, y_ptr)

    # @unittest.skip("not test")
    @testinfo()
    def test_new_zeros(self):
        size_list = [
            (),
            (2,),
            (2, 3, 4),
            (2, 3, 4, 5),
            (2, 0, 3, 4),
            (2, 3, 4, 5, 6),
            (2, 3, 4, 5, 6, 7),
        ]
        type_list = [
            None,
            torch.bool,
            torch.float,
            torch.half,
            torch.int,
            torch.short,
            torch.int8,
            torch.long,
            torch.uint8,
        ]
        data = torch.tensor(())
        for size, type in product(size_list, type_list):
            result_cpu = data.new_zeros(size, dtype=type)
            result_mlu = data.new_zeros(size, dtype=type, device="mlu")
            self.assertTensorsEqual(
                result_cpu,
                result_mlu.cpu() if type != torch.half else result_mlu.cpu().float(),
                0,
            )
            self.assertEqual(result_mlu.dtype, result_cpu.dtype)
            self.assertEqual(result_mlu.dtype, type if type else torch.float)
            self.assertEqual(result_mlu.size(), result_cpu.size())
            self.assertEqual(result_mlu.size(), size)


if __name__ == "__main__":
    unittest.main()
