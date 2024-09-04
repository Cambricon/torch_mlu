from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411


class TestUniqueOp(TestCase):
    def _test_unique(self, input_cpu, input_mlu, sort, inverse, counts, dim=None):
        if inverse and counts:
            output_cpu, inverse_indices_cpu, counts_cpu = torch.unique(
                input_cpu,
                sorted=sort,
                return_inverse=inverse,
                return_counts=counts,
                dim=dim,
            )
            output_mlu, inverse_indices_mlu, counts_mlu = torch.unique(
                input_mlu,
                sorted=sort,
                return_inverse=inverse,
                return_counts=counts,
                dim=dim,
            )
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(inverse_indices_cpu, inverse_indices_mlu.cpu(), 0)
            self.assertTensorsEqual(counts_cpu, counts_mlu.cpu(), 0)
            if input_mlu.dtype != torch.half:
                self.assertEqual(output_cpu.dtype, output_mlu.dtype)
            self.assertEqual(inverse_indices_cpu.dtype, inverse_indices_mlu.dtype)
            self.assertEqual(counts_cpu.dtype, counts_mlu.dtype)
        elif inverse:
            output_cpu, inverse_indices_cpu = torch.unique(
                input_cpu, sorted=sort, return_inverse=inverse, dim=dim
            )
            output_mlu, inverse_indices_mlu = torch.unique(
                input_mlu, sorted=sort, return_inverse=inverse, dim=dim
            )
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(inverse_indices_cpu, inverse_indices_mlu.cpu(), 0)
            if input_mlu.dtype != torch.half:
                self.assertEqual(output_cpu.dtype, output_mlu.dtype)
            self.assertEqual(inverse_indices_cpu.dtype, inverse_indices_mlu.dtype)
        elif counts:
            output_cpu, counts_cpu = torch.unique(
                input_cpu, sorted=sort, return_counts=counts, dim=dim
            )
            output_mlu, counts_mlu = torch.unique(
                input_mlu, sorted=sort, return_counts=counts, dim=dim
            )
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(counts_cpu, counts_mlu.cpu(), 0)
            if input_mlu.dtype != torch.half:
                self.assertEqual(output_cpu.dtype, output_mlu.dtype)
            self.assertEqual(counts_cpu.dtype, counts_mlu.dtype)
        else:
            output_cpu = torch.unique(input_cpu, sorted=sort, dim=dim)
            output_mlu = torch.unique(input_mlu, sorted=sort, dim=dim)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            if input_mlu.dtype != torch.half:
                self.assertEqual(output_cpu.dtype, output_mlu.dtype)

        # test _unique
        output_cpu = torch._unique(input_cpu)
        output_mlu = torch._unique(input_mlu)
        self.assertTensorsEqual(output_cpu[0], output_mlu[0].cpu(), 0)
        self.assertTensorsEqual(output_cpu[1], output_mlu[1].cpu(), 0)
        if input_mlu.dtype != torch.half:
            self.assertEqual(output_cpu[0].dtype, output_mlu[0].dtype)
            self.assertEqual(output_cpu[1].dtype, output_mlu[1].dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_unique(self):
        type_list = [torch.float, torch.int, torch.long, torch.double, torch.half]
        shape_list = [
            (64,),
            (4, 64),
            # (3, 4, 64), (100, 64, 7, 7)  # TODO: https://github.com/pytorch/pytorch/issues/101681/
        ]
        # sorted == False is error and need to be fixed. just support True as gpu now
        sort_list = [True]
        inverse_list = [True, False]
        counts_list = [True, False]
        dim_list = [None, 0, -1]
        loop_var = [
            type_list,
            shape_list,
            sort_list,
            inverse_list,
            counts_list,
            dim_list,
        ]
        for param in product(*loop_var):
            torch.manual_seed(1)
            t, shape, sort, inverse, counts, dim = param
            if not ((t == torch.half) and (dim != None)):  # TODO:[CNNLCORE-16716]
                input_cpu = torch.randint(0, 64, shape).to(t)
                input_mlu = input_cpu.to("mlu")
                if t == torch.half:
                    input_cpu = input_cpu.to(torch.float)
                self._test_unique(input_cpu, input_mlu, sort, inverse, counts, dim)

    # @unittest.skip("not test")
    @testinfo()
    def test_zero_element_unique(self):
        sort_list = [True]
        inverse_list = [True, False]
        counts_list = [True, False]
        dim_list = [None, 0]
        loop_var = [sort_list, inverse_list, counts_list, dim_list]
        for param in product(*loop_var):
            torch.manual_seed(1)
            sort, inverse, counts, dim = param
            input_cpu = torch.Tensor([])
            input_mlu = input_cpu.to("mlu")
            self._test_unique(input_cpu, input_mlu, sort, inverse, counts, dim)

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_channel_last(self):
        type_list = [torch.float, torch.int]
        shape_list = [(100, 64, 7, 7)]
        # sorted == False is error and need to be fixed. just support True as gpu now
        sort_list = [True]
        inverse_list = [True, False]
        counts_list = [True, False]
        dim_list = [None, 0]
        loop_var = [
            type_list,
            shape_list,
            sort_list,
            inverse_list,
            counts_list,
            dim_list,
        ]
        for param in product(*loop_var):
            torch.manual_seed(1)
            t, shape, sort, inverse, counts, dim = param
            input_cpu = (
                torch.randint(0, 64, shape).to(t).to(memory_format=torch.channels_last)
            )
            input_mlu = input_cpu.to("mlu")
            self._test_unique(input_cpu, input_mlu, sort, inverse, counts, dim)

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_not_dense(self):
        type_list = [torch.float, torch.int]
        shape_list = [(64,), (4, 64), (3, 4, 64), (100, 64, 7, 7)]
        # sorted == False is error and need to be fixed. just support True as gpu now
        sort_list = [True]
        inverse_list = [True, False]
        counts_list = [True, False]
        dim_list = [None, 0]
        loop_var = [
            type_list,
            shape_list,
            sort_list,
            inverse_list,
            counts_list,
            dim_list,
        ]
        for param in product(*loop_var):
            torch.manual_seed(1)
            t, shape, sort, inverse, counts, dim = param
            input_cpu = torch.randint(0, 64, shape).to(t)
            input_mlu = input_cpu.to("mlu")
            self._test_unique(input_cpu, input_mlu, sort, inverse, counts, dim)

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_exception(self):
        x_mlu = torch.arange(1, 9, dtype=torch.uint8).to("mlu")
        ref_msg = r"\"unique\" not implemented for"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.unique(x_mlu, sorted=True, return_inverse=True, return_counts=True)

        shape = (0, 2, 0, 3)
        x_mlu = torch.randint(0, 64, shape).mlu()
        ref_msg = "Number of zero sized dimensions is more than one, so unique cannot be applied "
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.unique(
                x_mlu, dim=0, sorted=True, return_inverse=True, return_counts=True
            )

        shape = (0, 2, 1, 3)
        x_mlu = torch.randint(0, 64, shape).mlu()
        ref_msg = "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.unique(
                x_mlu, dim=1, sorted=True, return_inverse=True, return_counts=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_spec(self):
        x = torch.zeros((0, 0, 3), dtype=torch.float, device="mlu")
        _, inverse, _ = torch.unique(
            x, sorted=True, return_inverse=True, return_counts=True
        )
        expected_inverse = torch.empty((0, 0, 3), dtype=torch.long, device="mlu")
        self.assertTrue(inverse.shape == expected_inverse.shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_empty(self):
        x_empty = torch.empty(5, 0, dtype=torch.float, device="mlu")
        expected_unique_empty = torch.empty(5, 0, dtype=torch.float, device="mlu")
        expected_inverse_empty = torch.tensor([], dtype=torch.long, device="mlu")
        expected_counts_empty = torch.tensor([], dtype=torch.long, device="mlu")
        # test empty tensor
        x_unique, x_inverse, x_counts = torch.unique(
            x_empty, return_inverse=True, return_counts=True, dim=1
        )
        self.assertEqual(expected_unique_empty, x_unique)
        self.assertEqual(expected_inverse_empty, x_inverse)
        self.assertEqual(expected_counts_empty, x_counts)


if __name__ == "__main__":
    unittest.main()
