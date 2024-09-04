from __future__ import print_function
import logging
import sys
import os
import unittest
import torch
from itertools import product

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    TEST_BFLOAT16,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

torch.manual_seed(6503)


class TestBinCountOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_all_type_bincount(self):
        input_size = (5000,)
        weight_type_list = [
            torch.half,
            torch.float,
            torch.double,
            torch.int,
            torch.short,
            torch.int8,
            torch.bool,
            torch.long,
            torch.uint8,
        ]
        input_type_list = [torch.int, torch.short, torch.int8, torch.long, torch.uint8]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for input_type, weight_type, func in product(
            input_type_list, weight_type_list, func_list
        ):
            input = torch.randint(50, input_size, dtype=input_type, device="cpu")
            weight = torch.testing.make_tensor(
                input_size, dtype=weight_type, device="cpu"
            )
            input_mlu = input.mlu()
            weight_mlu = weight.mlu()
            out_cpu = torch.bincount(func(input), func(weight))
            out_mlu = torch.bincount(func(input_mlu), func(weight_mlu))
            self.assertTrue(out_cpu.dtype, out_mlu.dtype)
            self.assertTrue(out_cpu.size(), out_mlu.size())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
            # test without weight
            out_cpu = torch.bincount(func(input))
            out_mlu = torch.bincount(func(input_mlu))
            self.assertTrue(out_cpu.dtype, out_mlu.dtype)
            self.assertTrue(out_cpu.size(), out_mlu.size())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_zero_element_bincount(self):
        input_size = (0,)
        input = torch.randint(50, input_size, dtype=torch.int, device="cpu")
        weight = torch.testing.make_tensor(input_size, dtype=torch.float, device="cpu")
        input_mlu = input.mlu()
        weight_mlu = weight.mlu()
        out_cpu = torch.bincount(input, weight)
        out_mlu = torch.bincount(input_mlu, weight_mlu)
        self.assertTrue(out_cpu.dtype, out_mlu.dtype)
        self.assertTrue(out_cpu.size(), out_mlu.size())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bincount_exception(self):
        input_size = (1, 10)
        input = torch.randint(50, input_size, dtype=torch.int, device="cpu")
        weight = torch.testing.make_tensor(input_size, dtype=torch.float, device="cpu")
        input_mlu = input.mlu()
        weight_mlu = weight.mlu()
        ref_msg = r"bincount only supports 1-d non-negative integral inputs."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out_mlu = torch.bincount(input_mlu, weight_mlu)
        input = torch.randint(50, (1,), dtype=torch.int, device="cpu")
        input_mlu = input.mlu()
        ref_msg = r"weights should be 1-d and have the same length as input."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out_mlu = torch.bincount(input_mlu, weight_mlu)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("21GB")
    def test_bincount_large(self):
        input_size = (4 * 1024 * 1024 * 1025,)
        input = torch.randint(50, input_size, dtype=torch.int, device="cpu")
        input_mlu = input.mlu()
        out_cpu = torch.bincount(input)
        out_mlu = torch.bincount(input_mlu)
        self.assertTrue(out_cpu.dtype, out_mlu.dtype)
        self.assertTrue(out_cpu.size(), out_mlu.size())
        # TODO: temporarily set err from 0.0 -> 0.003, change back after CNNLCORE-18812 is solved.
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
