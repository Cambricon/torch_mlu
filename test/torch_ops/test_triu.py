import sys
import os
import copy
import unittest
import logging
from itertools import product
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestTriuOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_torch_triu(self):
        shape_list = [
            (3, 3),
            (5, 8),
            (8, 10),
            (18, 25),
            (2, 3, 4),
            (2, 3, 4, 5),
            (2, 3, 4, 5, 6),
        ]
        for shape in shape_list:
            min_ = min(shape)
            for diagonal in range(-1 * min_, min_):
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.triu(x, diagonal)
                out_mlu = torch.triu(self.to_mlu(x), diagonal)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_torch_triu_dtype(self):
        dtype_list = [
            torch.float,
            torch.half,
            torch.int,
            torch.short,
            torch.int8,
            torch.bool,
            torch.double,
            torch.long,
            torch.bfloat16,
        ]
        shape_list = [(3, 3)]
        for shape, dtype in product(shape_list, dtype_list):
            min_ = min(shape)
            for diagonal in range(-1 * min_, min_):
                x = torch.randn(shape).to(dtype=dtype)
                out_cpu = torch.triu(x, diagonal)
                out_mlu = torch.triu(self.to_mlu(x), diagonal)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_torch_triu_not_dense(self):
        shape_list = [(2, 3, 24, 30), (1, 1, 1, 30)]
        for shape in shape_list:
            min_ = min(shape)
            for diagonal in range(-1 * min_, min_):
                x = torch.randn(shape, dtype=torch.float)[:, :, :, :15]
                out_cpu = torch.triu(x, diagonal)
                out_mlu = torch.triu(self.to_mlu(x), diagonal)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_tensor_triu_inplace(self):
        shape_list = [(3, 3), (5, 8), (8, 10), (18, 25)]
        for shape in shape_list:
            min_ = min(shape)
            for diagonal in range(-1 * min_, min_):
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu(copy.deepcopy(x))
                x_mlu_data_ptr = x_mlu.data_ptr()
                x.triu_(diagonal)
                x_mlu.triu_(diagonal)
                self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_tensor_triu_not_contiguous_inplace(self):
        shape_list = [(3, 3), (5, 8), (8, 10), (18, 25)]
        contiguous = [True, False]
        for shape in shape_list:
            min_ = min(shape)
            for diagonal in range(0, min_):
                for con in contiguous:
                    x = torch.rand(shape, dtype=torch.float)
                    if con is True:
                        x = self.get_not_contiguous_tensor(x)
                    x_mlu = self.to_mlu(copy.deepcopy(x))
                    x_mlu_data_ptr = x_mlu.data_ptr()
                    x.triu_(diagonal)
                    x_mlu.triu_(diagonal)
                    self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_triu_out(self):
        shape_list = [(3, 3), (5, 8), (8, 10), (18, 25)]
        for shape in shape_list:
            min_ = min(shape)
            for diagonal in range(-1 * min_, min_):
                x = torch.randn(shape, dtype=torch.float)
                out_mlu = self.to_mlu(torch.ones(shape, dtype=torch.float))
                out_mlu_data_ptr = out_mlu.data_ptr()
                out_cpu = torch.triu(x, diagonal)
                torch.triu(self.to_mlu(x), diagonal, out=out_mlu)
                self.assertEqual(out_mlu_data_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_triu_exception(self):
        input = torch.randn(3, 4, dtype=torch.complex32).to("mlu")
        ref_msg = "\"tri_out_mlu_impl\" not implemented for 'ComplexHalf'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.triu(input)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("45GB")
    def test_triu_large(self):
        shape = [5, 1024, 1024, 1024]
        diagonal = 4
        x = torch.randn(shape, dtype=torch.float)
        out_cpu = torch.triu(x, diagonal)
        out_mlu = torch.triu(self.to_mlu(x), diagonal)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
