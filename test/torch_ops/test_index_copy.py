import sys
import logging
from itertools import product
import os
import copy
import unittest
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestIndexCopyOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_index_copy_(self):
        # index_copy do not suppor double because cnnl_fill do not support double
        dtype_list = [torch.bool, torch.float, torch.long, torch.int16, torch.int32]
        shape_list = ((5, 4, 3), (5, 4), (5,), ())
        for dtype, shape in product(dtype_list, shape_list):
            dim_size = len(shape)
            x_cpu = torch.rand(shape).to(dtype)
            t_cpu = torch.rand(shape).to(dtype)
            x_mlu = copy.deepcopy(x_cpu).to("mlu")
            t_mlu = copy.deepcopy(t_cpu).to("mlu")
            if dim_size == 0:  # test of scalar
                index_cpu = torch.tensor(0)
                index_mlu = copy.deepcopy(index_cpu).to("mlu")
                x_cpu.index_copy_(0, index_cpu, t_cpu)
                ori_ptr = x_mlu.data_ptr()
                x_mlu.index_copy_(0, index_mlu, t_mlu)
                self.assertTensorsEqual(
                    x_cpu.float(), x_mlu.cpu().float(), 0.0, use_MSE=True
                )
                self.assertEqual(ori_ptr, x_mlu.data_ptr())
            else:
                for dim in range(dim_size):
                    index_cpu = torch.randint(0, shape[dim], (shape[dim],))
                    index_mlu = copy.deepcopy(index_cpu).to("mlu")
                    x_cpu.index_copy_(dim, index_cpu, t_cpu)
                    ori_ptr = x_mlu.data_ptr()
                    x_mlu.index_copy_(dim, index_mlu, t_mlu)
                    self.assertTensorsEqual(
                        x_cpu.float(), x_mlu.cpu().float(), 0.0, use_MSE=True
                    )
                    self.assertEqual(ori_ptr, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_index_copy(self):
        dtype_list = [
            torch.bool,
            torch.float,
            torch.double,
            torch.long,
            torch.int16,
            torch.int32,
        ]
        shape_list = ((5, 4, 3), (5, 4), (5,), ())
        for dtype, shape in product(dtype_list, shape_list):
            dim_size = len(shape)
            x_cpu = torch.rand(shape).to(dtype)
            t_cpu = torch.rand(shape).to(dtype)
            x_mlu = copy.deepcopy(x_cpu).to("mlu")
            t_mlu = copy.deepcopy(t_cpu).to("mlu")
            for dim in range(dim_size):
                index_cpu = torch.randint(0, shape[dim], (shape[dim],))
                index_mlu = copy.deepcopy(index_cpu).to("mlu")
                out_cpu = torch.index_copy(x_cpu, dim, index_cpu, t_cpu)
                out_mlu = torch.index_copy(x_mlu, dim, index_mlu, t_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_index_copy_neg_dim(self):
        dtype_list = [
            torch.bool,
            torch.float,
            torch.double,
            torch.long,
            torch.int16,
            torch.int32,
        ]
        shape_list = ((5, 4, 3), (5, 4), (5,), ())
        for dtype, shape in product(dtype_list, shape_list):
            dim_size = len(shape)
            x_cpu = torch.rand(shape).to(dtype)
            t_cpu = torch.rand(shape).to(dtype)
            x_mlu = copy.deepcopy(x_cpu).to("mlu")
            t_mlu = copy.deepcopy(t_cpu).to("mlu")
            for dim in range(dim_size):
                dim_neg = -dim
                if dim_neg < 0:
                    index_cpu = torch.randint(
                        0, shape[dim_neg + dim_size], (shape[dim_neg + dim_size],)
                    )
                else:
                    index_cpu = torch.randint(0, shape[dim_neg], (shape[dim_neg],))
                index_mlu = copy.deepcopy(index_cpu).to("mlu")
                out_cpu = torch.index_copy(x_cpu, dim_neg, index_cpu, t_cpu)
                out_mlu = torch.index_copy(x_mlu, dim_neg, index_mlu, t_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_index_copy_exception(self):
        device = "mlu"
        dtype = torch.complex64
        src = torch.randn((3, 4, 5), dtype=dtype, device=device)
        idx = torch.randint(high=3, size=(3,), device=device)
        dest = torch.randn((5, 4, 5), dtype=dtype, device=device)
        ref_msg = "Complex type input is not supported yet!"
        with self.assertRaises(RuntimeError) as info:
            dest.index_copy_(0, idx, src)
        self.assertEqual(info.exception.args[0], ref_msg)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("44GB")
    def test_index_copy_large(self):
        # FIXME(huangqipeng):CTR-4537, indexCopy of cpu is nondeterministic by default.
        torch.use_deterministic_algorithms(True)
        dtype_list = [torch.half]
        shape_list = ((5, 1024, 1024, 1024),)
        for dtype, shape in product(dtype_list, shape_list):
            dim_size = len(shape)
            x_cpu = torch.rand(shape).to(dtype)
            t_cpu = torch.rand(shape).to(dtype)
            x_mlu = copy.deepcopy(x_cpu).to("mlu")
            t_mlu = copy.deepcopy(t_cpu).to("mlu")
            # TODO(huangqipeng): run op in other dim, will be timeout
            # for dim in range(dim_size):
            dim = 0
            index_cpu = torch.randint(0, shape[dim], (shape[dim],))
            index_mlu = copy.deepcopy(index_cpu).to("mlu")
            out_cpu = torch.index_copy(x_cpu, dim, index_cpu, t_cpu)
            out_mlu = torch.index_copy(x_mlu, dim, index_mlu, t_mlu)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_index_copy_bfloat16(self):
        dtype = torch.bfloat16
        shape = [5, 4, 3]
        dim = 1
        x_cpu = torch.rand(shape).to(dtype)
        t_cpu = torch.rand(shape).to(dtype)
        x_mlu = copy.deepcopy(x_cpu).to("mlu")
        t_mlu = copy.deepcopy(t_cpu).to("mlu")
        index_cpu = torch.randint(0, shape[dim], (shape[dim],))
        index_mlu = copy.deepcopy(index_cpu).to("mlu")
        out_cpu = torch.index_copy(x_cpu, dim, index_cpu, t_cpu)
        out_mlu = torch.index_copy(x_mlu, dim, index_mlu, t_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )


if __name__ == "__main__":
    run_tests()
