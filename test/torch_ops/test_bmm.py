from __future__ import print_function

import sys
import os
import copy
import logging
import unittest
from itertools import product
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()


class TestBmmOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_bmm(self):
        dtype_list = [(torch.float, 3e-3), (torch.float16, 3e-3)]
        shape_ab_list = [
            ((3, 4, 5), (3, 5, 6)),
            ((4, 3, 5), (4, 5, 6)),
            ((256, 10, 64), (256, 64, 10)),
            ((256, 10, 10), (256, 10, 64)),
            ((0, 4, 5), (0, 5, 6)),
            ((3, 0, 4), (3, 4, 5)),
            ((3, 4, 0), (3, 0, 6)),
            ((3, 4, 5), (3, 5, 0)),
        ]
        mode_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for (data_type, err), (shape_a, shape_b), mode in product(
            dtype_list, shape_ab_list, mode_list
        ):
            a = torch.randn(shape_a, dtype=torch.float)
            b = torch.randn(shape_b, dtype=torch.float)
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type))
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type))
            out_cpu = torch.bmm(a, b)
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bmm_trans(self):
        dtype_list = [(torch.float, 3e-3), (torch.float16, 3e-3)]
        shape_ab_list = [
            ((3, 4, 5), (3, 5, 6)),
            ((4, 3, 5), (4, 5, 6)),
            ((256, 10, 64), (256, 64, 10)),
            ((256, 10, 10), (256, 10, 64)),
        ]
        mode_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for (data_type, err), (shape_a, shape_b), mode in product(
            dtype_list, shape_ab_list, mode_list
        ):
            # trans self
            a = torch.randn(shape_a, dtype=torch.float).transpose(1, 2).contiguous()
            b = torch.randn(shape_b, dtype=torch.float)
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type)).transpose(1, 2)
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type))
            out_cpu = torch.bmm(a.transpose(1, 2), b)
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            # trans other
            a = torch.randn(shape_a, dtype=torch.float)
            b = torch.randn(shape_b, dtype=torch.float).transpose(1, 2).contiguous()
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type))
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type)).transpose(1, 2)
            out_cpu = torch.bmm(a, b.transpose(1, 2))
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            # trans self and other
            a = torch.randn(shape_a, dtype=torch.float).transpose(1, 2).contiguous()
            b = torch.randn(shape_b, dtype=torch.float).transpose(1, 2).contiguous()
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type)).transpose(1, 2)
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type)).transpose(1, 2)
            out_cpu = torch.bmm(a.transpose(1, 2), b.transpose(1, 2))
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bmm_permute(self):
        dtype_list = [(torch.float, 3e-3), (torch.float16, 3e-3)]
        shape_ab_list = [
            ((3, 4, 5), (3, 5, 6)),
            ((4, 3, 5), (4, 5, 6)),
            ((256, 10, 64), (256, 64, 10)),
            ((256, 10, 10), (256, 10, 64)),
            ((23, 23, 23), (23, 23, 23)),
        ]
        mode_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for (data_type, err), (shape_a, shape_b), mode in product(
            dtype_list, shape_ab_list, mode_list
        ):
            # permute self
            a = torch.randn(shape_a, dtype=torch.float).permute(1, 2, 0).contiguous()
            b = torch.randn(shape_b, dtype=torch.float)
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type)).permute(
                2, 0, 1
            )
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type))
            out_cpu = torch.bmm(a.permute(2, 0, 1), b)
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            # permute other
            a = torch.randn(shape_a, dtype=torch.float)
            b = torch.randn(shape_b, dtype=torch.float).permute(2, 1, 0).contiguous()
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type))
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type)).permute(
                2, 1, 0
            )
            out_cpu = torch.bmm(a, b.permute(2, 1, 0))
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            # permute self and other
            a = torch.randn(shape_a, dtype=torch.float).permute(1, 0, 2).contiguous()
            b = torch.randn(shape_b, dtype=torch.float).permute(2, 0, 1).contiguous()
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type)).permute(
                1, 0, 2
            )
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type)).permute(
                1, 2, 0
            )
            out_cpu = torch.bmm(a.permute(1, 0, 2), b.permute(1, 2, 0))
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bmm_out(self):
        dtype_list = [(torch.float, 3e-3), (torch.double, 3e-3), (torch.float16, 3e-3)]
        shape_ab_list = [((3, 4, 5), (3, 5, 6))]
        shape_out_list = [(3, 4, 5), (3, 5, 6)]
        mode_list = [self.to_non_dense, lambda x: x]
        for (data_type, err), (shape_a, shape_b), shape_o, mode, mode_o in product(
            dtype_list, shape_ab_list, shape_out_list, mode_list, mode_list
        ):
            a = torch.randn(shape_a, dtype=torch.float)
            b = torch.randn(shape_b, dtype=torch.float)
            o = torch.randn(shape_o, dtype=torch.float)
            a_mlu = mode(self.to_mlu_dtype(copy.deepcopy(a), data_type))
            b_mlu = mode(self.to_mlu_dtype(copy.deepcopy(b), data_type))
            o_mlu = mode_o(self.to_mlu_dtype(copy.deepcopy(o), data_type))
            out_cpu = torch.bmm(a, b, out=o)
            out_mlu = torch.bmm(a_mlu, b_mlu, out=o_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bmm_exception(self):
        batch1 = torch.randn((8, 4, 5), dtype=torch.half)
        batch2 = torch.randn((8, 5, 6), dtype=torch.float)
        ref_msg = "expected scalar type Float but found Half"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.bmm(batch1.mlu(), batch2.mlu())

        batch1 = torch.randn((8, 4, 5)).int()
        batch2 = torch.randn((8, 5, 6)).int()
        ref_msg = f"\"MLU bmm\" not implemented for 'Int'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.bmm(batch1.mlu(), batch2.mlu())

    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_bmm_bfloat16(self):
        # CPU side accumulate matmul using bfloat16, but MLU side and GPU side
        # is using float.
        a = torch.randn((3, 4, 5), dtype=torch.bfloat16).float()
        b = torch.randn((3, 5, 6), dtype=torch.bfloat16).float()
        a_cpu = torch.nn.Parameter(a)
        b_cpu = torch.nn.Parameter(b)
        a_mlu = torch.nn.Parameter(a.mlu().bfloat16())
        b_mlu = torch.nn.Parameter(b.mlu().bfloat16())
        out_cpu = torch.bmm(a_cpu, b_cpu)
        out_mlu = torch.bmm(a_mlu, b_mlu)
        grad = torch.randn(out_cpu.shape).bfloat16().float()
        grad_mlu = grad.mlu().bfloat16()
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            a_cpu.grad.float(), a_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            b_cpu.grad.float(), b_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    def test_bmm_large(self):
        dtype_list = [(torch.float16, 3e-3)]
        shape_ab_list = [((1024, 1024, 4096), (1024, 4096, 6))]
        for (data_type, err), (shape_a, shape_b) in product(dtype_list, shape_ab_list):
            a = torch.randn(shape_a, dtype=torch.float)
            b = torch.randn(shape_b, dtype=torch.float)
            a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
            b_mlu = self.to_mlu_dtype(copy.deepcopy(b), data_type)
            out_cpu = torch.bmm(a, b)
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    def test_bmm_large_exceptions(self):
        # [cnnlBatchMatMulBCast]: the max dim for A matrix should be less than 2147483648(2^31)
        ref_msg = r"CNNL error: CNNL_STATUS_BAD_PARAM"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            data_type, err = torch.float16, 3e-3
            shape_a, shape_b = (1, 1024 * 1024 * 4096, 2), (1, 2, 2)
            a = torch.randn(shape_a, dtype=torch.float)
            b = torch.randn(shape_b, dtype=torch.float)
            a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
            b_mlu = self.to_mlu_dtype(copy.deepcopy(b), data_type)
            out_cpu = torch.bmm(a, b)
            out_mlu = torch.bmm(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)


if __name__ == "__main__":
    run_tests()
