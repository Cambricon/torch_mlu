from __future__ import print_function
import sys
import os
import copy
from itertools import product
import unittest
import logging

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


class TestAddcMulOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_addcmul_contiguous(self):
        data_types = [torch.half, torch.float, torch.double, torch.int]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 0, 7), (128, 64, 0, 7), (128, 64, 0, 1)),
            ((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
            ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
            ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1)),
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out_cpu = torch.addcmul(a, b, c, value=0.35)
                if data_type == torch.int:
                    ref_msg = r"MLU addcmul don't support tensor dtype Int."
                    with self.assertRaisesRegex(RuntimeError, ref_msg):
                        out_mlu = torch.addcmul(
                            a_mlu,
                            self.to_mlu_dtype(b, data_type),
                            self.to_mlu_dtype(c, data_type),
                            value=0.35,
                        )
                else:
                    out_mlu = torch.addcmul(
                        a_mlu,
                        self.to_mlu_dtype(b, data_type),
                        self.to_mlu_dtype(c, data_type),
                        value=0.35,
                    )
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_addcmul_channel_last(self):
        data_types = [torch.half, torch.float, torch.double]
        only_one_tensor_channle_last = [False, True]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
            ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
            ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1)),
        ]:
            for data_type, only_one in product(
                data_types, only_one_tensor_channle_last
            ):
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1
                if not only_one:
                    a = self.convert_to_channel_last(a)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out_cpu = torch.addcmul(a, b, c, value=0.35)

                out_mlu = torch.addcmul(
                    a_mlu,
                    self.to_mlu_dtype(b, data_type),
                    self.to_mlu_dtype(c, data_type),
                    value=0.35,
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_addcmul_not_dense(self):
        data_types = [torch.half, torch.float, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 14), (128, 64, 1, 14), (128, 64, 7, 2)),
            ((1024, 512, 3, 6), (1024, 512, 1, 6), (1024, 512, 3, 6)),
            ((512, 256, 3, 6), (1, 256, 1, 6), (512, 1, 3, 2)),
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)[
                    :, :, :, : int(shape_a[-1] / 2)
                ]
                b_mlu = self.to_mlu_dtype(copy.deepcopy(b), data_type)[
                    :, :, :, : int(shape_b[-1] / 2)
                ]
                c_mlu = self.to_mlu_dtype(copy.deepcopy(c), data_type)[
                    :, :, :, : int(shape_c[-1] / 2)
                ]

                a_cpu = a[:, :, :, : int(shape_a[-1] / 2)]
                b_cpu = b[:, :, :, : int(shape_b[-1] / 2)]
                c_cpu = c[:, :, :, : int(shape_c[-1] / 2)]

                out_cpu = torch.addcmul(a_cpu, b_cpu, c_cpu, value=0.35)

                out_mlu = torch.addcmul(a_mlu, b_mlu, c_mlu, value=0.35)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_addcmul_contiguous_(self):
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
            ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
            ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1)),
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                a_ptr = a_mlu.data_ptr()
                a.addcmul_(b, c, value=0.35)

                a_mlu.addcmul_(
                    self.to_mlu_dtype(b, data_type),
                    self.to_mlu_dtype(c, data_type),
                    value=0.35,
                )
                self.assertEqual(a_ptr, a_mlu.data_ptr())
                self.assertTensorsEqual(
                    a.float(), a_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_addcmul_channel_last_(self):
        data_types = [torch.float, torch.half, torch.double]
        only_one_tensor_channle_last = [False, True]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
            ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
            ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1)),
        ]:
            for data_type, only_one in product(
                data_types, only_one_tensor_channle_last
            ):
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1
                if not only_one:
                    a = self.convert_to_channel_last(a)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                a_ptr = a_mlu.data_ptr()
                a.addcmul_(b, c, value=0.35)

                a_mlu.addcmul_(
                    self.to_mlu_dtype(b, data_type),
                    self.to_mlu_dtype(c, data_type),
                    value=0.35,
                )
                self.assertEqual(a_ptr, a_mlu.data_ptr())
                self.assertTensorsEqual(
                    a.float(), a_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_addcmul_channel_last_not_dense(self):
        data_types = [torch.float, torch.half, torch.double]
        only_one_tensor_channle_last = [False, True]
        for shape_a, shape_b, shape_c in [
            ((2, 2, 2, 4), (2, 2, 1, 4), (2, 2, 1, 4)),
            ((2, 1, 2, 4), (2, 1, 1, 4), (2, 1, 1, 4)),
            ((128, 64, 7, 14), (128, 64, 1, 14), (128, 1, 7, 14)),
            ((1024, 512, 3, 6), (1024, 512, 1, 6), (1024, 512, 3, 6)),
            ((512, 256, 3, 6), (1, 256, 1, 6), (512, 1, 3, 6)),
        ]:
            for data_type, only_one in product(
                data_types, only_one_tensor_channle_last
            ):
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1
                if not only_one:
                    a = self.convert_to_channel_last(a)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)[
                    :, :, :, : int(shape_a[-1] / 2)
                ]
                b_mlu = self.to_mlu_dtype(copy.deepcopy(b), data_type)[
                    :, :, :, : int(shape_b[-1] / 2)
                ]
                c_mlu = self.to_mlu_dtype(copy.deepcopy(c), data_type)[
                    :, :, :, : int(shape_c[-1] / 2)
                ]
                a_cpu = a[:, :, :, : int(shape_a[-1] / 2)]
                b_cpu = b[:, :, :, : int(shape_b[-1] / 2)]
                c_cpu = c[:, :, :, : int(shape_c[-1] / 2)]
                a_cpu.addcmul_(b_cpu, c_cpu, value=0.35)
                a_mlu.addcmul_(b_mlu, c_mlu, value=0.35)
                self.assertEqual(
                    a_mlu.is_contiguous(memory_format=torch.preserve_format),
                    a_cpu.is_contiguous(memory_format=torch.preserve_format),
                )
                self.assertEqual(
                    a_mlu.is_contiguous(memory_format=torch.channels_last),
                    a_cpu.is_contiguous(memory_format=torch.channels_last),
                )
                self.assertEqual(
                    a_mlu.is_contiguous(memory_format=torch.contiguous_format),
                    a_cpu.is_contiguous(memory_format=torch.contiguous_format),
                )
                self.assertTensorsEqual(
                    a_cpu.float(), a_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_addcmul_not_dense_(self):
        data_types = [torch.float, torch.half, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 14), (128, 64, 1, 14), (128, 64, 7, 2)),
            ((1024, 512, 3, 6), (1024, 512, 1, 6), (1024, 512, 3, 6)),
            ((512, 256, 3, 6), (1, 256, 1, 6), (512, 1, 3, 2)),
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)[
                    :, :, :, : int(shape_a[-1] / 2)
                ]
                b_mlu = self.to_mlu_dtype(copy.deepcopy(b), data_type)[
                    :, :, :, : int(shape_b[-1] / 2)
                ]
                c_mlu = self.to_mlu_dtype(copy.deepcopy(c), data_type)[
                    :, :, :, : int(shape_c[-1] / 2)
                ]

                a_cpu = a[:, :, :, : int(shape_a[-1] / 2)]
                b_cpu = b[:, :, :, : int(shape_b[-1] / 2)]
                c_cpu = c[:, :, :, : int(shape_c[-1] / 2)]

                a_cpu.addcmul_(b_cpu, c_cpu, value=0.35)
                a_mlu.addcmul_(b_mlu, c_mlu, value=0.35)
                self.assertEqual(
                    a_mlu.is_contiguous(memory_format=torch.preserve_format),
                    a_cpu.is_contiguous(memory_format=torch.preserve_format),
                )
                self.assertEqual(
                    a_mlu.is_contiguous(memory_format=torch.channels_last),
                    a_cpu.is_contiguous(memory_format=torch.channels_last),
                )
                self.assertEqual(
                    a_mlu.is_contiguous(memory_format=torch.contiguous_format),
                    a_cpu.is_contiguous(memory_format=torch.contiguous_format),
                )
                self.assertTensorsEqual(
                    a_cpu.float(), a_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_addcmul_out_contiguous(self):
        data_types = [torch.half, torch.float, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
            ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
            ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1)),
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out_cpu = torch.randn(1)
                out_mlu = self.to_mlu_dtype(torch.randn(1), data_type)

                torch.addcmul(a, b, c, value=0.35, out=out_cpu)
                torch.addcmul(
                    a_mlu,
                    self.to_mlu_dtype(b, data_type),
                    self.to_mlu_dtype(c, data_type),
                    value=0.35,
                    out=out_mlu,
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_addcmul_out_channel_last(self):
        data_types = [torch.half, torch.float, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
            ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
            ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1)),
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1

                a = self.convert_to_channel_last(a)
                b = self.convert_to_channel_last(b)
                c = self.convert_to_channel_last(c)

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out_cpu = torch.randn(1)
                out_mlu = self.to_mlu_dtype(torch.randn(1), data_type)

                torch.addcmul(a, b, c, value=0.35, out=out_cpu)
                torch.addcmul(
                    a_mlu,
                    self.to_mlu_dtype(b, data_type),
                    self.to_mlu_dtype(c, data_type),
                    value=0.35,
                    out=out_mlu,
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_addcmul_out_not_dense(self):
        data_types = [torch.half, torch.float, torch.double]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 7, 14), (128, 64, 1, 14), (128, 64, 7, 14)),
            ((1024, 512, 3, 6), (1024, 512, 1, 6), (1024, 512, 3, 6)),
            ((512, 256, 3, 6), (1, 256, 1, 6), (512, 1, 3, 2)),
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)[
                    :, :, :, : int(shape_a[-1] / 2)
                ]
                b_mlu = self.to_mlu_dtype(copy.deepcopy(b), data_type)[
                    :, :, :, : int(shape_b[-1] / 2)
                ]
                c_mlu = self.to_mlu_dtype(copy.deepcopy(c), data_type)[
                    :, :, :, : int(shape_c[-1] / 2)
                ]
                a_cpu = a[:, :, :, : int(shape_a[-1] / 2)]
                b_cpu = b[:, :, :, : int(shape_b[-1] / 2)]
                c_cpu = c[:, :, :, : int(shape_c[-1] / 2)]
                out_cpu = torch.randn(1)
                out_mlu = self.to_mlu_dtype(torch.randn(1), data_type)

                torch.addcmul(a_cpu, b_cpu, c_cpu, value=0.35, out=out_cpu)
                torch.addcmul(a_mlu, b_mlu, c_mlu, value=0.35, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("46GB")
    def test_addcmul_large(self):
        data_types = [torch.half]
        for shape_a, shape_b, shape_c in [
            ((5, 1024, 1024, 1024), (5, 1024, 1, 1024), (5, 1024, 1024, 1))
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1
                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out_cpu = torch.addcmul(a, b, c, value=0.35)
                out_mlu = torch.addcmul(
                    a_mlu,
                    self.to_mlu_dtype(b, data_type),
                    self.to_mlu_dtype(c, data_type),
                    value=0.35,
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_addcmul_contiguous_bfloat16(self):
        data_types = [torch.bfloat16]
        for shape_a, shape_b, shape_c in [
            ((128, 64, 0, 7), (128, 64, 0, 7), (128, 64, 0, 1)),
            ((128, 64, 7, 7), (128, 64, 1, 7), (128, 64, 7, 1)),
            ((1024, 512, 3, 3), (1024, 512, 1, 3), (1024, 512, 3, 3)),
            ((512, 256, 3, 3), (1, 256, 1, 3), (512, 1, 3, 1)),
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1

                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out_cpu = torch.addcmul(a, b, c, value=0.35)
                out_mlu = torch.addcmul(
                    a_mlu,
                    self.to_mlu_dtype(b, data_type),
                    self.to_mlu_dtype(c, data_type),
                    value=0.35,
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )
        # test bfloat16 backward
        a = torch.rand((128, 64, 7, 7), dtype=torch.bfloat16).float()
        b = torch.rand((128, 64, 1, 7), dtype=torch.bfloat16).float()
        c = torch.rand((128, 64, 7, 1), dtype=torch.bfloat16).float() + 1
        a_cpu = torch.nn.Parameter(a)
        b_cpu = torch.nn.Parameter(b)
        c_cpu = torch.nn.Parameter(c)
        a_mlu = torch.nn.Parameter(a.mlu().bfloat16())
        b_mlu = torch.nn.Parameter(b.mlu().bfloat16())
        c_mlu = torch.nn.Parameter(c.mlu().bfloat16())
        out_cpu = torch.addcmul(a_cpu, b_cpu, c_cpu, value=0.35)
        out_mlu = torch.addcmul(a_mlu, b_mlu, c_mlu, value=0.35)
        grad = torch.randn_like(out_cpu)
        grad_mlu = grad.mlu().bfloat16()
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
        self.assertTensorsEqual(
            a_cpu.grad, a_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            b_cpu.grad, b_mlu.grad.cpu().float(), 0.005, use_MSE=True
        )
        self.assertTensorsEqual(
            c_cpu.grad, c_mlu.grad.cpu().float(), 0.005, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @largeTensorTest("46GB")
    def test_addcmul_large_bfloat16(self):
        data_types = [torch.bfloat16]
        for shape_a, shape_b, shape_c in [
            ((5, 1024, 1024, 1024), (5, 1024, 1, 1024), (5, 1024, 1024, 1))
        ]:
            for data_type in data_types:
                a = torch.rand(shape_a, dtype=torch.float)
                b = torch.rand(shape_b, dtype=torch.float)
                c = torch.rand(shape_c, dtype=torch.float) + 1
                a_mlu = self.to_mlu_dtype(copy.deepcopy(a), data_type)
                out_cpu = torch.addcmul(a, b, c, value=0.35)
                out_mlu = torch.addcmul(
                    a_mlu,
                    self.to_mlu_dtype(b, data_type),
                    self.to_mlu_dtype(c, data_type),
                    value=0.35,
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
