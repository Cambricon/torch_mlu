from __future__ import print_function

import sys
import os
import unittest
import logging
import copy
import random as rd

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)

logging.basicConfig(level=logging.DEBUG)


class TestBitwiseXorOps(TestCase):
    def _generate_tensor(self, shape, dtype):
        if dtype == torch.bool:
            out = torch.randint(2, shape).type(dtype)
        elif dtype == torch.uint8:
            out = torch.randint(16777216, shape).type(dtype)
        else:
            out = torch.randint(-16777216, 16777216, shape).type(dtype)
        return out

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_xor(self):
        dtype_lst = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        for dtype in dtype_lst:
            for shape1, shape2 in [
                (
                    (
                        1,
                        2,
                    ),
                    (1, 1),
                ),
                ((2, 30, 80), (2, 30, 80)),
                ((3, 20), (3, 20)),
                ((3, 273), (1, 273)),
                ((1, 273), (3, 273)),
                ((2, 2, 4, 2), (1, 2)),
                ((1, 2), (2, 2, 4, 2)),
                ((1, 3, 224, 224), (1, 1, 1)),
                ((1, 1, 1), (1, 3, 224, 224)),
                ((1, 1, 3), (1, 224, 3)),
                ((1, 3, 224), (1, 3, 1)),
                ((1, 3, 1), (1, 3, 224)),
                ((1, 3, 224, 224), (1, 1)),
            ]:
                a = self._generate_tensor(shape1, dtype)
                b = self._generate_tensor(shape2, dtype)
                result_cpu = torch.bitwise_xor(a, b)
                result_mlu = torch.bitwise_xor(self.to_device(a), self.to_device(b))
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

                # no dense
                x = self._generate_tensor(shape1, dtype)
                y = self._generate_tensor(shape2, dtype)
                x_cpu = x[..., :2]
                y_cpu = y[..., :2]
                x_mlu = self.to_device(copy.deepcopy(x))[..., :2]
                y_mlu = self.to_device(copy.deepcopy(y))[..., :2]
                result_cpu = x_cpu.bitwise_xor(y_cpu)
                result_mlu = x_mlu.bitwise_xor(y_mlu)
                self.assertTensorsEqual(
                    result_cpu.float(), result_mlu.cpu().float(), 0.003, use_MSE=True
                )

                # scalar
                b_scalar = 2
                result_cpu = torch.bitwise_xor(a, b_scalar)
                result_mlu = torch.bitwise_xor(self.to_device(a), b_scalar)
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

                # channels last
                if a.dim() == 4:
                    x = self._generate_tensor(shape1, dtype).to(
                        memory_format=torch.channels_last
                    )
                    y = self._generate_tensor(shape2, dtype)
                    x_mlu = self.to_device(copy.deepcopy(x))
                    y_mlu = self.to_device(copy.deepcopy(y))
                    out_cpu = torch.bitwise_xor(x, y)

                    out_mlu = torch.bitwise_xor(x_mlu, y_mlu)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_xor_inplace(self):
        dtype_lst = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        for dtype in dtype_lst:
            for shape1, shape2 in [
                ((1, 3, 224, 224), (1, 3, 224, 1)),
                ((2, 30, 80), (2, 30, 80)),
                ((3, 20), (3, 20)),
                ((3, 273), (1, 273)),
                ((2, 2, 4, 2), (1, 2)),
                ((1, 3, 224, 224), (1, 1, 1)),
                ((1, 3, 224), (1, 3, 1)),
                ((1, 3, 224, 224), (1, 1)),
            ]:
                a = self._generate_tensor(shape1, dtype)
                b = self._generate_tensor(shape2, dtype)
                a_copy = copy.deepcopy(a)
                a.bitwise_xor_(b)
                a_mlu = self.to_device(a_copy)
                raw_ptr = a_mlu.data_ptr()
                a_mlu.bitwise_xor_(self.to_device(b))
                cur_ptr = a_mlu.data_ptr()
                self.assertEqual(raw_ptr, cur_ptr)
                self.assertTensorsEqual(a, a_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_xor_out(self):
        device = "mlu"

        a = torch.randint(-16776321, 16776565, (1, 2, 3, 4))
        b = torch.randint(-16776321, 16776565, (4, 2, 3, 4))
        c = torch.empty(0, dtype=torch.int32)
        c_mlu = torch.empty(0, dtype=torch.int32).to(device)
        torch.bitwise_xor(a, b, out=c)
        torch.bitwise_xor(a.to(device), b.to(device), out=c_mlu)
        self.assertTensorsEqual(c, c_mlu.cpu(), 0)

        # scalar
        a = torch.randint(-16776321, 16776565, (1, 2, 3, 4))
        b_scalar = 6654734
        c = torch.empty(0, dtype=torch.int32)
        c_mlu = torch.empty(0, dtype=torch.int32).to(device)
        torch.bitwise_xor(a, b_scalar, out=c)
        torch.bitwise_xor(a.to(device), b_scalar, out=c_mlu)
        self.assertTensorsEqual(c, c_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_xor_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.randint(2, shape_list[i]).type(torch.bool)
            y = torch.randint(2, shape_list[i]).type(torch.bool)
            out = torch.randint(2, shape_list[i]).type(torch.bool)
            x_mlu = copy.deepcopy(x).mlu()
            y_mlu = copy.deepcopy(y).mlu()
            out_mlu = copy.deepcopy(out).mlu()
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
            torch.bitwise_xor(x, y, out=out)
            torch.bitwise_xor(x_mlu, y_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_xor_exception(self):
        ref_msg = r"MLU bitwise_xor don't support tensor dtype Float."
        a = torch.arange(24, dtype=torch.float).reshape(1, 2, 3, 4)
        b = torch.arange(96).reshape(4, 2, 3, -1).float()
        a_mlu = self.to_device(a)
        b_mlu = self.to_device(b)
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a_mlu.bitwise_xor(b_mlu)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_bitwise_xor_large(self):
        dtype_lst = (torch.int8,)
        for dtype in dtype_lst:
            for shape1, shape2 in [((5, 1024, 1024, 1024), (5, 1024, 1024, 1024))]:
                a = self._generate_tensor(shape1, dtype)
                b = self._generate_tensor(shape2, dtype)
                result_cpu = torch.bitwise_xor(a, b)
                result_mlu = torch.bitwise_xor(self.to_device(a), self.to_device(b))
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)


if __name__ == "__main__":
    run_tests()
