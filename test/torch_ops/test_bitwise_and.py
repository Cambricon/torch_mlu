from __future__ import print_function

import sys
import os
import unittest
import logging
import random as rd
import copy
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


class TestAndOp(TestCase):
    def _generate_tensor(self, shape, dtype):
        if dtype == torch.bool:
            out = torch.randint(2, shape).type(dtype)
        elif dtype == torch.uint8:
            out = torch.randint(16777216, shape).type(dtype)
        else:
            out = torch.randint(-16777216, 16777216, shape).type(dtype)
        return out

    def _generate_scalar(self, dtype):
        if dtype == torch.bool:
            out = rd.choice([True, False])
        elif dtype == torch.uint8:
            out = rd.randint(0, 16777216)
        else:
            out = rd.randint(-16777216, 16777216)
        return out

    # @unittest.skip("not test")
    @testinfo()
    def test_and(self):
        dtype_lst = [
            (torch.bool, torch.bool),
            (torch.uint8, torch.uint8),
            (torch.int16, torch.int16),
            (torch.int32, torch.int32),
            (torch.long, torch.long),
            (torch.int32, torch.int8),
            (torch.int32, torch.int16),
            (torch.int8, torch.int8),
        ]
        for dtype1, dtype2 in dtype_lst:
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
                ((1, 3, 1), (1, 3, 224)),
                ((1, 3, 224), (1, 3, 1)),
                ((1, 1, 3), (1, 224, 3)),
                ((1, 3, 224, 224), (1, 1)),
            ]:
                x = self._generate_tensor(shape1, dtype1)
                y = self._generate_tensor(shape2, dtype2)
                out_cpu = torch.bitwise_and(x, y)
                out_mlu = torch.bitwise_and(self.to_device(x), self.to_device(y))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

                out_cpu = x & y
                out_mlu = self.to_device(x) & self.to_device(y)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

                # test no dense
                x = self._generate_tensor(shape1, dtype1)[..., :2]
                y = self._generate_tensor(shape2, dtype2)[..., :2]
                x_mlu = self.to_device(copy.deepcopy(x))
                y_mlu = self.to_device(copy.deepcopy(y))
                out_cpu = x & y
                out_mlu = x_mlu & y_mlu
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

                # test scalar input
                if dtype1 in [torch.uint8, torch.int16, torch.int8]:
                    continue
                input1 = self._generate_tensor(shape1, dtype1)
                input3 = self._generate_scalar(dtype2)
                result_cpu = input1 & input3
                result_mlu = self.to_device(input1) & input3
                self.assertTensorsEqual(
                    result_cpu.float(), result_mlu.cpu().float(), 0.0, use_MSE=True
                )

                if x.dim() == 4:
                    # test channels last
                    x = self._generate_tensor(shape1, dtype1).to(
                        memory_format=torch.channels_last
                    )
                    y = self._generate_tensor(shape2, dtype2)
                    x_mlu = self.to_device(copy.deepcopy(x))
                    y_mlu = self.to_device(copy.deepcopy(y))
                    out_cpu = x.bitwise_and(y)
                    out_mlu = x_mlu.bitwise_and(y_mlu)
                    self.assertTrue(x.stride() == x_mlu.stride())
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                    )

        input1 = self._generate_tensor((1, 20, 30, 40), torch.int)
        input3 = self._generate_scalar(torch.int)
        out_big = self._generate_tensor((5, 20, 30, 40), dtype=torch.int)
        out_big_mlu = self.to_device(out_big)
        torch.bitwise_and(input1, input3, out=out_big)
        torch.bitwise_and(
            self.to_device(input1), self.to_device(input3), out=out_big_mlu
        )
        self.assertTensorsEqual(
            out_big.float(), out_big_mlu.cpu().float(), 0, use_MSE=True
        )

        input1 = self._generate_tensor((1, 20, 30, 40), torch.int)
        input3 = self._generate_tensor((4, 20, 30, 40), torch.int)
        out_big = self._generate_tensor((5, 20, 30, 40), dtype=torch.int)
        out_big_mlu = self.to_device(out_big)
        torch.bitwise_and(input1, input3, out=out_big)
        torch.bitwise_and(
            self.to_device(input1), self.to_device(input3), out=out_big_mlu
        )
        self.assertTensorsEqual(
            out_big.float(), out_big_mlu.cpu().float(), 0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_and_inplace(self):
        dtype_lst = [
            (torch.bool, torch.bool),
            (torch.uint8, torch.uint8),
            (torch.int16, torch.int16),
            (torch.int32, torch.int32),
            (torch.long, torch.long),
            (torch.int32, torch.int8),
            (torch.int32, torch.int16),
            (torch.int8, torch.int8),
        ]
        for dtype1, dtype2 in dtype_lst:
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
                x = self._generate_tensor(shape1, dtype1)
                y = self._generate_tensor(shape2, dtype2)
                x_copy = copy.deepcopy(x)
                x_copy_mlu = self.to_device(x_copy)
                raw_ptr = x_copy_mlu.data_ptr()
                out_cpu = x.bitwise_and_(y)
                x_copy_mlu.bitwise_and_(self.to_device(y))
                self.assertTensorsEqual(
                    out_cpu.float(), x_copy_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertEqual(raw_ptr, x_copy_mlu.data_ptr())

                if dtype1 == torch.bool or dtype2 == torch.bool:
                    x_copy = x_copy.int()
                    y = y.int()
                x_copy_mlu = self.to_device(x_copy)
                raw_ptr = x_copy_mlu.data_ptr()
                x_copy &= y
                x_copy_mlu &= self.to_device(y)
                self.assertTensorsEqual(
                    x_copy.float(), x_copy_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertEqual(raw_ptr, x_copy_mlu.data_ptr())

                # test scalar input
                # FIXME(guyi): scalar computation are not supported for some types
                # will be fixed in future.
                if dtype1 in [torch.uint8, torch.int16, torch.int8]:
                    continue
                input1 = self._generate_tensor(shape1, dtype1)
                input3 = self._generate_scalar(dtype2)
                input1_copy = copy.deepcopy(input1)
                input1.bitwise_and_(input3)
                input1_mlu = self.to_device(input1_copy)
                raw_ptr = input1_mlu.data_ptr()
                input1_mlu.bitwise_and_(input3)
                cur_ptr = input1_mlu.data_ptr()
                self.assertEqual(raw_ptr, cur_ptr)
                self.assertTensorsEqual(input1, input1_mlu.cpu(), 0)

                if x.dim() == 4:
                    # test no dense
                    x = self._generate_tensor(shape1, dtype1)[..., :2]
                    y = self._generate_tensor(shape2, dtype2)[..., :2]
                    x_mlu = self.to_device(copy.deepcopy(x))
                    x.bitwise_and_(y)
                    mlu_ptr = x_mlu.data_ptr()
                    x_mlu.bitwise_and_(self.to_device(y))
                    self.assertTensorsEqual(
                        x.float(), x_mlu.cpu().float(), 0.003, use_MSE=True
                    )
                    self.assertEqual(mlu_ptr, x_mlu.data_ptr())

                    # test channels last
                    x = self._generate_tensor(shape1, dtype1).to(
                        memory_format=torch.channels_last
                    )
                    y = self._generate_tensor(shape2, dtype2)
                    x_mlu = self.to_device(copy.deepcopy(x))
                    x.bitwise_and_(y)
                    mlu_ptr = x_mlu.data_ptr()
                    x_mlu.bitwise_and_(self.to_device(y))
                    self.assertTensorsEqual(
                        x.float(), x_mlu.cpu().float(), 0.003, use_MSE=True
                    )
                    self.assertEqual(mlu_ptr, x_mlu.data_ptr())
                    self.assertTrue(x.stride() == x_mlu.stride())

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_and_out(self):
        self_tensor = torch.randint(-16776321, 16776565, (1, 2, 3, 4))
        other = 6654734
        device = "mlu"
        cpu_result = torch.ones(5, 2, 3, 4, dtype=torch.int32)
        device_result = torch.ones(5, 2, 3, 4, dtype=torch.int32).to(device)
        device_result1 = torch.ones(5, 2, 3, 4, dtype=torch.int32).to(
            device, memory_format=torch.channels_last
        )
        torch.bitwise_and(self_tensor, other, out=cpu_result)
        torch.bitwise_and(self_tensor.to(device), other, out=device_result1)
        torch.bitwise_and(
            self_tensor.to(device, memory_format=torch.channels_last),
            other,
            out=device_result,
        )
        self.assertTensorsEqual(cpu_result, device_result.cpu(), 0)
        self.assertTensorsEqual(cpu_result, device_result1.cpu(), 0)

        self_tensor = torch.randint(-16776321, 16776565, (1, 2, 3, 4))
        other = torch.randint(-16776321, 16776565, (4, 2, 3, 4))
        cpu_result = torch.ones(5, 2, 3, 4, dtype=torch.int32)
        device_result = torch.ones(5, 2, 3, 4, dtype=torch.int32).to(device)
        torch.bitwise_and(self_tensor, other, out=cpu_result)
        torch.bitwise_and(
            self_tensor.to(device, memory_format=torch.channels_last),
            other.to(device, memory_format=torch.channels_last),
            out=device_result,
        )
        self.assertTensorsEqual(cpu_result, device_result.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_and_permute(self):
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
            torch.bitwise_and(x, y, out=out)
            torch.bitwise_and(x_mlu, y_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_and_PYTORCH_11152(self):
        shape = (1, 4, 1, 64, 64)
        x = torch.randint(2, shape).type(torch.bool)
        y = torch.randint(2, shape).type(torch.bool)
        x.as_strided_(x.size(), stride=(4, 1, 4, 256, 4))
        y.as_strided_(y.size(), stride=(16384, 1, 16384, 256, 4))
        x_mlu = copy.deepcopy(x).mlu()
        y_mlu = copy.deepcopy(y).mlu()
        out = torch.bitwise_and(x, y)
        out_mlu = torch.bitwise_and(x_mlu, y_mlu)
        self.assertTensorsEqual(out, out_mlu.cpu(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_and_exception(self):
        ref_msg = r"MLU bitwise_and don't support tensor dtype Float."
        input1 = torch.arange(24, dtype=torch.float).reshape(1, 2, 3, 4)
        input2 = torch.arange(96).reshape(4, 2, 3, -1).float()
        input1_mlu = self.to_device(input1)
        input2_mlu = self.to_device(input2)
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            input1_mlu & input2_mlu

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("86GB")
    def test_bitwise_and_large(self):
        dtype_lst = [(torch.bool, torch.bool), (torch.int32, torch.int8)]
        for dtype1, dtype2 in dtype_lst:
            for shape1, shape2 in [((5, 1024, 1024, 1024), (5, 1024, 1024, 1024))]:
                x = self._generate_tensor(shape1, dtype1)
                y = self._generate_tensor(shape2, dtype2)
                out_cpu = torch.bitwise_and(x, y)
                out_mlu = torch.bitwise_and(self.to_device(x), self.to_device(y))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
