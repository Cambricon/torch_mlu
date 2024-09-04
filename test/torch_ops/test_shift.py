from __future__ import print_function

import sys
import os

# pylint: disable=all
import unittest
import logging
import copy
import random as rd

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
# pylint: disable=C0413,C0411
from common_utils import testinfo, TestCase

logging.basicConfig(level=logging.DEBUG)
torch.manual_seed(0)


class TestShiftOps(TestCase):
    """
    add left and shift op
    """

    def _generate_tensor(self, shape, dtype, min_val=2, max_val=10):
        out = torch.randint(min_val, max_val, shape).type(dtype)
        return out

    def _shift_with_cpu(self, input_data, shift_num, op="lshift"):
        """
        implement half with cpu
        """
        if isinstance(shift_num, torch.Tensor) and shift_num.numel() == 1:
            shift_num = shift_num.numpy()[0]
        assert op in [
            "lshift",
            "rshift",
        ], "op must be in lshift and rshift but now is {}".format(op)
        if op == "lshift":
            return (input_data * (1 << shift_num)).type(input_data.dtype)
        else:
            return (input_data / (2 ** (shift_num))).type(input_data.dtype)

    @testinfo()
    def test_left_and_right_shift(self):
        # rshift same as pytorch1.9 so lshift test only test on positive
        # number because of negative number can't  same as cpu
        dtype_lst = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        for dtype in dtype_lst:
            for shape1 in [
                (4,),
                (100,),
                (1000,),
                (30, 40),
                (40, 50, 50),
            ]:
                shift_nums = [2, 3, 4]
                for shift_num in shift_nums:
                    # lshift
                    input_data = self._generate_tensor(shape1, dtype)
                    result_cpu = input_data << shift_num
                    result_mlu = self.to_device(input_data) << self.to_device(shift_num)
                    self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                    shift_num_tensor = torch.Tensor([shift_num]).type(torch.int32).cpu()
                    result_cpu = input_data << shift_num_tensor
                    result_mlu = self.to_device(input_data) << self.to_device(
                        shift_num_tensor
                    )
                    self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                    # bitwise_left_shift
                    input_data = self._generate_tensor(shape1, dtype)
                    result_cpu = torch.bitwise_left_shift(input_data, shift_num)
                    result_mlu = torch.bitwise_left_shift(
                        self.to_device(input_data), self.to_device(shift_num)
                    )
                    self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                    shift_num_tensor = torch.Tensor([shift_num]).type(torch.int32).cpu()
                    result_cpu = torch.bitwise_left_shift(input_data, shift_num_tensor)
                    result_mlu = torch.bitwise_left_shift(
                        self.to_device(input_data), self.to_device(shift_num_tensor)
                    )
                    self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                    # rshift
                    input_data = self._generate_tensor(shape1, dtype, 1, 16)
                    result_cpu = input_data >> shift_num
                    result_mlu = self.to_device(input_data) >> self.to_device(shift_num)
                    self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                    shift_num_tensor = torch.Tensor([shift_num]).type(torch.int32).cpu()
                    result_cpu = input_data >> shift_num_tensor
                    result_mlu = self.to_device(input_data) >> self.to_device(
                        shift_num_tensor
                    )
                    self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                    # bitwise_right_shift
                    input_data = self._generate_tensor(shape1, dtype, 1, 16)
                    result_cpu = torch.bitwise_right_shift(input_data, shift_num)
                    result_mlu = torch.bitwise_right_shift(
                        self.to_device(input_data), self.to_device(shift_num)
                    )
                    self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                    shift_num_tensor = torch.Tensor([shift_num]).type(torch.int32).cpu()
                    result_cpu = torch.bitwise_right_shift(input_data, shift_num_tensor)
                    result_mlu = torch.bitwise_right_shift(
                        self.to_device(input_data), self.to_device(shift_num_tensor)
                    )
                    self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                    # ilshift
                    input_data = self._generate_tensor(shape1, dtype)
                    result_data_cpu = input_data.clone()
                    result_data_cpu <<= shift_num
                    result_data_mlu = input_data.clone()
                    data_on_mlu = self.to_device(result_data_mlu)
                    data_on_mlu <<= self.to_device(shift_num)
                    self.assertTensorsEqual(result_data_cpu, data_on_mlu.cpu(), 0)

                    shift_num_tensor = torch.Tensor([shift_num]).type(torch.int32).cpu()
                    result_data_cpu_t = input_data.clone()
                    result_data_cpu_t <<= shift_num_tensor
                    result_data_mlu_t = input_data.clone()
                    result_data_mlu_t = self.to_device(result_data_mlu_t)
                    result_data_mlu_t <<= self.to_device(shift_num_tensor)
                    self.assertTensorsEqual(
                        result_data_cpu_t, result_data_mlu_t.cpu(), 0
                    )
                    # irshift
                    input_data = self._generate_tensor(shape1, dtype, 1, 16)
                    result_data_cpu = input_data.clone()
                    result_data_cpu >>= shift_num
                    result_data_mlu = input_data.clone()
                    data_on_mlu = self.to_device(result_data_mlu)
                    data_on_mlu >>= self.to_device(shift_num)
                    self.assertTensorsEqual(result_data_cpu, data_on_mlu.cpu(), 0)

                    shift_num_tensor = torch.Tensor([shift_num]).type(torch.int32).cpu()
                    result_data_cpu_t = input_data.clone()
                    result_data_cpu_t >>= shift_num_tensor
                    result_data_mlu_t = input_data.clone()
                    result_data_mlu_t = self.to_device(result_data_mlu_t)
                    result_data_mlu_t >>= self.to_device(shift_num_tensor)
                    self.assertTensorsEqual(
                        result_data_cpu_t, result_data_mlu_t.cpu(), 0
                    )

        @testinfo()
        def test_left_and_right_shift_with_half(self):
            """
            test shift with half
            """
            dtype_lst = (torch.half,)
            for dtype in dtype_lst:
                for shape1 in [
                    (4,),
                    (100,),
                    (1000,),
                    (3, 5),
                    (300, 400),
                ]:
                    shift_nums = [2, 3, 4]
                    for shift_num in shift_nums:
                        input_data = self._generate_tensor(shape1, dtype)
                        # lshift
                        result_cpu = self._shift_with_cpu(input_data, shift_num)
                        result_mlu = self.to_device(input_data) << self.to_device(
                            shift_num
                        )
                        self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                        shift_num_tensor = (
                            torch.Tensor([shift_num]).type(torch.int32).cpu()
                        )
                        result_cpu = self._shift_with_cpu(input_data, shift_num)
                        result_mlu = self.to_device(input_data) << self.to_device(
                            shift_num_tensor
                        )
                        self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                        # rshift
                        result_cpu = self._shift_with_cpu(
                            input_data, shift_num, "rshift"
                        )
                        result_mlu = self.to_device(input_data) >> self.to_device(
                            shift_num
                        )
                        self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)
                        shift_num_tensor = (
                            torch.Tensor([shift_num]).type(torch.int32).cpu()
                        )
                        result_cpu = self._shift_with_cpu(
                            input_data, shift_num, "rshift"
                        )
                        result_mlu = self.to_device(input_data) >> self.to_device(
                            shift_num_tensor
                        )
                        self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

                        # ilshift
                        result_data_cpu = self._shift_with_cpu(input_data, shift_num)

                        result_data_mlu = input_data.clone()
                        result_data_mlu = self.to_device(result_data_mlu)
                        result_data_mlu <<= shift_num

                        self.assertTensorsEqual(
                            result_data_cpu, result_data_mlu.cpu(), 0
                        )

                        result_data_mlu = self.to_device(input_data.clone())
                        result_data_mlu <<= self.to_device(shift_num)
                        self.assertTensorsEqual(
                            result_data_cpu, result_data_mlu.cpu(), 0
                        )
                        # irshift
                        result_data_cpu = self._shift_with_cpu(
                            input_data, shift_num, "rshift"
                        )

                        result_data_mlu = input_data.clone()
                        result_data_mlu = self.to_device(result_data_mlu)
                        result_data_mlu >>= shift_num

                        self.assertTensorsEqual(
                            result_data_cpu, result_data_mlu.cpu(), 0
                        )

                        result_data_mlu = self.to_device(input_data.clone())
                        result_data_mlu >>= self.to_device(shift_num)
                        self.assertTensorsEqual(
                            result_data_cpu, result_data_mlu.cpu(), 0
                        )


if __name__ == "__main__":
    unittest.main()
