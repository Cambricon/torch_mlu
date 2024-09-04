from __future__ import print_function

import sys
import logging
import os

import unittest
from itertools import product
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    read_card_info,
    TestCase,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


# only have gpu op, don't support in cpu side.
def cpu_mask_scale(input, mask, scale):
    return input * mask.to(torch.float) * scale


def TensorGenerator(shape, dtype, func=lambda x: x):
    if dtype.is_floating_point:
        cpu_tensor = torch.randn(shape).to(torch.half).to(torch.float)
        mlu_tensor = func(cpu_tensor.to("mlu").to(dtype))
        cpu_tensor = func(cpu_tensor)
        return cpu_tensor, mlu_tensor
    elif dtype.is_complex:
        cpu_tensor = torch.randn(shape, dtype=dtype)
        mlu_tensor = func(cpu_tensor.to("mlu"))
        cpu_tensor = func(cpu_tensor)
        return cpu_tensor, mlu_tensor
    elif dtype == torch.bool:
        cpu_tensor = torch.randint(0, 2, shape, dtype=dtype)
        mlu_tensor = func(cpu_tensor.to("mlu"))
        cpu_tensor = func(cpu_tensor)
        return cpu_tensor, mlu_tensor
    else:
        cpu_tensor = torch.randint(100, shape, dtype=dtype)
        mlu_tensor = func(cpu_tensor.to("mlu"))
        cpu_tensor = func(cpu_tensor)
        return cpu_tensor, mlu_tensor


class Test_Masked_ScaleOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scale(self):
        shape_list = [(), (2, 3), (8, 224, 224), (1, 3, 16, 16), (128, 128, 1, 8, 3)]
        scale_list = [0.01, 0.9, 0.1, 1, 2, 3]
        type_list = [torch.float, torch.double, torch.half]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape, type, scale, func in product(
            shape_list, type_list, scale_list, func_list
        ):
            cpu_input, mlu_input = TensorGenerator(shape, type, func)
            cpu_mask, mlu_mask = TensorGenerator(shape, torch.uint8, func)
            cput_result = cpu_mask_scale(cpu_input, cpu_mask, scale)
            mlu_result = torch._masked_scale(mlu_input, mlu_mask, scale)
            self.assertTensorsEqual(cput_result, mlu_result.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scale_broadcast(self):
        shape_list = [
            ((2, 3), (2, 2, 3)),
            ((8, 22, 22), (1, 22, 22)),
            ((1, 3, 16, 16), (1, 1, 16, 16)),
            ((12, 12, 1, 8, 3), (12, 20, 8, 3)),
        ]
        scale_list = [0.01, 0.9, 0.1, 1, 2, 3]
        type_list = [torch.float, torch.double, torch.half]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape, type, scale, func in product(
            shape_list, type_list, scale_list, func_list
        ):
            cpu_input, mlu_input = TensorGenerator(shape[0], type, func)
            cpu_mask, mlu_mask = TensorGenerator(shape[1], torch.uint8, func)
            cput_result = cpu_mask_scale(cpu_input, cpu_mask, scale)
            mlu_result = torch._masked_scale(mlu_input, mlu_mask, scale)
            self.assertTensorsEqual(cput_result, mlu_result.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scale_exception(self):
        ref_msg = r"mask should be torch.uint8 dtype"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            input_mlu = torch.randn((2, 3, 4)).mlu()
            mask_mlu = torch.randn((2, 3, 4)).mlu()
            torch._masked_scale(input_mlu, mask_mlu, 2.0)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_masked_scale_bfloat16(self):
        scale_list = [0.01, 0.9, 0.1, 1, 2, 3]
        for scale in scale_list:
            cpu_input, mlu_input = TensorGenerator(
                (1, 3, 16, 16), torch.bfloat16, lambda x: x
            )
            cpu_mask, mlu_mask = TensorGenerator(
                (1, 3, 16, 16), torch.uint8, lambda x: x
            )
            cpu_result = cpu_mask_scale(cpu_input, cpu_mask, scale)
            mlu_result = torch._masked_scale(mlu_input, mlu_mask, scale)
            self.assertTensorsEqual(cpu_result, mlu_result.cpu(), 3e-3, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
