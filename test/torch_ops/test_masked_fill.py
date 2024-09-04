from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
import torch
import copy

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestMaskedFill(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_masked_fill_tensor(self):
        types = [torch.half, torch.float, torch.double, torch.int, torch.long]
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100,), ()]
        err = 0.0
        for t, shape in product(types, shapes):
            x = torch.rand(shape, dtype=torch.float).to(t)
            mask = torch.ones(shape, dtype=torch.bool)
            value = torch.tensor(2.33, dtype=t)
            x_mlu = self.to_device(x)
            mask_mlu = self.to_device(mask)
            value_mlu = self.to_device(value)
            ori_ptr = x_mlu.data_ptr()
            if t == torch.half:
                x, value = x.float(), value.float()
                err = 0.003
            out_cpu = torch.Tensor.masked_fill_(x, mask, value)
            out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(x, x_mlu.cpu().float(), err, use_MSE=True)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_fill_channels_last_and_not_dense(self):
        shape = (100, 512, 2, 5)
        # channels last
        x = torch.rand(shape, dtype=torch.float)
        mask = torch.ones(shape, dtype=torch.bool)
        value = torch.tensor(2.33, dtype=torch.float)
        value_mlu = self.to_device(value)
        x = x.to(memory_format=torch.channels_last)
        mask = mask.to(memory_format=torch.channels_last)
        x_mlu = self.to_device(x)
        mask_mlu = self.to_device(mask)
        out_cpu = torch.Tensor.masked_fill_(x, mask, value)
        out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )
        # not dense
        x = torch.rand(shape, dtype=torch.float)
        mask = torch.ones(shape, dtype=torch.bool)
        x_mlu = self.to_device(x)
        mask_mlu = self.to_device(mask)
        out_cpu = torch.Tensor.masked_fill_(x[..., 2], mask[..., 2], value)
        out_mlu = torch.Tensor.masked_fill_(x_mlu[..., 2], mask_mlu[..., 2], value_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_fill_scalar(self):
        types = [torch.half, torch.float, torch.double]
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100,), ()]
        err = 0.0
        for t, shape in product(types, shapes):
            x = torch.rand(shape, dtype=t)
            mask = torch.ones(shape, dtype=torch.bool)
            value = 3.14159
            x_mlu = self.to_device(x)
            ori_ptr = x_mlu.data_ptr()
            mask_mlu = self.to_device(mask)
            if t == torch.half:
                x = x.float()
                err = 0.003
            out_cpu = torch.Tensor.masked_fill_(x, mask, value)
            out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(x, x_mlu.cpu().float(), err, use_MSE=True)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())
        # following code is used to test the cases when masked_fill has int or bool inputs
        int_types = [torch.int8, torch.int16, torch.int32]
        err = 0.0
        for t, shape in product(int_types, shapes):
            max = torch.iinfo(t).max
            min = torch.iinfo(t).min
            x = torch.randint(low=min, high=max, size=shape, dtype=t)
            mask = torch.ones(shape, dtype=torch.bool)
            value = torch.tensor(1, dtype=t)
            x_mlu = self.to_device(x)
            ori_ptr = x_mlu.data_ptr()
            mask_mlu = self.to_device(mask)
            value_mlu = self.to_device(value)
            out_cpu = torch.Tensor.masked_fill_(x, mask, value)
            out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(x, x_mlu.cpu().float(), err, use_MSE=True)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())
        for shape in shapes:
            max = torch.iinfo(torch.int32).max
            min = torch.iinfo(torch.int32).min
            x = torch.randint(low=min, high=max, size=shape, dtype=torch.int32).long()
            mask = torch.ones(shape, dtype=torch.bool)
            value = torch.tensor(1, dtype=torch.int32).long()
            x_mlu = self.to_device(x)
            ori_ptr = x_mlu.data_ptr()
            mask_mlu = self.to_device(mask)
            value_mlu = self.to_device(value)
            out_cpu = torch.Tensor.masked_fill_(x, mask, value)
            out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(x, x_mlu.cpu().float(), err, use_MSE=True)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())
        for shape in shapes:
            max = torch.iinfo(torch.int8).max
            min = torch.iinfo(torch.int8).min
            x = torch.randint(low=min, high=max, size=shape, dtype=t).to(bool)
            mask = torch.ones(shape, dtype=torch.bool)
            value = torch.tensor(1, dtype=bool)
            x_mlu = self.to_device(x)
            ori_ptr = x_mlu.data_ptr()
            mask_mlu = self.to_device(mask)
            value_mlu = self.to_device(value)
            out_cpu = torch.Tensor.masked_fill_(x, mask, value)
            out_mlu = torch.Tensor.masked_fill_(x_mlu, mask_mlu, value_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(x, x_mlu.cpu().float(), err, use_MSE=True)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_fill_channels_last_broadcast(self):
        x_cpu = torch.randn(1, 3, 640, 992).to(memory_format=torch.channels_last)
        x_mlu = x_cpu.mlu()
        indices_cpu = torch.randn(1, 640, 992).to(torch.bool)
        indices_mlu = indices_cpu.mlu()
        out_cpu = x_cpu.masked_fill_(indices_cpu, 0.0)
        out_mlu = x_mlu.masked_fill_(indices_mlu, 0.0)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_fill_backward(self):
        # value scalar
        x = torch.randn((2, 2), requires_grad=True)
        x_mlu = x.mlu()
        mask = torch.tensor([[0, 1], [1, 0]]).bool()
        mask_mlu = mask.mlu()
        value_scalar = 0
        x.masked_fill(mask, value_scalar)
        x_mlu.masked_fill(mask_mlu, value_scalar)
        grad = torch.randn(2, 2)
        grad_mlu = grad.mlu()
        x.backward(grad)
        out_grad = copy.deepcopy(x.grad)
        x.grad.zero_()
        x_mlu.backward(grad_mlu)
        out_grad_mlu = x.grad
        self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(out_grad, out_grad_mlu.cpu(), 0.0, use_MSE=True)

        # value tensor (requires masked_select and sum.IntList_out)
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
        )
        x_mlu = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            device="mlu",
            requires_grad=True,
        )
        mask = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 1, 0]]).bool()
        mask_mlu = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 1, 0]], device="mlu").bool()
        value_tensor = torch.tensor(0.0, requires_grad=True)
        value_tensor_mlu = torch.tensor(0.0, device="mlu", requires_grad=True)
        out_cpu = x.masked_fill(mask, value_tensor)
        out_mlu = x_mlu.masked_fill(mask_mlu, value_tensor_mlu)
        grad = torch.randn(3, 3)
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_grad = copy.deepcopy(x.grad)
        out_mlu.backward(grad_mlu)
        out_grad_mlu = x_mlu.grad
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(out_grad, out_grad_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_fill_exceptions_and_warnings(self):
        device = "mlu"
        shape = (100, 512, 2, 5)
        max_byte = torch.iinfo(torch.uint8).max
        min_byte = torch.iinfo(torch.uint8).min
        x = torch.rand(shape, dtype=torch.float, device=device)
        x_expanded = torch.rand((1,), device=device).expand(shape)
        mask = torch.ones(shape, dtype=torch.bool, device=device)
        value = torch.tensor(1.23, dtype=torch.float, device=device)
        value_1D = torch.tensor([1.23], dtype=torch.float, device=device)
        value_byte = torch.tensor(1, dtype=torch.uint8, device=device)
        x_byte = torch.randint(
            low=min_byte, high=max_byte, size=shape, dtype=torch.uint8, device="cpu"
        ).mlu()
        mask_float = torch.randn(shape, dtype=torch.float, device=device)
        mask_byte = torch.randint(
            low=min_byte, high=max_byte, size=shape, dtype=torch.uint8, device="cpu"
        ).mlu()
        mask_not_expandable = torch.ones(
            size=(100, 512, 2, 5, 1), dtype=torch.bool, device=device
        )
        err_msg = "input type is not support uint8 in cnnl_masked_fill_internal"
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            torch.Tensor.masked_fill_(x_byte, mask, value)
        self.assertEqual(err_msg, err_msg_mlu.exception.args[0])
        err_msg = "masked_fill only supports boolean masks, but got dtype Float"
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            torch.Tensor.masked_fill_(x, mask_float, value)
        self.assertEqual(err_msg, err_msg_mlu.exception.args[0])
        err_msg = (
            "expand(mluBoolType{[100, 512, 2, 5, 1]}, size=[100, 512, 2, 5]): "
            + "the number of sizes provided (4) must be greater or equal to "
            + "the number of dimensions in the tensor (5)"
        )
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            torch.Tensor.masked_fill_(x, mask_not_expandable, value)
        self.assertEqual(err_msg, err_msg_mlu.exception.args[0])
        warning_msg = (
            r"^Use of masked_fill_ on expanded tensors is deprecated. "
            + r"Please clone\(\) the tensor before performing this operation. "
            + r"This also applies to advanced indexing e.g. tensor\[mask\] = scalar "
        )
        with self.assertWarnsRegex(UserWarning, warning_msg):
            torch.Tensor.masked_fill_(x_expanded, mask, value)
        err_msg = (
            "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
            "with ",
            value.dim(),
            " dimension(s).",
        )
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            torch.Tensor.masked_fill_(x, mask, value_1D)
        err_msg = "value tensor does not suppot uint8"
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            torch.Tensor.masked_fill_(x, mask, value_byte)
        err_msg = (
            "unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. "
            "Please clone() the tensor before performing the operation."
        )
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            torch.Tensor.masked_fill_(mask[1:], mask[:-1], False)
        input_cpu = torch.randn(2, 2).cpu()
        mask_mlu = torch.tensor([[1, 0], [0, 1]]).bool().mlu()
        err_msg = "expected self and mask to be on the same device, but got mask on mlu:0 and self on cpu"
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            input_cpu.masked_fill_(mask, 0.0)
        err_msg = "masked_fill_: Expected inputs to be on same device"
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            input_cpu.masked_fill(mask.cpu(), torch.tensor(0).mlu())

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_masked_fill_bfloat16(self):
        input_cpu = torch.rand((10, 12, 2, 5), dtype=torch.bfloat16)
        input_mlu = input_cpu.mlu()
        mask_cpu = torch.testing.make_tensor(
            (10, 12, 2, 5), dtype=torch.bool, device="cpu"
        )
        mask_mlu = mask_cpu.mlu()
        value_scalar = 2.33
        value_tensor = torch.tensor(4.6, dtype=torch.bfloat16)
        out_cpu_with_scalar = torch.Tensor.masked_fill_(
            input_cpu, mask_cpu, value_scalar
        )
        out_mlu_with_scalar = torch.Tensor.masked_fill_(
            input_mlu, mask_mlu, value_scalar
        )
        self.assertTensorsEqual(
            out_cpu_with_scalar, out_mlu_with_scalar.cpu(), 0.0, use_MSE=True
        )
        input_cpu = torch.rand((10, 12, 2, 5), dtype=torch.bfloat16)
        input_mlu = input_cpu.mlu()
        mask_cpu = torch.testing.make_tensor(
            (10, 12, 2, 5), dtype=torch.bool, device="cpu"
        )
        mask_mlu = mask_cpu.mlu()
        out_cpu_with_tensor = torch.Tensor.masked_fill_(
            input_cpu, mask_cpu, value_tensor
        )
        out_mlu_with_tensor = torch.Tensor.masked_fill_(
            input_mlu, mask_mlu, value_tensor
        )
        self.assertTensorsEqual(
            out_cpu_with_tensor, out_mlu_with_tensor.cpu(), 0.0, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("30GB")
    def test_masked_fill_large_half(self):
        input_cpu = torch.rand((4, 1025, 1024, 1024), dtype=torch.half)
        input_mlu = input_cpu.mlu()
        mask_cpu = torch.testing.make_tensor(
            (4, 1025, 1024, 1024), dtype=torch.bool, device="cpu"
        )
        mask_mlu = mask_cpu.mlu()
        value_scalar = 2.33
        value_tensor = torch.tensor(4.6, dtype=torch.half)
        out_cpu_with_scalar = torch.Tensor.masked_fill_(
            input_cpu, mask_cpu, value_scalar
        )
        out_mlu_with_scalar = torch.Tensor.masked_fill_(
            input_mlu, mask_mlu, value_scalar
        )
        self.assertTensorsEqual(
            out_cpu_with_scalar.float(),
            out_mlu_with_scalar.cpu().float(),
            0.0,
            use_MSE=True,
        )
        input_cpu = torch.rand((4, 1025, 1024, 1024), dtype=torch.half)
        input_mlu = input_cpu.mlu()
        mask_cpu = torch.testing.make_tensor(
            (4, 1025, 1024, 1024), dtype=torch.bool, device="cpu"
        )
        mask_mlu = mask_cpu.mlu()
        out_cpu_with_tensor = torch.Tensor.masked_fill_(
            input_cpu, mask_cpu, value_tensor
        )
        out_mlu_with_tensor = torch.Tensor.masked_fill_(
            input_mlu, mask_mlu, value_tensor
        )
        self.assertTensorsEqual(
            out_cpu_with_tensor.float(),
            out_mlu_with_tensor.cpu().float(),
            0.0,
            use_MSE=True,
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("30GB")
    def test_masked_fill_onedim_large_half(self):
        shape = 4294967297
        input_cpu = torch.rand(shape, dtype=torch.half)
        input_mlu = input_cpu.mlu()
        mask_cpu = torch.testing.make_tensor(shape, dtype=torch.bool, device="cpu")
        mask_mlu = mask_cpu.mlu()
        value_scalar = 2.33
        value_tensor = torch.tensor(4.6, dtype=torch.half)
        out_cpu_with_scalar = torch.Tensor.masked_fill_(
            input_cpu, mask_cpu, value_scalar
        )
        out_mlu_with_scalar = torch.Tensor.masked_fill_(
            input_mlu, mask_mlu, value_scalar
        )
        self.assertTensorsEqual(
            out_cpu_with_scalar.float(),
            out_mlu_with_scalar.cpu().float(),
            0.003,
            use_MSE=True,
        )
        input_cpu = torch.rand(shape, dtype=torch.half)
        input_mlu = input_cpu.mlu()
        mask_cpu = torch.testing.make_tensor(shape, dtype=torch.bool, device="cpu")
        mask_mlu = mask_cpu.mlu()
        out_cpu_with_tensor = torch.Tensor.masked_fill_(
            input_cpu, mask_cpu, value_tensor
        )
        out_mlu_with_tensor = torch.Tensor.masked_fill_(
            input_mlu, mask_mlu, value_tensor
        )
        self.assertTensorsEqual(
            out_cpu_with_tensor.float(),
            out_mlu_with_tensor.cpu().float(),
            0.003,
            use_MSE=True,
        )


if __name__ == "__main__":
    run_tests()
