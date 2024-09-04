# pylint: disable=W0511,W0105
from __future__ import print_function

import sys
import os
import unittest
import logging
import copy
import torch


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestSignOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_sign_torch(self):
        type_list = [torch.float, torch.half, torch.double]
        for shape_ in [
            (1, 3, 224, 224),
            (2, 30, 80),
            (3, 20),
            (10),
            (1, 3, 224),
            (1),
            (),
        ]:
            for type_ in type_list:
                x = torch.randn(shape_, dtype=type_)
                x_origin = x.clone()
                out_cpu = torch.sign(x)
                out_mlu = torch.sign(x.to("mlu"))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
                self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_torch_channel_last(self):
        for shape_ in [(1, 3, 224, 224)]:
            x = torch.randn(shape_)
            x_origin = x.clone()
            x_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last)
            x_mlu = x_mlu.to(memory_format=torch.channels_last)
            out_cpu = torch.sign(x)
            out_mlu = torch.sign(x_mlu)

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)

        for shape_ in [(1, 3, 224, 224, 112)]:
            x = torch.randn(shape_)
            x_origin = x.clone()
            x_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last_3d)
            x_mlu = x_mlu.to(memory_format=torch.channels_last_3d)
            out_cpu = torch.sign(x)
            out_mlu = torch.sign(x_mlu)

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_torch_not_dense(self):
        for shape_ in [(1, 3, 224, 224)]:
            x = torch.randn(shape_)
            x_origin = x.clone()
            out_cpu = torch.sign(x[:, :, :, :4])
            out_mlu = torch.sign(x.to("mlu")[:, :, :, :4])

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_tensor(self):
        type_list = [torch.float, torch.half, torch.double]
        for shape_ in [
            (1, 3, 224, 224),
            (2, 30, 80),
            (3, 20),
            (10),
            (1, 3, 224),
            (1),
            (),
        ]:
            for type_ in type_list:
                x = torch.randn(shape_, dtype=type_)
                x_origin = x.clone()
                out_cpu = x.sign()
                out_mlu = x.to("mlu").sign()

                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0, use_MSE=True
                )
                self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_tensor_channel_last(self):
        for shape_ in [(1, 3, 224, 224)]:
            x = torch.randn(shape_)
            x_origin = x.clone()
            x_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last)
            x_mlu = x_mlu.to(memory_format=torch.channels_last)
            out_cpu = x.sign()
            out_mlu = x_mlu.sign()

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)
            self.assertTrue(out_mlu.is_contiguous(memory_format=torch.channels_last))

        for shape_ in [(1, 3, 224, 224, 112)]:
            x = torch.randn(shape_)
            x_origin = x.clone()
            x_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last_3d)
            x_mlu = x_mlu.to(memory_format=torch.channels_last_3d)
            out_cpu = x.sign()
            out_mlu = x_mlu.sign()

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)
            self.assertTrue(out_mlu.is_contiguous(memory_format=torch.channels_last_3d))

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_tensor_not_dense(self):
        for shape in [(1, 3, 224, 224)]:
            x = torch.randn(shape)
            x_origin = x.clone()
            out_cpu = x[:, :, :, :4].sign()
            out_mlu = x.to("mlu")
            out_mlu = out_mlu[:, :, :, :4].sign()

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_tensor_inplace(self):
        type_list = [torch.float, torch.half, torch.double]
        for shape_ in [
            (1, 3, 224, 224),
            (2, 30, 80),
            (3, 20),
            (10),
            (1, 3, 224),
            (1),
            (),
        ]:
            for type_ in type_list:
                x = torch.randn(shape_, dtype=type_)
                out_cpu = x.clone()
                out_mlu = x.to("mlu")
                out_ptr = out_mlu.data_ptr()
                out_cpu.sign_()
                out_mlu.sign_()

                self.assertEqual(out_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_tensor_inplace_channel_last(self):
        for shape in [(1, 3, 224, 224)]:
            x = torch.randn(shape)
            out_cpu = x.clone()
            out_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last)
            out_mlu = out_mlu.to(memory_format=torch.channels_last)
            out_ptr = out_mlu.data_ptr()
            out_cpu.sign_()
            out_mlu.sign_()

            self.assertEqual(out_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)

        for shape in [(1, 3, 224, 224, 112)]:
            x = torch.randn(shape)
            out_cpu = x.clone()
            out_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last_3d)
            out_mlu = out_mlu.to(memory_format=torch.channels_last_3d)
            out_ptr = out_mlu.data_ptr()
            out_cpu.sign_()
            out_mlu.sign_()

            self.assertEqual(out_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_tensor_inplace_not_dense(self):
        for shape in [(1, 3, 224, 224)]:
            x = torch.randn(shape)
            out_cpu = x.clone()
            out_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last)
            out_mlu = out_mlu.to(memory_format=torch.channels_last)
            out_ptr = out_mlu.data_ptr()
            out_cpu[:, :, :, :4].sign_()
            out_mlu[:, :, :, :4].sign_()

            self.assertEqual(out_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)

        for shape in [(1, 3, 224, 224, 112)]:
            x = torch.randn(shape)
            out_cpu = x.clone()
            out_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last_3d)
            out_mlu = out_mlu.to(memory_format=torch.channels_last_3d)
            out_ptr = out_mlu.data_ptr()
            out_cpu[:, :, :, :4].sign_()
            out_mlu[:, :, :, :4].sign_()

            self.assertEqual(out_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_torch_with_out(self):
        type_list = [torch.float, torch.half, torch.double]
        for shape_ in [
            (1, 3, 224, 224),
            (2, 30, 80),
            (3, 20),
            (10),
            (1, 3, 224),
            (1),
            (),
        ]:
            for type_ in type_list:
                x = torch.randn(shape_, dtype=type_)
                x_origin = x.clone()
                out_cpu = x.clone()
                out_mlu = out_cpu.to("mlu")
                out_ptr = out_mlu.data_ptr()
                torch.sign(x, out=out_cpu)
                torch.sign(x.to("mlu"), out=out_mlu)

                self.assertEqual(out_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0, use_MSE=True
                )
                self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            out_mlu = copy.deepcopy(out).to("mlu")
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                permute_shape[i]
            )
            torch.sign(x, out=out)
            torch.sign(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for type_ in dtype_list:
            x = torch.randn((2, 3, 4, 5, 6), dtype=type_)
            x_mlu = x.to("mlu")
            x.sign_()
            x_mlu.sign_()
            self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_backward(self):
        for shape_ in [
            (1),
            (10),
            (2, 3),
            (8, 224, 224),
            (1, 3, 16, 16),
            (1, 3, 16, 16, 3),
        ]:
            x = torch.randn(shape_, dtype=torch.float, requires_grad=True)
            x_mlu = x.to("mlu")

            out_cpu = torch.sign(x)
            out_mlu = torch.sign(x_mlu)

            grad_in = torch.randn(out_cpu.shape, dtype=torch.float)
            grad_in_mlu = grad_in.to("mlu")

            out_cpu.backward(grad_in)
            grad_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()

            out_mlu.backward(grad_in_mlu)
            grad_mlu = copy.deepcopy(x.grad)

            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sign_exception(self):
        x0 = torch.randint(high=255, size=(5,), dtype=torch.int)
        x0_mlu = x0.to("mlu")
        ref_msg = "MLU sign don't support tensor dtype Int."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x0_mlu.sign_()

        x1 = torch.randn((2, 3, 4, 10), dtype=torch.float)
        out1_mlu = x1.clone().to("mlu").to(torch.half)
        ref_msg = "Found dtype Half but expected Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.sign(x1.to("mlu"), out=out1_mlu)

        x2 = torch.randn((2, 3, 4, 10), dtype=torch.complex64)
        x2_mlu = x2.to("mlu")
        ref_msg = "Unlike NumPy, torch.sign is not intended to support complex numbers. Please use torch.sgn instead."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x2_mlu.sign_()

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_sign_bfloat16(self):
        for shape_ in [
            (1),
            (10),
            (2, 3),
            (8, 224, 224),
            (1, 3, 16, 16),
            (1, 3, 16, 16, 3),
        ]:
            x = torch.randn(shape_, dtype=torch.bfloat16, requires_grad=True)
            x_mlu = x.to("mlu")
            out_cpu = torch.sign(x)
            out_mlu = torch.sign(x_mlu)

            grad_in = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
            grad_in_mlu = grad_in.to("mlu")

            out_cpu.backward(grad_in)
            grad_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()
            out_mlu.backward(grad_in_mlu)
            grad_mlu = copy.deepcopy(x.grad)

            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.0, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
