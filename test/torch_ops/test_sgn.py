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


class TestSgnOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_torch(self):
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
                out_cpu = torch.sgn(x)
                out_mlu = torch.sgn(x.to("mlu"))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
                self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_torch_channel_last(self):
        for shape_ in [(1, 3, 224, 224)]:
            x = torch.randn(shape_)
            x_origin = x.clone()
            x_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last)
            x_mlu = x_mlu.to(memory_format=torch.channels_last)
            out_cpu = torch.sgn(x)
            out_mlu = torch.sgn(x_mlu)

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)

        for shape_ in [(1, 3, 224, 224, 112)]:
            x = torch.randn(shape_)
            x_origin = x.clone()
            x_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last_3d)
            x_mlu = x_mlu.to(memory_format=torch.channels_last_3d)
            out_cpu = torch.sgn(x)
            out_mlu = torch.sgn(x_mlu)

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_torch_not_dense(self):
        for shape_ in [(1, 3, 224, 224)]:
            x = torch.randn(shape_)
            x_origin = x.clone()
            out_cpu = torch.sgn(x[:, :, :, :4])
            out_mlu = torch.sgn(x.to("mlu")[:, :, :, :4])

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_tensor(self):
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
                out_cpu = x.sgn()
                out_mlu = x.to("mlu").sgn()

                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0, use_MSE=True
                )
                self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_tensor_channel_last(self):
        for shape_ in [(1, 3, 224, 224)]:
            x = torch.randn(shape_)
            x_origin = x.clone()
            x_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last)
            x_mlu = x_mlu.to(memory_format=torch.channels_last)
            out_cpu = x.sgn()
            out_mlu = x_mlu.sgn()

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)
            self.assertTrue(out_mlu.is_contiguous(memory_format=torch.channels_last))

        for shape_ in [(1, 3, 224, 224, 112)]:
            x = torch.randn(shape_)
            x_origin = x.clone()
            x_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last_3d)
            x_mlu = x_mlu.to(memory_format=torch.channels_last_3d)
            out_cpu = x.sgn()
            out_mlu = x_mlu.sgn()

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)
            self.assertTrue(out_mlu.is_contiguous(memory_format=torch.channels_last_3d))

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_tensor_not_dense(self):
        for shape in [(1, 3, 224, 224)]:
            x = torch.randn(shape)
            x_origin = x.clone()
            out_cpu = x[:, :, :, :4].sgn()
            out_mlu = x.to("mlu")
            out_mlu = out_mlu[:, :, :, :4].sgn()

            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)
            self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_tensor_inplace(self):
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
                out_cpu.sgn_()
                out_mlu.sgn_()

                self.assertEqual(out_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_tensor_inplace_channel_last(self):
        for shape in [(1, 3, 224, 224)]:
            x = torch.randn(shape)
            out_cpu = x.clone()
            out_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last)
            out_mlu = out_mlu.to(memory_format=torch.channels_last)
            out_ptr = out_mlu.data_ptr()
            out_cpu.sgn_()
            out_mlu.sgn_()

            self.assertEqual(out_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)

        for shape in [(1, 3, 224, 224, 112)]:
            x = torch.randn(shape)
            out_cpu = x.clone()
            out_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last_3d)
            out_mlu = out_mlu.to(memory_format=torch.channels_last_3d)
            out_ptr = out_mlu.data_ptr()
            out_cpu.sgn_()
            out_mlu.sgn_()

            self.assertEqual(out_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_tensor_inplace_not_dense(self):
        for shape in [(1, 3, 224, 224)]:
            x = torch.randn(shape)
            out_cpu = x.clone()
            out_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last)
            out_mlu = out_mlu.to(memory_format=torch.channels_last)
            out_ptr = out_mlu.data_ptr()
            out_cpu[:, :, :, :4].sgn_()
            out_mlu[:, :, :, :4].sgn_()

            self.assertEqual(out_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)

        for shape in [(1, 3, 224, 224, 112)]:
            x = torch.randn(shape)
            out_cpu = x.clone()
            out_mlu = x.to("mlu")
            x = x.to(memory_format=torch.channels_last_3d)
            out_mlu = out_mlu.to(memory_format=torch.channels_last_3d)
            out_ptr = out_mlu.data_ptr()
            out_cpu[:, :, :, :4].sgn_()
            out_mlu[:, :, :, :4].sgn_()

            self.assertEqual(out_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_torch_with_out(self):
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
                torch.sgn(x, out=out_cpu)
                torch.sgn(x.to("mlu"), out=out_mlu)

                self.assertEqual(out_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0, use_MSE=True
                )
                self.assertTensorsEqual(x, x_origin, 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_permute(self):
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
            torch.sgn(x, out=out)
            torch.sgn(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half, torch.complex64]
        for type_ in dtype_list:
            x = torch.randn((2, 3, 4, 5, 6), dtype=type_)
            x_mlu = x.to("mlu")
            x.sgn_()
            x_mlu.sgn_()
            if type_ == torch.complex64:
                self.assertTensorsEqual(
                    x.real.float(), x_mlu.real.cpu().float(), 0.0, use_MSE=True
                )
                self.assertTensorsEqual(
                    x.imag.float(), x_mlu.imag.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_backward(self):
        for shape_ in [
            (1),
            (10),
            (2, 3),
            (8, 224, 224),
            (1, 3, 16, 16),
            (1, 3, 16, 16, 3),
        ]:
            x = torch.randn(shape_, dtype=torch.float, requires_grad=True)
            x_mlu = copy.deepcopy(x).to("mlu")
            x_mlu.retain_grad()
            grad_in = torch.randn(shape_, dtype=torch.float)
            grad_in_mlu = grad_in.to("mlu")

            out_cpu = torch.sgn(x)
            out_cpu.backward(grad_in)

            out_mlu = torch.sgn(x_mlu)
            out_mlu.backward(grad_in_mlu)

            self.assertTensorsEqual(x.grad, x_mlu.grad.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sgn_exception(self):
        x0 = torch.randint(high=255, size=(5,), dtype=torch.int)
        x0_mlu = x0.to("mlu")
        ref_msg = "MLU sgn don't support tensor dtype Int."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x0_mlu.sgn_()

        x1 = torch.randn((2, 3, 4, 10), dtype=torch.float)
        out1_mlu = x1.clone().to("mlu").to(torch.half)
        ref_msg = "Found dtype Half but expected Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.sgn(x1.to("mlu"), out=out1_mlu)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_sgn_bfloat16(self):
        for shape_ in [
            (1),
            (10),
            (2, 3),
            (8, 224, 224),
            (1, 3, 16, 16),
            (1, 3, 16, 16, 3),
        ]:
            x = torch.randn(shape_, dtype=torch.bfloat16, requires_grad=True)
            x_mlu = copy.deepcopy(x).to("mlu")
            x_mlu.retain_grad()
            grad_in = torch.randn(shape_, dtype=torch.bfloat16)
            grad_in_mlu = grad_in.to("mlu")

            out_cpu = torch.sgn(x)
            out_cpu.backward(grad_in)

            out_mlu = torch.sgn(x_mlu)
            out_mlu.backward(grad_in_mlu)
            self.assertTensorsEqual(x.grad, x_mlu.grad.cpu(), 0.0, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
