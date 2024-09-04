import os
import sys
import logging
import unittest
import copy
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
)  # pylint: disable=C0413,C0411


class TestPolar(TestCase):
    def _test_polar(self, magnitude, angle, complex_cpu):
        # out-of-place
        out_cpu = torch.polar(magnitude, angle)
        out_mlu = torch.polar(magnitude.to("mlu"), angle.to("mlu"))
        self.assertTensorsEqual(out_cpu.real, out_mlu.real.cpu(), 0.003, use_MSE=True)
        self.assertTensorsEqual(out_cpu.imag, out_mlu.imag.cpu(), 0.003, use_MSE=True)

        # in-place
        complex_mlu = copy.deepcopy(complex_cpu).to("mlu")
        complex_mlu_ptr = complex_mlu.data_ptr()
        torch.polar(magnitude, angle, out=complex_cpu)
        torch.polar(magnitude.to("mlu"), angle.to("mlu"), out=complex_mlu)
        self.assertTensorsEqual(
            complex_cpu.real, complex_mlu.real.cpu(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            complex_cpu.imag, complex_mlu.imag.cpu(), 0.003, use_MSE=True
        )
        self.assertEqual(complex_mlu_ptr, complex_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_polar(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
            (512, 1024, 0, 2, 4),
            (10, 0, 32, 32),
            (2, 0, 4),
            (0, 254, 112, 1, 1, 3),
        ]
        dtype_list = [
            (torch.float32, torch.complex64),
            (torch.float64, torch.complex128),
        ]
        for shape in shape_list:
            for dtype_float, dtype_complex in dtype_list:
                magnitude = torch.abs(torch.randn(shape, dtype=dtype_float))
                angle = torch.randn(shape, dtype=dtype_float)
                complex_cpu = torch.randn(shape, dtype=dtype_complex)
                self._test_polar(magnitude, angle, complex_cpu)

    # @unittest.skip("not test")
    @testinfo()
    def test_polar_PYTORCH_11152(self):
        dtype_list = [
            (torch.float32, torch.complex64),
            (torch.float64, torch.complex128),
        ]
        for dtype_float, dtype_complex in dtype_list:
            magnitude = torch.abs(torch.randn((1, 4, 1, 64, 64), dtype=dtype_float))
            angle = torch.randn((1, 4, 1, 64, 64), dtype=dtype_float)
            magnitude.as_strided_(magnitude.size(), stride=(4, 1, 4, 256, 4))
            angle.as_strided_(angle.size(), stride=(16384, 1, 16384, 256, 4))
            out_cpu = torch.polar(magnitude, angle)
            out_mlu = torch.polar(magnitude.to("mlu"), angle.to("mlu"))
            self.assertTensorsEqual(
                out_cpu.real, out_mlu.real.cpu(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                out_cpu.imag, out_mlu.imag.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_polar_channel_last(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        dtype_list = [
            (torch.float32, torch.complex64),
            (torch.float64, torch.complex128),
        ]
        for shape in shape_list:
            for dtype_float, dtype_complex in dtype_list:
                magnitude = torch.abs(torch.randn(shape, dtype=dtype_float))
                angle = torch.randn(shape, dtype=dtype_float)
                complex_cpu = torch.randn(shape, dtype=dtype_complex)
                magnitude = self.convert_to_channel_last(magnitude)
                angle = self.convert_to_channel_last(angle)
                complex_cpu = self.convert_to_channel_last(complex_cpu)
                self._test_polar(magnitude, angle, complex_cpu)

    # @unittest.skip("not test")
    @testinfo()
    def test_polar_not_dense(self):
        shape_list = [
            (512, 1024, 2, 2, 8),
            (10, 3, 32, 64),
            (2, 3, 8),
            (254, 254, 112, 1, 1, 6),
        ]
        dtype_list = [
            (torch.float32, torch.complex64),
            (torch.float64, torch.complex128),
        ]
        for shape in shape_list:
            for dtype_float, dtype_complex in dtype_list:
                magnitude = torch.abs(torch.randn(shape, dtype=dtype_float))
                angle = torch.randn(shape, dtype=dtype_float)
                complex_cpu = torch.randn(shape, dtype=dtype_complex)
                if len(shape) == 4:
                    magnitude = magnitude[:, :, :, : int(shape[-1] / 2)]
                    angle = angle[:, :, :, : int(shape[-1] / 2)]
                    complex_cpu = complex_cpu[:, :, :, : int(shape[-1] / 2)]
                elif len(shape) == 3:
                    magnitude = magnitude[:, :, : int(shape[-1] / 2)]
                    angle = angle[:, :, : int(shape[-1] / 2)]
                    complex_cpu = complex_cpu[:, :, : int(shape[-1] / 2)]
                self._test_polar(magnitude, angle, complex_cpu)

    # @unittest.skip("not test")
    @testinfo()
    def test_poalr_backward(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        for shape in shape_list:
            magnitude = torch.randn(shape, dtype=torch.float, requires_grad=True)
            angle = torch.randn(shape, dtype=torch.float, requires_grad=True)

            out_cpu = torch.polar(magnitude, angle)
            out_mlu = torch.polar(magnitude.to("mlu"), angle.to("mlu"))

            grad_in = torch.randn(out_cpu.shape, dtype=out_cpu.dtype)

            out_cpu.backward(grad_in)
            grad_cpu_magnitude = copy.deepcopy(magnitude.grad)
            grad_cpu_angle = copy.deepcopy(angle.grad)

            magnitude.grad.zero_()
            angle.grad.zero_()

            out_mlu.backward(grad_in.to("mlu"))
            grad_mlu_magnitude = copy.deepcopy(magnitude.grad)
            grad_mlu_angle = copy.deepcopy(angle.grad)

            self.assertTensorsEqual(
                grad_cpu_magnitude, grad_mlu_magnitude.cpu(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                grad_cpu_angle, grad_mlu_angle.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_polar_exception(self):
        magnitude = torch.abs(torch.randn((2, 3, 4, 10), dtype=torch.bfloat16))
        angle = torch.randn((2, 3, 4, 10), dtype=torch.float32)
        ref_msg = "Expected both inputs to be Half, Float or Double tensors but got BFloat16 and Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.polar(magnitude.to("mlu"), angle.to("mlu"))

        magnitude = torch.abs(torch.randn((2, 3, 4, 10), dtype=torch.float32))
        angle = torch.randn((2, 3, 4, 10), dtype=torch.float64)
        ref_msg = "Expected object of scalar type Float but got scalar type Double for second argument"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.polar(magnitude.to("mlu"), angle.to("mlu"))

        magnitude = torch.abs(torch.randn((2, 3, 4, 10), dtype=torch.float32))
        angle = torch.randn((2, 3, 4, 10), dtype=torch.float32)
        complex_cpu = torch.randn((2, 3, 4, 10), dtype=torch.complex128)
        complex_mlu = copy.deepcopy(complex_cpu).to("mlu")
        ref_msg = "Expected object of scalar type ComplexFloat but got scalar type ComplexDouble"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.polar(magnitude.to("mlu"), angle.to("mlu"), out=complex_mlu)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("60GB", device="mlu")
    def test_polar_large(self):
        magnitude = torch.abs(torch.randn((3, 1024, 1024, 1024), dtype=torch.float32))
        angle = torch.randn((3, 1024, 1024, 1024), dtype=torch.float32)
        out_cpu = torch.polar(magnitude, angle)
        out_mlu = torch.polar(magnitude.to("mlu"), angle.to("mlu"))
        self.assertTensorsEqual(out_cpu.real, out_mlu.real.cpu(), 0.003, use_MSE=True)
        self.assertTensorsEqual(out_cpu.imag, out_mlu.imag.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
