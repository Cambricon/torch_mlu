from __future__ import print_function
import logging
import sys
import os
import unittest

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413


class TestCloneOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_clone(self):
        a = torch.randn(1, 3, 512, 224, dtype=torch.float)
        b = torch.zeros(1, 3, 512, 224, dtype=torch.float)
        a_mlu = a.mlu()
        b_mlu = b.mlu()
        out_cpu = a.clone()
        out_mlu = a_mlu.clone()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

        c = a + b
        c_mlu = a_mlu + b_mlu
        out_cpu = c.clone()
        out_mlu = c_mlu.clone()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # TODO(miaochen): dependency aten::complex.out
    @unittest.skip("not test")
    @testinfo()
    def test_clone_complex(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype in dtype_list:
            a = torch.randn(1, 3, 512, 224, dtype=dtype)
            b = torch.randn(1, 3, 512, 224, dtype=dtype)
            a_mlu = a.mlu()
            b_mlu = b.mlu()
            c = torch.complex(a, b)
            c_mlu = torch.complex(a_mlu, b_mlu)
            out_cpu = c.clone()
            out_mlu = c_mlu.clone()
            self.assertTensorsEqual(
                out_cpu.real.float(), out_mlu.real.float().cpu(), 0.0, use_MSE=True
            )
            self.assertTensorsEqual(
                out_cpu.imag.float(), out_mlu.imag.float().cpu(), 0.0, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_clone_channels_last(self):
        x = torch.randn(1, 3, 512, 224, dtype=torch.float)
        x_cpu = x.to(memory_format=torch.channels_last)
        x_mlu = x_cpu.to("mlu")
        out_cpu = x_cpu.clone()
        out_mlu = x_mlu.clone()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTrue(out_cpu.stride() == out_mlu.stride())
        self.assertTrue(out_cpu.storage_offset() == out_mlu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_clone_not_dense(self):
        x = torch.randn(1, 3, 512, 224, dtype=torch.float)
        x_cpu = x[:, :, :, 100:200]
        x_mlu = x.to("mlu")[:, :, :, 100:200]
        out_cpu = x_cpu.clone()
        out_mlu = x_mlu.clone()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTrue(out_cpu.stride() == out_mlu.stride())
        self.assertTrue(out_cpu.storage_offset() == out_mlu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_clone_conj(self):
        input_mlu = torch.randn(3, dtype=torch.complex64).mlu()
        mlu_conj_physical = torch.conj_physical(input_mlu)
        mlu_conj_pyhsical_conj = torch.conj(mlu_conj_physical)
        common_shape = [5, 2, 3]
        result = mlu_conj_pyhsical_conj.expand(common_shape)
        reference = input_mlu.expand(common_shape)
        result = result.to(torch.complex128)
        reference = reference.to(torch.complex128)
        self.assertTensorsEqual(
            reference.real.cpu(), result.real.cpu(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            reference.imag.cpu(), result.imag.cpu(), 0.0, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
