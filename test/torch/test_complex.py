import sys
import os
import unittest
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411


class TestComplexTensor(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_dtype_inference(self):
        for dtype in [torch.float32, torch.float64]:
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            x = torch.tensor([3.0, 3.0 + 5.0j], device=torch.device("mlu:0"))
            torch.set_default_dtype(default_dtype)
            self.assertEqual(
                x.dtype, torch.cdouble if dtype == torch.float64 else torch.cfloat
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_complex_contiguous(self):
        for shape in [(2, 3, 4, 5), (3, 2, 4), (7, 5, 4, 3, 2)]:
            for func in [self.convert_to_channel_last, self.to_non_dense]:
                x = torch.randn(shape, dtype=torch.cfloat)
                x_cpu = func(x)
                x_mlu = func(x.mlu())
                o_cpu = x_cpu.contiguous()
                o_mlu = x_mlu.contiguous()
                self.assertTrue(x_cpu.stride() == x_mlu.stride())
                self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
                self.assertTensorsEqual(o_cpu, o_mlu.cpu(), 0.0)
                self.assertTrue(o_cpu.size() == o_mlu.size())
                self.assertTrue(o_cpu.stride() == o_mlu.stride())
                self.assertTrue(o_mlu.storage_offset() == o_cpu.storage_offset())


if __name__ == "__main__":
    unittest.main()
