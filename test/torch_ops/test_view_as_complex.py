from __future__ import print_function
import unittest
import logging
import sys
import os
import torch

os.environ["ENABLE_FALLBACK_TO_CPU"] = "0"
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestViewAsComplexOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_view_as_complex(self):
        def fn(dtype, contiguous_input=True, dim0=0, dim1=1):
            t = torch.randn(3, 2, 2, dtype=dtype).to("mlu")
            if not contiguous_input:
                t = t.transpose(dim0, dim1)
                res_mlu = torch.view_as_complex(t)
                self.assertTensorsEqual(t[:, :, 0].cpu(), res_mlu.real.cpu(), 0.0)
                self.assertTensorsEqual(t[:, :, 1].cpu(), res_mlu.imag.cpu(), 0.0)

        dtype_list = [torch.float32, torch.double, torch.half]
        for dtype in dtype_list:
            fn(dtype)
            fn(dtype, contiguous_input=False)

    # @unittest.skip("not test")
    @testinfo()
    def test_view_as_complex_exception(self):
        x = torch.randn(3, 3, dtype=torch.float32).to("mlu")
        t = torch.as_strided(x, (2, 2), (1, 1))
        ref_msg = (
            r"^Tensor must have a stride divisible by 2 for all but last dimension"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.view_as_complex(t)
        x = torch.tensor([], dtype=torch.float32).to("mlu")
        ref_msg = r"^Tensor must have a last dimension of size 2"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.view_as_complex(x)
        x = torch.tensor(2.0)
        ref_msg = r"^Input tensor must have one or more dimensions"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.view_as_complex(x)
        t = torch.randn(3, 2, 2, dtype=torch.float).to("mlu")
        t = t.transpose(1, 2)
        ref_msg = r"^Tensor must have a last dimension with stride 1"
        if t.stride()[-1] != 1:
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                torch.view_as_complex(t)


if __name__ == "__main__":
    unittest.main()
