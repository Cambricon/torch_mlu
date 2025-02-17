import torch
import torch_mlu

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_dtype import all_types_and_complex_and

import os
import sys
import unittest

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, run_tests, TestCase

TEST_FLOAT8 = torch.mlu.is_fp8_supported()


class TestTensorCreation(TestCase):
    @unittest.skipUnless(TEST_FLOAT8, "float8 only support on specific MLU version.")
    @testinfo()
    def test_cat_float8_dtypes_and_devices(self, device):
        for dt in (torch.float8_e4m3fn, torch.float8_e5m2):
            x = torch.tensor([[1, 2], [3, 4]], dtype=dt, device=device)

            expected1 = torch.tensor(
                [[1, 2], [3, 4], [1, 2], [3, 4]], dtype=dt, device="cpu"
            )
            self.assertEqual(torch.cat((x, x), 0).to("cpu"), expected1)

            expected2 = torch.tensor(
                [[1, 2, 1, 2], [3, 4, 3, 4]], dtype=dt, device="cpu"
            )
            self.assertEqual(torch.cat((x, x), 1).to("cpu"), expected2)

    @unittest.skipUnless(TEST_FLOAT8, "float8 only support on specific MLU version.")
    @testinfo()
    def test_cast_float8_to_other_dtypes(self, device):
        shape = (2, 3, 4)
        type_list = [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
        other_type_list = [
            torch.half,
            torch.float,
            torch.double,
            torch.int,
            torch.short,
            torch.int8,
            torch.bool,
            torch.long,
            torch.uint8,
        ]
        for ori_t in type_list:
            x = torch.testing.make_tensor(shape, dtype=ori_t, device="cpu")
            for tar_t in other_type_list:
                out_cpu = x.to(tar_t)
                out_mlu = x.to("mlu").to(tar_t)
                self.assertEqual(out_cpu, out_mlu.cpu())

    @unittest.skipUnless(TEST_FLOAT8, "float8 only support on spedific MLU version.")
    @testinfo()
    def test_cast_other_dtypes_to_float8(self, device):
        shape = (2, 3, 4)
        type_list = [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
        other_type_list = [
            torch.half,
            torch.float,
            torch.double,
            torch.int,
            torch.short,
            torch.int8,
            torch.bool,
            torch.long,
            torch.uint8,
        ]
        for ori_t in other_type_list:
            x = torch.testing.make_tensor(shape, dtype=ori_t, device="cpu")
            for tar_t in type_list:
                out_cpu = x.to(tar_t)
                out_mlu = x.to("mlu").to(tar_t)
                self.assertEqual(out_cpu, out_mlu.cpu())

    @unittest.skipUnless(TEST_FLOAT8, "float8 only support on specific MLU version.")
    @testinfo()
    def test_copy_float8_H2D(self):
        shape = (2, 3, 4)
        dtype_list = [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
        for dtype in dtype_list:
            a = torch.testing.make_tensor(shape, dtype=dtype, device="cpu").to("mlu")
            a_data_ptr = a.data_ptr()
            b = torch.randn(shape).to(dtype)
            a.copy_(b)
            self.assertEqual(b.dtype, dtype)
            self.assertEqual(a_data_ptr, a.data_ptr())
            self.assertEqual(a.dtype, dtype)


instantiate_device_type_tests(TestTensorCreation, globals(), only_for=["mlu"])

if __name__ == "__main__":
    run_tests()
