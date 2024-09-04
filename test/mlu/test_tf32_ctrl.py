import unittest
from unittest import TestCase
import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

import torch
import torch_mlu


class TF32TestCases(TestCase):
    def test_mlu_allow_tf32_get_set(self):
        orig = torch.backends.cnnl.allow_tf32
        self.assertEqual(orig, True)
        torch.backends.cnnl.allow_tf32 = not orig
        self.assertEqual(torch.backends.cnnl.allow_tf32, False)
        torch.backends.cnnl.allow_tf32 = orig
        orig = torch.backends.mlu.custom.allow_tf32
        self.assertEqual(orig, False)
        torch.backends.mlu.custom.allow_tf32 = not orig
        self.assertEqual(torch.backends.mlu.custom.allow_tf32, True)
        torch.backends.mlu.custom.allow_tf32 = orig

    def test_float32_matmul_precision_get_set(self):
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        skip_tf32_cnmatmul = "TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE"]
        )
        if not skip_tf32_cnmatmul:
            orig = torch.backends.mlu.matmul.allow_tf32
            self.assertEqual(orig, False)
            torch.backends.mlu.matmul.allow_tf32 = not orig
            self.assertEqual(torch.backends.mlu.matmul.allow_tf32, True)
            torch.backends.cuda.matmul.allow_tf32 = False
            self.assertEqual(torch.backends.mlu.matmul.allow_tf32, True)
            torch.backends.mlu.matmul.allow_tf32 = orig
        for p in ("medium", "high"):
            torch.set_float32_matmul_precision(p)
            self.assertEqual(torch.get_float32_matmul_precision(), p)
            if not skip_tf32_cnmatmul:
                self.assertTrue(torch.backends.mlu.matmul.allow_tf32)
        torch.set_float32_matmul_precision("highest")
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        if not skip_tf32_cnmatmul:
            self.assertFalse(torch.backends.mlu.matmul.allow_tf32)

    def test__C_api(self):
        self.assertFalse(torch._C._get_cnmatmul_allow_tf32())
        torch._C._set_cnmatmul_allow_tf32(True)
        self.assertTrue(torch._C._get_cnmatmul_allow_tf32())
        torch._C._set_cnmatmul_allow_tf32(False)
        self.assertFalse(torch._C._get_cnmatmul_allow_tf32())


if __name__ == "__main__":
    unittest.main()
