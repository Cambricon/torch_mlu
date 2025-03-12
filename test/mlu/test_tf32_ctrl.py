import unittest
from unittest import TestCase
import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

# must set before import torch/torch_mlu
os.environ["TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE"] = "1"

import torch
import torch_mlu


class TF32TestCases(TestCase):
    def test_mlu_allow_tf32_get_set(self):
        # torch.testing._internal.common_utils call
        # torch.backends.disable_global_flags() function,
        # this will disable setter, so need to use flags().
        orig = torch.backends.cnnl.allow_tf32
        self.assertEqual(orig, True)
        with torch.backends.cnnl.flags(allow_tf32=not orig):
            self.assertEqual(torch.backends.cnnl.allow_tf32, False)
        orig = torch.backends.mlu.custom.allow_tf32
        self.assertEqual(orig, False)
        torch.backends.mlu.custom.allow_tf32 = not orig
        self.assertEqual(torch.backends.mlu.custom.allow_tf32, True)
        torch.backends.mlu.custom.allow_tf32 = orig

    def test_1_float32_matmul_precision_init(self):
        # env set tf32 must before all api set
        self.assertEqual(torch.get_float32_matmul_precision(), "high")
        torch.set_float32_matmul_precision("highest")
        self.assertFalse(torch.backends.mlu.matmul.allow_tf32)
        torch.set_float32_matmul_precision("high")
        self.assertTrue(torch.backends.mlu.matmul.allow_tf32)

    def test_2_float32_matmul_precision_get_set(self):
        skip_tf32_cnmatmul = "TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE" in os.environ and int(
            os.environ["TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE"]
        )
        self.assertTrue(skip_tf32_cnmatmul)
        self.assertTrue(torch.backends.mlu.matmul.allow_tf32)
        torch.backends.mlu.matmul.allow_tf32 = False
        self.assertFalse(torch.backends.mlu.matmul.allow_tf32)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        orig = torch.backends.mlu.matmul.allow_tf32
        self.assertEqual(orig, False)
        torch.backends.mlu.matmul.allow_tf32 = not orig
        self.assertEqual(torch.backends.mlu.matmul.allow_tf32, True)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.assertEqual(torch.backends.mlu.matmul.allow_tf32, True)
        torch.backends.mlu.matmul.allow_tf32 = orig
        torch.set_float32_matmul_precision("highest")
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.mlu.matmul.allow_tf32)
        for p in ("medium", "high"):
            torch.set_float32_matmul_precision(p)
            self.assertEqual(torch.get_float32_matmul_precision(), p)
            self.assertTrue(torch.backends.mlu.matmul.allow_tf32)

    def test_3__C_api(self):
        self.assertTrue(torch._C._get_cnmatmul_allow_tf32())
        torch._C._set_cnmatmul_allow_tf32(False)
        self.assertFalse(torch._C._get_cnmatmul_allow_tf32())
        torch._C._set_cnmatmul_allow_tf32(True)
        self.assertTrue(torch._C._get_cnmatmul_allow_tf32())


if __name__ == "__main__":
    unittest.main()
