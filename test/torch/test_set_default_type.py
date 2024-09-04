import os
import sys
import unittest
import torch
import torch_mlu  # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import TestCase  # pylint: disable=C0413,C0411


class SetDefaultType(TestCase):
    def test_print_tensor(self):
        # reset default data type
        torch.set_default_tensor_type(torch.FloatTensor)
        x = torch.tensor([123], device="mlu:0")
        self.assertEqual(x.__repr__(), str(x))
        assert str(x) == "tensor([123], device='mlu:0')"

    def test_default_type_origin(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        self.assertIs(torch.float32, torch.get_default_dtype())
        self.assertIs(torch.float32, torch.mlu.FloatTensor.dtype)

        torch.set_default_dtype(torch.float64)
        self.assertIs(torch.float64, torch.get_default_dtype())

    def test_default_type(self):
        torch.set_default_tensor_type(torch.DoubleTensor)
        a = torch.rand(2, 3, 4, device="mlu")
        self.assertIs(torch.float64, a.dtype)

        torch.set_default_tensor_type(torch.FloatTensor)
        a = torch.rand(2, 3, 4, device="mlu")
        self.assertIs(torch.float32, a.dtype)

        torch.set_default_tensor_type(torch.HalfTensor)
        a = torch.rand(2, 3, 4, device="mlu")
        self.assertIs(torch.float16, a.dtype)

        torch.set_default_device("mlu")
        torch.set_default_dtype(torch.float32)
        a = torch.rand(2, 3, 4)
        self.assertTrue(a.is_mlu)
        self.assertIs(torch.float32, a.dtype)


if __name__ == "__main__":
    unittest.main()
