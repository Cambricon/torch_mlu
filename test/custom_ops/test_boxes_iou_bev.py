from __future__ import print_function

import sys
import logging
import os
import unittest
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
data_dir = os.path.join(cur_dir, "boxes_iou_bev_data/")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

iou_bev = torch.ops.torch_mlu.boxes_iou_bev


class TestBoxesIouBevOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_boxes_iou_bev(self):
        for i in range(16):
            case = torch.load(data_dir + f"boxes_iou_bev_{i}.npy")
            a = case["a"].mlu()
            b = case["b"].mlu()
            result = case["ans"]
            result_mlu = iou_bev(a, b)
            self.assertTensorsEqual(result, result_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_boxes_iou_bev_exceptions(self):
        a = torch.randn(1, 7, 3).mlu()
        b = torch.randn(1, 7).mlu()
        with self.assertRaises(RuntimeError) as info:
            iou_bev(a, b)
        msg = "Boxes_a is not a 2-dims Tensor."
        self.assertEqual(info.exception.args[0], msg)

        a = torch.randn(1, 7).mlu()
        b = torch.randn(1, 7, 3).mlu()
        with self.assertRaises(RuntimeError) as info:
            iou_bev(a, b)
        msg = "Boxes_b is not a 2-dims Tensor."
        self.assertEqual(info.exception.args[0], msg)

        a = torch.randn(1, 6).mlu()
        b = torch.randn(1, 7).mlu()
        with self.assertRaises(RuntimeError) as info:
            iou_bev(a, b)
        msg = "Boxes_a is not an [N, 7] shaped Tensor."
        self.assertEqual(info.exception.args[0], msg)

        a = torch.randn(1, 7).mlu()
        b = torch.randn(1, 6).mlu()
        with self.assertRaises(RuntimeError) as info:
            iou_bev(a, b)
        msg = "Boxes_b is not an [N, 7] shaped Tensor."
        self.assertEqual(info.exception.args[0], msg)

        a = torch.randn(1, 7, dtype=torch.half).mlu()
        b = torch.randn(1, 7, dtype=torch.half).mlu()
        with self.assertRaises(RuntimeError) as info:
            iou_bev(a, b)
        msg = "inputs dtype should be float"
        self.assertEqual(info.exception.args[0], msg)


if __name__ == "__main__":
    unittest.main()
