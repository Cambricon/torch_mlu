import os

import sys
import logging
import unittest
import numpy as np
import torch
import torchvision

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


def nms_mlu(boxes, scores, thresh):
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()
    output = torch.ops.torch_mlu.nms3D(boxes, thresh)
    return order[output].contiguous()


def nms_cpu(boxes, scores, thresh):
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()
    output = torch.ops.torch_mlu.nms3D_cpu(boxes, thresh).long()
    return order[output].contiguous()


class TestNmsOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_nms(self):
        boxes_num = [15, 20, 70, 100, 119, 200, 300]
        thresholds = [0.1, 0.2, 0.3, 0.4]
        for box_num in boxes_num:
            for thresh in thresholds:
                torch.manual_seed(12345)
                boxes = torch.randn(box_num, 7, dtype=torch.float32).abs() + 1
                scores = torch.randn(box_num, dtype=torch.float32).abs() + 1
                output_mlu = nms_mlu(boxes.mlu(), scores.mlu(), thresh)
                output_cpu = nms_cpu(boxes, scores, thresh)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nms_exceptions(self):
        input = torch.randn(1, 2).to("mlu")
        ref_msg = r"boxes sizes should be \(batch_size, 7\)\."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = torch.ops.torch_mlu.nms3D(input, 0.1)  # pylint: disable=W0612
        input = torch.randn(2).to("mlu")
        ref_msg = "boxes should be a 2D tensor, got 1D"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = torch.ops.torch_mlu.nms3D(input, 0.1)  # pylint: disable=W0612
        input = torch.randn((1, 7), dtype=torch.half).to("mlu")
        ref_msg = "dets dtype should be float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = torch.ops.torch_mlu.nms3D(input, 0.1)  # pylint: disable=W0612


if __name__ == "__main__":
    unittest.main()
