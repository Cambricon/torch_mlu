from __future__ import print_function

import sys
import logging
import os
import unittest
import numpy as np
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
data_dir = os.path.join(cur_dir, "boxes_overlap_bev_data/")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


# Add this function for segment of network test, Using later.
def boxes_iou3d_test(boxes_a, boxes_b, flag):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    if flag:
        overlaps_bev = torch.ops.torch_mlu.boxes_overlap_bev(
            boxes_a.contiguous(), boxes_b.contiguous()
        )
    else:
        overlaps_bev = torch.ops.torch_mlu.boxes_overlap_bev_cpu(
            boxes_a.contiguous(), boxes_b.contiguous()
        ).mlu()

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


class TestBoxesOverlapBevOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_boxes_overlap_bev(self):
        for i in range(1, 6):
            input = torch.Tensor(
                np.load(data_dir + "boxes_overlap_a_" + str(i) + ".npy")
            ).mlu()
            other = torch.Tensor(
                np.load(data_dir + "boxes_overlap_b_" + str(i) + ".npy")
            ).mlu()
            output_mlu = torch.ops.torch_mlu.boxes_overlap_bev(input, other)
            output_cuda = torch.Tensor(
                np.load(data_dir + "overlaps_bev_" + str(i) + ".npy")
            )
            self.assertTensorsEqual(output_cuda, output_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_boxes_overlap_bev_not_dense(self):  # pylint: disable=R0201
        shape_list = (
            ((279, 14), (279, 14)),
            ((279, 14), (200, 14)),
            ((54, 14), (50, 14)),
            ((60, 14), (140, 14)),
        )
        for shape in shape_list:
            input = torch.randn(shape[0]).mlu()[..., ::2]
            other = torch.randn(shape[1]).mlu()[..., ::2]
            output_mlu = torch.ops.torch_mlu.boxes_overlap_bev(input, other)
            output_mlu.cpu()


if __name__ == "__main__":
    unittest.main()
