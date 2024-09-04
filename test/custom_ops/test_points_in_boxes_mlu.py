from __future__ import print_function

import sys
import os
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


def point_in_boxes(points, boxes):
    batch_size, point_nums, _ = points.shape
    _, boxes_nums, _ = boxes.shape
    out = points.new_zeros((batch_size, point_nums), dtype=torch.int).fill_(-1)
    MARGIN = 1e-5
    for batch in range(batch_size):
        for point in range(point_nums):
            index = -1
            for box in range(boxes_nums):
                x, y, z = points[batch][point]
                cx, cy, cz, dx, dy, dz, rz = boxes[batch][box]
                flag = torch.abs(z - cz) <= dz / 2
                flag &= (
                    torch.abs((x - cx) * torch.cos(-rz) - (y - cy) * torch.sin(-rz))
                    < dx / 2 + MARGIN
                )
                flag &= (
                    torch.abs((x - cx) * torch.sin(-rz) + (y - cy) * torch.cos(-rz))
                    < dy / 2 + MARGIN
                )
                if flag:
                    index = box
                    break
            out[batch][point] = index
    return out


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_point_in_boxes_mlu(self):
        shape_list = [
            [(1, 224, 3), (1, 2, 7)],
            [(1, 102400, 3), (1, 1, 7)],
            [(3, 526, 3), (3, 12, 7)],
        ]
        m = torch.ops.torch_mlu.points_in_boxes_mlu
        for shape in shape_list:
            points_shape, boxes_shape = shape
            points = torch.randn(points_shape, dtype=torch.float)
            boxes = torch.randn(boxes_shape, dtype=torch.float)
            points_mlu = points.mlu()
            boxes_mlu = boxes.mlu()
            out_cpu = point_in_boxes(points, boxes)
            out_mlu = m(points_mlu, boxes_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_point_in_boxes_mlu_not_dense(self):
        points_shape = (1, 102400, 3)
        boxes_shape = (1, 2, 7)
        m = torch.ops.torch_mlu.points_in_boxes_mlu
        points = torch.randn(points_shape, dtype=torch.float)
        boxes = torch.randn(boxes_shape, dtype=torch.float)
        points_mlu = points.mlu()
        boxes_mlu = boxes.mlu()
        points_mlu = self.to_non_dense(points_mlu)
        boxes_mlu = self.to_non_dense(boxes_mlu)
        out_cpu = point_in_boxes(points, boxes)
        out_mlu = m(points_mlu, boxes_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)


if __name__ == "__main__":
    unittest.main()
