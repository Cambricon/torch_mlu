from __future__ import print_function

import sys
import logging
import os
import copy
import unittest
import numpy as np
import torch
import torchvision

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestRoiAlignOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_roi_align(self):
        input_shape = [
            (16, 256, 200, 304),
            (16, 256, 100, 152),
            (16, 256, 50, 76),
            (16, 256, 25, 38),
            (16, 256, 200, 304),
            (16, 256, 100, 152),
            (16, 256, 50, 76),
            (16, 256, 25, 38),
        ]
        boxes_shape = [
            (150, 5),
            (740, 5),
            (179, 5),
            (20, 5),
            (81, 5),
            (1, 5),
            (0, 5),
            (0, 5),
        ]
        output_size = [7, 7, 7, 7, 14, 14, 14, 14]
        spatial_scale = [0.25, 0.125, 0.0625, 0.03125, 0.25, 0.125, 0.0625, 0.03125]
        sampling_ratio = [2, 2, 2, 2, 2, 2, 2, 2]
        param = zip(
            input_shape, boxes_shape, output_size, spatial_scale, sampling_ratio
        )
        for i_shape, b_shape, o_size, s_scale, s_ratio in param:
            a = torch.randn(i_shape, dtype=torch.float, requires_grad=True)
            a_ = copy.deepcopy(a)
            boxes_index = np.random.randint(low=0, high=16, size=(b_shape[0],))
            boxes_x1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
            boxes_y1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
            boxes_x2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
            boxes_y2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
            boxes_ = torch.tensor(
                list(zip(boxes_index, boxes_x1, boxes_y1, boxes_x2, boxes_y2))
            ).float()
            if b_shape[0] == 0:
                boxes_ = torch.randn((b_shape), dtype=torch.float)
            out = torchvision.ops.roi_align(
                a,
                boxes=boxes_,
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )
            grad = torch.randn(out.shape, dtype=torch.float)
            out.backward(grad)
            a_mlu = a_.to("mlu")
            boxes_mlu = copy.deepcopy(boxes_).to("mlu")
            grad_mlu = copy.deepcopy(grad).to("mlu")
            out_mlu = torchvision.ops.roi_align(
                a_mlu,
                boxes=boxes_mlu,
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(a.grad, a_.grad, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_roi_align_boxes_list(self):
        input_shape = [(16, 256, 200, 304), (16, 256, 100, 152)]
        boxes_shape = [(150, 5), (740, 5)]
        output_size = [7, 7]
        spatial_scale = [0.25, 0.125]
        sampling_ratio = [2, 2]
        param = zip(
            input_shape, boxes_shape, output_size, spatial_scale, sampling_ratio
        )
        for i_shape, b_shape, o_size, s_scale, s_ratio in param:
            a = torch.randn(i_shape, dtype=torch.float, requires_grad=True)
            a_ = copy.deepcopy(a)
            boxes_x1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
            boxes_y1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
            boxes_x2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
            boxes_y2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
            boxes1 = torch.tensor(
                list(zip(boxes_x1, boxes_y1, boxes_x2, boxes_y2))
            ).float()
            boxes_ = [boxes1, boxes1, boxes1]
            if b_shape[0] == 0:
                boxes_ = torch.randn((b_shape), dtype=torch.float)
            out = torchvision.ops.roi_align(
                a,
                boxes=boxes_,
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )
            grad = torch.randn(out.shape, dtype=torch.float)
            out.backward(grad)
            a_mlu = a_.to("mlu")
            boxes_mlu = [boxes1.to("mlu"), boxes1.to("mlu"), boxes1.to("mlu")]
            grad_mlu = copy.deepcopy(grad).to("mlu")
            out_mlu = torchvision.ops.roi_align(
                a_mlu,
                boxes=boxes_mlu,
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(a.grad, a_.grad, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_roi_align_channels_last(self):
        input_shape = [(16, 256, 200, 304)]
        boxes_shape = [
            (150, 5),
        ]
        output_size = [
            7,
        ]
        spatial_scale = [
            0.25,
        ]
        sampling_ratio = [
            2,
        ]
        param = zip(
            input_shape, boxes_shape, output_size, spatial_scale, sampling_ratio
        )
        for i_shape, b_shape, o_size, s_scale, s_ratio in param:
            a = torch.randn(i_shape, dtype=torch.float, requires_grad=True)
            a_ = a.to(memory_format=torch.channels_last)

            boxes_index = np.random.randint(low=0, high=16, size=(b_shape[0],))
            boxes_x1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
            boxes_y1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
            boxes_x2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
            boxes_y2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
            boxes = torch.tensor(
                list(zip(boxes_index, boxes_x1, boxes_y1, boxes_x2, boxes_y2))
            ).float()
            out = torchvision.ops.roi_align(
                a_,
                boxes=boxes,
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )
            grad = torch.randn(out.shape, dtype=torch.float)
            out.backward(grad)
            grad_cpu = copy.deepcopy(a.grad)
            a.grad.zero_()

            a_mlu = a_.to("mlu")
            boxes_mlu = copy.deepcopy(boxes).to("mlu")
            grad_mlu = copy.deepcopy(grad).to("mlu")
            out_mlu = torchvision.ops.roi_align(
                a_mlu,
                boxes=boxes_mlu,
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(grad_cpu, a.grad, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_roi_align_no_dense(self):
        input_shape = [(16, 256, 200, 304)]
        boxes_shape = [
            (150, 5),
        ]
        output_size = [
            7,
        ]
        spatial_scale = [
            0.25,
        ]
        sampling_ratio = [
            2,
        ]
        param = zip(
            input_shape, boxes_shape, output_size, spatial_scale, sampling_ratio
        )
        for i_shape, b_shape, o_size, s_scale, s_ratio in param:
            a = torch.randn(i_shape, dtype=torch.float, requires_grad=True)

            boxes_index = np.random.randint(low=0, high=16, size=(b_shape[0],))
            boxes_x1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
            boxes_y1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
            boxes_x2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
            boxes_y2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
            boxes = torch.tensor(
                list(zip(boxes_index, boxes_x1, boxes_y1, boxes_x2, boxes_y2))
            ).float()
            out = torchvision.ops.roi_align(
                a[..., ::2],
                boxes=boxes,
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )
            grad = torch.randn(out.shape, dtype=torch.float)
            out.backward(grad)
            grad_cpu = copy.deepcopy(a.grad)
            a.grad.zero_()

            a_mlu = a.to("mlu")
            boxes_mlu = copy.deepcopy(boxes).to("mlu")
            grad_mlu = copy.deepcopy(grad).to("mlu")
            out_mlu = torchvision.ops.roi_align(
                a_mlu[..., ::2],
                boxes=boxes_mlu,
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(grad_cpu, a.grad, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_roialign_exception(self):
        a = torch.randn((1, 1, 3, 3), dtype=torch.float).to("mlu")
        boxes_ = torch.randn((2, 5), device="mlu").to(torch.double)
        ref_msg = "Expected tensor for argument #1 'input' to have the same type as"
        ref_msg = (
            ref_msg
            + " tensor for argument #2 'rois'; but type mluFloatType does not equal"
        )
        ref_msg = (
            ref_msg + r" mluDoubleType \(while checking arguments for cnnl_roi_align\)"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torchvision.ops.roi_align(
                a, boxes=boxes_, output_size=1, spatial_scale=0.125, sampling_ratio=2
            )

    @testinfo()
    def test_roi_align_with_amp(self):
        a = torch.randn((16, 256, 200, 304))
        a_mlu = a.to("mlu")

        b_shape = (150, 5)
        boxes_index = np.random.randint(low=0, high=16, size=(b_shape[0],))
        boxes_x1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
        boxes_y1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
        boxes_x2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
        boxes_y2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
        boxes = torch.tensor(
            list(zip(boxes_index, boxes_x1, boxes_y1, boxes_x2, boxes_y2))
        ).float()

        boxes_mlu = boxes.mlu()
        o_size = 7
        s_scale = 0.25
        s_ratio = 2
        with torch.amp.autocast(device_type="mlu"):
            out_cpu = torchvision.ops.roi_align(
                a,
                boxes=boxes,
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )

            out_mlu = torchvision.ops.roi_align(
                a_mlu,
                boxes=boxes_mlu.half(),
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )

            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_roi_align_bfloat16(self):
        input_shape = [
            (16, 256, 200, 304),
            (16, 256, 100, 152),
            (16, 256, 50, 76),
            (16, 256, 25, 38),
            (16, 256, 200, 304),
            (16, 256, 100, 152),
            (16, 256, 50, 76),
            (16, 256, 25, 38),
        ]
        boxes_shape = [
            (150, 5),
            (740, 5),
            (179, 5),
            (20, 5),
            (81, 5),
            (1, 5),
            (0, 5),
            (0, 5),
        ]
        output_size = [7, 7, 7, 7, 14, 14, 14, 14]
        spatial_scale = [0.25, 0.125, 0.0625, 0.03125, 0.25, 0.125, 0.0625, 0.03125]
        sampling_ratio = [2, 2, 2, 2, 2, 2, 2, 2]
        param = zip(
            input_shape, boxes_shape, output_size, spatial_scale, sampling_ratio
        )
        for i_shape, b_shape, o_size, s_scale, s_ratio in param:
            a = torch.randn(i_shape, dtype=torch.float, requires_grad=True)
            a_ = copy.deepcopy(a).to(torch.bfloat16)
            boxes_index = np.random.randint(low=0, high=16, size=(b_shape[0],))
            boxes_x1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
            boxes_y1 = np.random.randint(low=0, high=15, size=(b_shape[0],))
            boxes_x2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
            boxes_y2 = np.random.randint(low=15, high=32, size=(b_shape[0],))
            boxes_ = torch.tensor(
                list(zip(boxes_index, boxes_x1, boxes_y1, boxes_x2, boxes_y2))
            ).float()
            if b_shape[0] == 0:
                boxes_ = torch.randn((b_shape), dtype=torch.float)
            out = torchvision.ops.roi_align(
                a,
                boxes=boxes_,
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )
            grad = torch.randn(out.shape, dtype=torch.float)
            out.backward(grad)
            a_mlu = a_.to("mlu").to(torch.bfloat16)
            boxes_mlu = copy.deepcopy(boxes_).to("mlu").to(torch.bfloat16)
            grad_mlu = copy.deepcopy(grad).to("mlu").to(torch.bfloat16)
            out_mlu = torchvision.ops.roi_align(
                a_mlu,
                boxes=boxes_mlu,
                output_size=o_size,
                spatial_scale=s_scale,
                sampling_ratio=s_ratio,
            )
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(out, out_mlu.cpu().float(), 0.003, use_MSE=True)
            self.assertTensorsEqual(a.grad, a_.grad.float(), 0.003, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
