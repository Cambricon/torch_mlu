from __future__ import print_function

import sys
import os
import unittest
import logging
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0411, C0413

logging.basicConfig(level=logging.DEBUG)


class TestIsinOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_isin(self):
        type_list = [torch.float, torch.int, torch.long, torch.double, torch.half]
        invert_list = [True, False]
        for t in type_list:
            for invert in invert_list:
                for shape1, shape2 in [
                    ((), ()),
                    ((), (1)),
                    ((), (10)),
                    ((4), ()),
                    ((8), (1)),
                    ((5), (5)),
                    ((7), (3)),
                    ((2, 3, 4), (3, 4)),
                    ((1, 11, 1, 4), (11, 1, 5, 1)),
                    ((25, 14, 7, 15, 2), (1)),
                ]:
                    # test invert=Fasle
                    x = torch.randn(shape1).to(t)
                    y = torch.randn(shape2).to(t)
                    if t == torch.half:
                        out_cpu = torch.isin(
                            x.to(torch.float), y.to(torch.float), invert=invert
                        )
                    else:
                        out_cpu = torch.isin(x, y, invert=invert)
                    out_mlu = torch.isin(self.to_mlu(x), self.to_mlu(y), invert=invert)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_isin_assume_unique(self):
        type_list = [torch.float, torch.int, torch.long, torch.double, torch.half]
        invert_list = [True, False]
        for t in type_list:
            for invert in invert_list:
                shape1 = (2, 3, 4)
                shape2 = (3, 7)
                x = torch.randperm(24).view(shape1).to(t)
                y = torch.randperm(21).view(shape2).to(t)
                if t == torch.half:
                    out_cpu = torch.isin(
                        x.to(torch.float),
                        y.to(torch.float),
                        assume_unique=True,
                        invert=invert,
                    )
                else:
                    out_cpu = torch.isin(x, y, assume_unique=True, invert=invert)
                out_mlu = torch.isin(
                    self.to_mlu(x), self.to_mlu(y), assume_unique=True, invert=invert
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_isin_channels_last(self):
        type_list = [torch.float, torch.int, torch.long, torch.double]
        for t in type_list:
            for shape1, shape2 in [
                ((2, 3, 24, 30), (1, 1, 1, 30)),
                ((16, 8, 8, 32), (16, 8, 8, 32)),
            ]:
                x = torch.randn(shape1).to(t).to(memory_format=torch.channels_last)
                y = torch.randn(shape2).to(t).to(memory_format=torch.channels_last)
                out_cpu = torch.isin(x, y)
                out_mlu = torch.isin(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

                # mixed memory format
                z = torch.randn(shape2).to(t)
                out_cpu = torch.isin(x, z)
                out_mlu = torch.isin(self.to_mlu(x), self.to_mlu(z))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_isin_not_dense(self):
        type_list = [torch.float, torch.int, torch.long, torch.double]
        for t in type_list:
            for shape1, shape2 in [
                ((2, 3, 24, 30), (1, 1, 1, 30)),
                ((16, 8, 8, 32), (16, 8, 8, 32)),
            ]:
                x = torch.randn(shape1).to(t)[:, :, :, :15]
                y = torch.randn(shape2).to(t)[:, :, :, :15]
                out_cpu = torch.isin(x, y)
                out_mlu = torch.isin(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_isin_out(self):
        type_list = [torch.float, torch.int, torch.long, torch.double]
        for t in type_list:
            for shape1, shape2 in [
                ((), ()),
                ((), (1)),
                ((), (25, 14, 7, 15, 2, 1)),
                ((1), (256, 7)),
                ((5), (5)),
                ((2, 3, 4), (3, 4)),
                ((25, 14, 7, 15, 2, 1, 1), (1)),
                ((1), (25, 14, 7, 15, 2, 1, 1, 1)),
            ]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_tmpcpu = torch.zeros(shape1, dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape1, dtype=torch.bool).to("mlu")
                torch.isin(x, y, out=out_tmpcpu)
                torch.isin(self.to_mlu(x), self.to_mlu(y), out=out_tmpmlu)
                self.assertTensorsEqual(
                    out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_isin_scalar(self):
        type_list = [torch.float, torch.int, torch.long, torch.double]
        for t in type_list:
            for shape in [(), (256, 144, 7), (1), (256, 7), (2, 3, 4)]:
                x = torch.randn(shape).to(t)
                y = torch.randn(()).to(t).item()
                out_cpu = torch.isin(x, y)
                out_mlu = torch.isin(self.to_mlu(x), y)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
                if (x.numel() + 3) < 49152:  # CNNLCORE-4671
                    out_cpu_new = torch.isin(y, x)
                    out_mlu_new = torch.isin(y, self.to_mlu(x))
                    self.assertTensorsEqual(
                        out_cpu_new.float(),
                        out_mlu_new.cpu().float(),
                        0.0,
                        use_MSE=True,
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_isin_out_scalar(self):
        type_list = [torch.float, torch.int, torch.long, torch.double]
        for t in type_list:
            for shape in [
                (),
                (256, 144, 7, 15, 2, 1),
                (1),
                (256, 7),
                (2, 3, 4),
                (117, 1, 5, 1, 5, 1, 3, 1),
                (256, 144, 7, 15, 2, 1, 1, 1),
            ]:
                x = torch.randn(shape).to(t)
                y = torch.randn(()).to(t).item()
                out_tmpcpu = torch.zeros(shape, dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape, dtype=torch.bool).to("mlu")
                torch.isin(x, y, out=out_tmpcpu)
                torch.isin(self.to_mlu(x), y, out=out_tmpmlu)
                self.assertTensorsEqual(
                    out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_isin_exception(self):
        a = torch.randn(1, 2, 3, 1, 1, 1, 1, 1).float().to("mlu")
        b = torch.randn(1, 2, 3, 1, 1, 1, 1, 1).float().to("mlu")
        ref_msg = r"The dimensions of all input tensors should be less than 7, but currently, "
        ref_msg = (
            ref_msg
            + "the dimension of 'elements' is 8 and the dimension of 'test_elements' is 8."
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.isin(a, b)

    # TODO(CNNLCORE-17757): cnnl_unique not implement large tensor.
    @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    def test_isin_large(self):
        left = torch.testing.make_tensor(
            (2, 1024, 1024, 1024), dtype=torch.float, device="cpu"
        )
        right = torch.testing.make_tensor(
            (2, 1024, 1024, 1024), dtype=torch.float, device="cpu"
        )
        out_cpu = torch.isin(left, right)
        out_mlu = torch.isin(left.mlu(), right.mlu())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
