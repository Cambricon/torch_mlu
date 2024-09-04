import sys
import os
from itertools import product
import unittest
import logging
import torch
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)


class TestDotOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_dot(self):
        dtype_list = [(torch.half, 5e-2), (torch.float, 3e-3)]
        shape_list = [10, 12, 13, 1]
        func_list = [self.to_non_dense, lambda x: x]
        for dtype_err, in_shape, func in product(dtype_list, shape_list, func_list):
            dtype, err = dtype_err
            x_cpu = torch.randn(in_shape).to(dtype)
            y_cpu = torch.rand(in_shape).to(dtype)
            x_mlu = x_cpu.to("mlu")
            y_mlu = y_cpu.to("mlu")
            out_cpu = torch.dot(func(x_cpu).float(), func(y_cpu).float())
            out_mlu = torch.dot(func(x_mlu), func(y_mlu))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )

        # test sliced output
        x_cpu = torch.randn(10)
        y_cpu = torch.rand(10)
        out_cpu = torch.randn(10, 50)
        x_mlu = x_cpu.to("mlu")
        y_mlu = y_cpu.to("mlu")
        out_mlu = out_cpu.to("mlu")
        torch.dot(x_cpu, y_cpu, out=out_cpu[:, :25])
        torch.dot(x_mlu, y_mlu, out=out_mlu[:, :25])
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_dot_ori(self):
        def _test_dot_vdot_vs_numpy(device, dtype, torch_fn, np_fn, err):
            def check(x, y):
                # Compare with numpy
                res = torch_fn(x, y)
                ref = torch.from_numpy(
                    np.array(np_fn(x.cpu().numpy(), y.cpu().numpy()))
                )
                self.assertTensorsEqual(
                    res.float().cpu(), ref.float(), err, use_MSE=True
                )

                # Test out variant
                out = torch.empty_like(res)
                torch_fn(x, y, out=out)
                self.assertTensorsEqual(out.float(), res.float(), err, use_MSE=True)

            # Empty
            x = torch.tensor([], dtype=dtype, device=device)
            y = torch.tensor([], dtype=dtype, device=device)
            check(x, y)

            # Contiguous
            x = torch.randn(10, dtype=dtype, device=device)
            y = torch.randn(10, dtype=dtype, device=device)
            check(x, y)

            # 0 strided
            y = torch.randn(1, dtype=dtype, device=device).expand(10)
            check(x, y)

            # 2 strided
            check(x[::2], y[::2])

        device = "mlu"
        dtype_list = [(torch.half, 5e-2), (torch.float, 3e-3)]
        for dtype, err in dtype_list:
            _test_dot_vdot_vs_numpy(device, dtype, torch.dot, np.dot, err)

        def _test_dot_vdot_invalid_args(device, torch_fn):
            def check(x, y, regex):
                with self.assertRaisesRegex(RuntimeError, regex):
                    torch_fn(x, y)

            x = torch.randn(1, dtype=torch.float, device=device)
            y = torch.randn(3, device=device).to(torch.double)

            check(x, y, "dot : expected both vectors to have same dtype")
            check(x.reshape(1, 1), y, "1D tensors expected")
            check(x.expand(9), y.to(x.dtype), "inconsistent tensor size")

            x_cpu = x.expand(3).cpu()
            check(
                x_cpu,
                y.to(x.dtype),
                "Expected all tensors to be on the same device,"
                + " but found at least two devices, cpu and mlu:0!",
            )

        _test_dot_vdot_invalid_args(device, torch.dot)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_dot_bfloat16(self):
        in_shape = [10]
        x_cpu = torch.randn(in_shape).to(torch.bfloat16)
        y_cpu = torch.rand(in_shape).to(torch.bfloat16)
        x_mlu = x_cpu.to("mlu")
        y_mlu = y_cpu.to("mlu")
        out_cpu = torch.dot(x_cpu.float(), y_cpu.float())
        out_mlu = torch.dot(x_mlu, y_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 5e-2, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
