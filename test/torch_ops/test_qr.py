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
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestQrOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_tensor_empty(self):
        shape_list = [(0, 0), (7, 0), (5, 3, 0), (7, 5, 0, 3)]
        qr_funcs = [torch.qr, torch.linalg.qr]
        for qr_func in qr_funcs:
            for shape in shape_list:
                for t in [torch.double, torch.float, torch.half]:
                    x = torch.randn(shape, dtype=torch.float)
                    cpu_q, cpu_r = qr_func(x)
                    mlu_q, mlu_r = qr_func(self.to_mlu_dtype(x, t))
                    self.assertEqual(cpu_q.shape, mlu_q.shape)
                    self.assertEqual(cpu_q, mlu_q.cpu().to(torch.float))
                    self.assertEqual(cpu_r.shape, mlu_r.shape)
                    self.assertEqual(cpu_r, mlu_r.cpu().to(torch.float))

    def _test_qr(self, shape, some, device, dtype):
        qr_funcs = [
            (torch.qr, some),
            (torch.linalg.qr, "reduced" if some else "complete"),
        ]
        for qr_func in qr_funcs:
            cpu_dtype = dtype
            if dtype == torch.half or dtype == torch.bfloat16:
                cpu_dtype = torch.float32
            cpu_tensor = torch.randn(shape, device="cpu", dtype=torch.half)
            device_tensor = cpu_tensor.to(dtype).to(device=device)
            resq, resr = qr_func[0](cpu_tensor.to(cpu_dtype), qr_func[1])
            outq, outr = qr_func[0](device_tensor, qr_func[1])
            if dtype == torch.half or dtype == torch.bfloat16:
                outq = outq.to(torch.float32)
                outr = outr.to(torch.float32)
            m = min(cpu_tensor.shape[-2:])
            self.assertEqual(
                resq[..., :m].abs(), outq[..., :m].abs(), atol=3e-2, rtol=0
            )
            self.assertEqual(
                resr[..., :m].abs(), outr[..., :m].abs(), atol=3e-2, rtol=0
            )

    def _test_qr_not_dense(self, shape, some, device, dtype):
        qr_funcs = [
            (torch.qr, some),
            (torch.linalg.qr, "reduced" if some else "complete"),
        ]
        for qr_func in qr_funcs:
            cpu_dtype = dtype
            if dtype == torch.half:
                cpu_dtype = torch.float32
            cpu_tensor = torch.empty(0)
            if len(shape) == 2:
                cpu_tensor = torch.randn(shape, device="cpu", dtype=torch.half)[
                    :, : int(shape[-1] / 2)
                ]
            elif len(shape) == 3:
                cpu_tensor = torch.randn(shape, device="cpu", dtype=torch.half)[
                    :, :, : int(shape[-1] / 2)
                ]
            elif len(shape) == 4:
                cpu_tensor = torch.randn(shape, device="cpu", dtype=torch.half)[
                    :, :, :, : int(shape[-1] / 2)
                ]
            device_tensor = cpu_tensor.to(dtype).to(device=device)
            resq, resr = qr_func[0](cpu_tensor.to(cpu_dtype), qr_func[1])
            outq, outr = qr_func[0](device_tensor, qr_func[1])
            if dtype == torch.half:
                outq = outq.to(torch.float32)
                outr = outr.to(torch.float32)
            m = min(cpu_tensor.shape[-2:])
            self.assertEqual(
                resq[..., :m].abs(), outq[..., :m].abs(), atol=3e-2, rtol=0
            )
            self.assertEqual(
                resr[..., :m].abs(), outr[..., :m].abs(), atol=3e-2, rtol=0
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_qr_square(self, device="mlu"):
        for dtype in [torch.double, torch.float, torch.half]:
            self._test_qr((10, 10), True, device, dtype)
            self._test_qr_not_dense((10, 20), True, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_qr_tall_some(self, device="mlu"):
        for dtype in [torch.double, torch.float, torch.half]:
            self._test_qr((20, 5), True, device, dtype)
            self._test_qr_not_dense((20, 10), True, device, dtype)
            self._test_qr((5, 20), True, device, dtype)
            self._test_qr_not_dense((5, 40), True, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_qr_tall_all(self, device="mlu"):
        for dtype in [torch.double, torch.float, torch.half]:
            self._test_qr((20, 5), False, device, dtype)
            self._test_qr_not_dense((20, 10), False, device, dtype)
            self._test_qr((5, 20), False, device, dtype)
            self._test_qr_not_dense((5, 40), False, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_qr_some_3d(self, device="mlu"):
        for dtype in [torch.double, torch.float, torch.half]:
            self._test_qr((5, 7, 3), True, device, dtype)
            self._test_qr_not_dense((5, 7, 6), True, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_qr_tall_all_4d(self, device="mlu"):
        for dtype in [torch.double, torch.float, torch.half]:
            self._test_qr((7, 5, 3, 7), True, device, dtype)
            self._test_qr_not_dense((7, 5, 3, 7 * 2), True, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_qr_out(self):
        shape_list = [(5, 7, 3), (7, 5, 3, 3)]
        dtypes = [torch.double, torch.float, torch.half]
        channel_first = [True, False]
        qr_funcs = [torch.qr, torch.linalg.qr]
        for qr_func in qr_funcs:
            for dtype in dtypes:
                for shape in shape_list:
                    for channel in channel_first:
                        cpu_dtype = dtype
                        if dtype == torch.half:
                            cpu_dtype = torch.float32
                        x = torch.randn(shape, dtype=torch.half)
                        resq, resr = qr_func(x.to(cpu_dtype))
                        if channel is False:
                            x = self.convert_to_channel_last(x)
                        outq = torch.tensor((), dtype=dtype).to("mlu")
                        outr = torch.tensor((), dtype=dtype).to("mlu")
                        qr_func(self.to_mlu(x.to(dtype)), out=(outq, outr))
                        if dtype == torch.half:
                            outq = outq.to(torch.float32)
                            outr = outr.to(torch.float32)
                        m = min(x.shape[-2:])
                        self.assertEqual(
                            resq[..., :m].abs(), outq[..., :m].abs(), atol=3e-2, rtol=0
                        )
                        self.assertEqual(
                            resr[..., :m].abs(), outr[..., :m].abs(), atol=3e-2, rtol=0
                        )

    # @unittest.skip("not test")
    @testinfo()
    def test_linalg_qr(self):
        mode_list = ["reduced", "complete", "r"]
        shape_list = [
            (0, 0),
            (7, 0),
            (5, 3, 0),
            (7, 5, 0, 3),
            (7, 3),
            (5, 7, 3),
            (7, 5, 3, 7),
        ]
        for shape in shape_list:
            for mode in mode_list:
                a = torch.randn(shape)
                q_cpu, r_cpu = torch.linalg.qr(a, mode=mode)
                q, r = torch.linalg.qr(a.to("mlu"), mode=mode)
                out = (
                    torch.empty((0), dtype=torch.float, device="mlu"),
                    torch.empty((0), dtype=torch.float, device="mlu"),
                )
                q2, r2 = torch.linalg.qr(a.to("mlu"), mode=mode, out=out)
                self.assertEqual(q_cpu, q)
                self.assertEqual(r_cpu, r)
                self.assertEqual(q2, q)
                self.assertEqual(r2, r)
                self.assertEqual(q2, out[0])
                self.assertEqual(r2, out[1])

    # @unittest.skip("not test")
    @testinfo()
    def test_qr_exception(self):
        a = torch.randn(2).to("mlu")
        with self.assertRaises(RuntimeError) as info:
            torch.qr(a)
        msg = f"linalg.qr: The input tensor A must have at least 2 dimensions."
        self.assertEqual(info.exception.args[0], msg)

        mode = "hello"
        a = torch.randn(2, 3).to("mlu")
        with self.assertRaises(RuntimeError) as info:
            torch.linalg.qr(a, mode=mode)
        msg0 = f"qr received unrecognized mode '{mode}' but expected"
        msg1 = " one of 'reduced' (default), 'r', or 'complete'"
        msg = msg0 + msg1
        self.assertEqual(info.exception.args[0], msg)

        a = torch.randn(2, 2).mlu()
        q = torch.randn(2, 2).half().mlu()
        r = torch.randn(2, 2).double().mlu()
        msg = "Expected out tensor to have dtype Float, but got Half instead"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.qr(a, out=(q, r))
        msg = "Expected out tensor to have dtype Float, but got Double instead"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.qr(a, out=(r, q))

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_qr_bfloat16(self, device="mlu"):
        dtype = torch.bfloat16
        self._test_qr((10, 10), True, device, dtype)


if __name__ == "__main__":
    unittest.main()
