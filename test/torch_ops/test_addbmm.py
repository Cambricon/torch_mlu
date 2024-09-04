from __future__ import print_function
import logging
import unittest
import sys
import os
import itertools
import math
import torch
import torch_mlu
from torch.testing import make_tensor

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase, TEST_BFLOAT16  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)


class TestAddBmmOp(TestCase):
    def _test_addbmm(self, func, b1, b2, ref, out_tensor):
        getattr(out_tensor, func + "_")(b1, b2)
        self.assertEqual(out_tensor.cpu(), ref)

        res3 = out_tensor.clone()

        getattr(out_tensor, func + "_")(1, b1, b2)
        self.assertEqual(out_tensor.cpu(), ref * 2)

        getattr(res3, func + "_")(b1, b2, beta=1)
        self.assertEqual(out_tensor.cpu(), res3)

        getattr(out_tensor, func + "_")(1.0, 0.5, b1, b2)
        self.assertEqual(out_tensor.cpu(), ref * 2.5)

        getattr(res3, func + "_")(b1, b2, beta=1.0, alpha=0.5)
        self.assertEqual(out_tensor.cpu(), res3.cpu())

        self.assertEqual(
            out_tensor.cpu(), getattr(torch, func)(1, out_tensor, 0, b1, b2).cpu()
        )

        res = getattr(torch, func)(out_tensor, b1, b2, beta=1, alpha=0.5)
        self.assertEqual(res.cpu(), ref * 3)

        nan = torch.full_like(out_tensor, math.nan)
        res = getattr(torch, func)(nan, b1, b2, beta=0, alpha=1)
        self.assertEqual(res.cpu(), ref)

        res = getattr(torch, func)(out_tensor, b1, b2, beta=0.1, alpha=0.5)
        self.assertEqual(res.cpu(), out_tensor.cpu() * 0.1 + 0.5 * ref)

        res = torch.full_like(out_tensor, math.nan)
        getattr(torch, func)(nan, b1, b2, beta=0, out=res)
        self.assertEqual(res.cpu(), ref)

    # @unittest.skip("not test")
    @testinfo()
    def test_addbmm(self):
        num_batches = 2
        M, N, O = 2, 3, 4

        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2])

        def generate_tensor(dtype, device):
            # transposed tensors
            for perm1, perm2 in itertools.product(
                itertools.permutations((0, 1, 2)), repeat=2
            ):
                for perm3 in itertools.permutations((0, 1)):
                    b1 = make_tensor(
                        (num_batches, M, N), device=device, dtype=dtype, low=-1, high=1
                    )
                    b2 = make_tensor(
                        (num_batches, N, O), device=device, dtype=dtype, low=-1, high=1
                    )
                    b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                    b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                    ref = (
                        torch.from_numpy(
                            b1.cpu().to(dtype).numpy() @ b2.cpu().to(dtype).numpy()
                        )
                        .to(dtype)
                        .sum(0)
                    )
                    out_tensor = (
                        torch.zeros_like(ref, device=device)
                        .permute(perm3)
                        .contiguous()
                        .permute(perm3)
                    )
                    yield b1, b2, ref, out_tensor

            # broadcasting tensors
            for s1, s2, s3, s4, s5, s6 in itertools.product((True, False), repeat=6):
                shape1 = (num_batches if s1 else 1, M if s2 else 1, N if s3 else 1)
                shape2 = (num_batches if s4 else 1, N if s5 else 1, O if s6 else 1)
                b1 = make_tensor(
                    shape1, device=device, dtype=dtype, low=-1, high=1
                ).expand(num_batches, M, N)
                b2 = make_tensor(
                    shape2, device=device, dtype=dtype, low=-1, high=1
                ).expand(num_batches, N, O)
                ref = (
                    torch.from_numpy(
                        b1.cpu().to(dtype).numpy() @ b2.cpu().to(dtype).numpy()
                    )
                    .to(dtype)
                    .sum(0)
                )
                out_tensor = torch.zeros_like(ref, device=device)
                yield b1, b2, ref, out_tensor

            # zero-sized tensors
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = make_tensor(shape1, device=device, dtype=dtype, low=-1, high=1)
                b2 = make_tensor(shape2, device=device, dtype=dtype, low=-1, high=1)
                ref = (
                    torch.from_numpy(
                        b1.cpu().to(dtype).numpy() @ b2.cpu().to(dtype).numpy()
                    )
                    .to(dtype)
                    .sum(0)
                )
                out_tensor = torch.zeros_like(ref, device=device)
                yield b1, b2, ref, out_tensor

        # TODO: wait for tf32 support
        dtype_list = [torch.float32]
        # old_allow_tf32 = torch.backends.mlu.matmul.allow_tf32
        # torch.backends.mlu.matmul.allow_tf32 = False
        for dtype in dtype_list:
            for b1, b2, ref, out_tensor in generate_tensor(dtype=dtype, device="mlu"):
                self._test_addbmm("addbmm", b1, b2, ref, out_tensor)

        # # TF32 test case, we need to adjust the precision threshold.
        # if torch.mlu.get_device_properties(torch.mlu.current_device()).major >= 5:
        #     torch.backends.mlu.matmul.allow_tf32 = True
        #     old_precision = self.precision
        #     self.precision = 0.05
        #     for dtype in dtype_list:
        #         for b1, b2, ref, out_tensor in generate_tensor(dtype=dtype, device='mlu'):
        #             self._test_addbmm("addbmm", b1, b2, ref, out_tensor)
        #     self.precision = old_precision

        # torch.backends.mlu.matmul.allow_tf32 = old_allow_tf32

    # @unittest.skip("not test")
    @testinfo()
    def test_addbmm_exception(self):
        M = torch.randn(10, 25).to("mlu")
        batch1 = torch.randn(3, 10, 50).to("mlu")
        batch2 = torch.randn(3, 30, 25).to("mlu")
        ref_msg = "Incompatible matrix sizes for bmm \(10x50 and 30x25\)"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addbmm(M, batch1, batch2)

        batch1 = torch.randn(2, 10, 50).to("mlu")
        ref_msg = "batch1 and batch2 must have same number of batches, got 2 and 3"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addbmm(M, batch1, batch2)

        batch1 = torch.randn(3, 10, 50).to("mlu")
        batch2 = torch.randn(3, 50, 24).to("mlu")
        ref_msg = "The expanded size of the tensor \(24\) must match the existing size \(25\) "
        ref_msg += "at non-singleton dimension 1.  Target sizes: \[10, 24\].  Tensor sizes: \[10, 25\]"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addbmm(M, batch1, batch2)

    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_addbmm_bfloat16(self):
        mat = torch.randn((10, 25), dtype=torch.bfloat16).float()
        m1 = torch.randn((3, 10, 50), dtype=torch.bfloat16).float()
        m2 = torch.randn((3, 50, 25), dtype=torch.bfloat16).float()
        mat_cpu = torch.nn.Parameter(mat)
        a_cpu = torch.nn.Parameter(m1)
        b_cpu = torch.nn.Parameter(m2)
        mat_mlu = torch.nn.Parameter(mat.mlu().bfloat16())
        a_mlu = torch.nn.Parameter(m1.mlu().bfloat16())
        b_mlu = torch.nn.Parameter(m2.mlu().bfloat16())
        out_cpu = torch.addbmm(mat_cpu, a_cpu, b_cpu)
        out_mlu = torch.addbmm(mat_mlu, a_mlu, b_mlu)
        # TODO(CNNLCORE-14101): backward not support bfloat16
        # grad = torch.randn_like(out_cpu)
        # grad_mlu = grad.mlu().bfloat16()
        # out_cpu.backward(grad)
        # out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
        # self.assertTensorsEqual(mat_mlu.grad, mat_mlu.grad.cpu().float(), 0.003, use_MSE=True)
        # self.assertTensorsEqual(a_cpu.grad, a_mlu.grad.cpu().float(), 0.003, use_MSE=True)
        # self.assertTensorsEqual(b_cpu.grad, b_mlu.grad.cpu().float(), 0.003, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
