import sys
import os
import unittest
import logging
import copy
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestAddrOp(TestCase):
    def run_addr_test(self, m, v1, v2, beta=1, alpha=1, m_transform=lambda x: x):
        m = m_transform(m.clone())
        out_cpu = torch.addr(
            m.cpu().float(), v1.cpu().float(), v2.cpu().float(), beta=beta, alpha=alpha
        )
        out_mlu = torch.addr(m, v1, v2, beta=beta, alpha=alpha)
        # addr is a splicing operator, there may be accuracy problems,
        # so reduce the accuracy standard
        self.assertTensorsEqual(out_mlu.float().cpu(), out_cpu, 0.03, use_MSE=True)

    def run_addr_out_test(
        self, m, v1, v2, dtype, m_transform=lambda x: x, beta=1, alpha=1
    ):
        m = m_transform(m.clone())
        zero_mlu = torch.zeros(v1.size(0), v2.size(0), dtype=dtype, device="mlu")
        zero_cpu = torch.zeros(v1.size(0), v2.size(0), dtype=torch.float, device="cpu")
        out_cpu = torch.addr(
            m.cpu().float(),
            v1.cpu().float(),
            v2.cpu().float(),
            beta=beta,
            alpha=alpha,
            out=zero_cpu,
        )
        out_mlu = torch.addr(m, v1, v2, beta=beta, alpha=alpha, out=zero_mlu)
        self.assertTensorsEqual(out_mlu.float().cpu(), out_cpu, 0.03, use_MSE=True)
        self.assertTensorsEqual(zero_mlu.float().cpu(), zero_cpu, 0.03, use_MSE=True)

    def run_addr__test(self, m, v1, v2, m_transform=lambda x: x):
        m = m_transform(m.clone())
        m_cpu = m.cpu().float()
        m_cpu.addr_(v1.cpu().float(), v2.cpu().float(), beta=1, alpha=1)
        ptr1 = m.data_ptr()
        m.addr_(v1, v2, beta=1, alpha=1)
        ptr2 = m.data_ptr()

        # addr is a splicing operator, there may be accuracy problems,
        # so reduce the accuracy standard
        self.assertEqual(ptr1, ptr2)
        self.assertTensorsEqual(m_cpu.float(), m.float().cpu(), 0.03, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addr_contiguous(self, device="mlu"):
        for h, w in [(100, 110), (1, 20), (200, 2)]:
            for beta, alpha in [(0.5, 1), (1, 1), (1, 0.5), (0, 2), (2, 0)]:
                for dtype in [torch.float32, torch.float16]:
                    m = torch.randn(h, w).to(dtype=dtype).to(device)
                    v1 = torch.randn(h).to(dtype=dtype).to(device)
                    v2 = torch.randn(w).to(dtype=dtype).to(device)
                    self.run_addr_test(m, v1, v2)

                    # test broadcast
                    m = torch.randn(1, w).to(dtype=dtype).to(device)
                    self.run_addr_test(m, v1, v2)

                    # test transpose
                    self.run_addr_test(
                        m, v2, v1, beta, alpha, lambda x: x.transpose(0, 1)
                    )

                    # test 0 strided
                    v1 = torch.randn(1).expand(h).to(dtype=dtype).to(device)
                    self.run_addr_test(m, v1, v2, beta, alpha)
                    self.run_addr_test(
                        m, v2, v1, beta, alpha, lambda x: x.transpose(0, 1)
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_addr_not_dense(self, device="mlu"):
        for h, w in [(100, 110), (1, 20), (200, 2)]:
            for beta, alpha in [(0.5, 1), (1, 1), (1, 0.5)]:
                for dtype in [torch.float32, torch.float16, torch.double, torch.half]:
                    m = torch.randn(h, w).to(dtype=dtype)
                    M = m.to(device)[:, : int(w / 2)]
                    m = m[:, : int(w / 2)]
                    v1_cpu = torch.randn(h * 2).to(dtype=dtype)
                    v1_mlu = v1_cpu.to(device)[:h]
                    v1_cpu = v1_cpu[:h]
                    v2_cpu = torch.randn(w).to(dtype=dtype)
                    v2_mlu = v2_cpu.to(device)[: int(w / 2)]
                    v2_cpu = v2_cpu[: int(w / 2)]
                    out_cpu = torch.addr(
                        m.float(),
                        v1_cpu.float(),
                        v2_cpu.float(),
                        beta=beta,
                        alpha=alpha,
                    )
                    out_mlu = torch.addr(M, v1_mlu, v2_mlu, beta=beta, alpha=alpha)
                    self.assertTensorsEqual(
                        out_mlu.float().cpu(), out_cpu, 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_addr_out(self, device="mlu"):
        for h, w in [(100, 110), (1, 20), (200, 2), (0, 2), (2, 0)]:
            for beta, alpha in [(0.5, 1), (1, 1), (1, 0.5), (0, 2), (2, 0)]:
                for dtype in [torch.float32, torch.float16]:
                    m = torch.randn(h, w).to(dtype=dtype).to(device)
                    v1 = torch.randn(h).to(dtype=dtype).to(device)
                    v2 = torch.randn(w).to(dtype=dtype).to(device)
                    self.run_addr_out_test(m, v1, v2, dtype, beta=beta, alpha=alpha)

                    # test transpose
                    self.run_addr_out_test(
                        m, v2, v1, dtype, lambda x: x.transpose(0, 1)
                    )

                    # test 0 strided
                    v1 = torch.randn(1).expand(h).to(dtype=dtype).to(device)
                    self.run_addr_out_test(m, v1, v2, dtype)
                    self.run_addr_out_test(
                        m, v2, v1, dtype, lambda x: x.transpose(0, 1)
                    )

                    # test inplace
                    self.run_addr__test(m, v1, v2)

    # @unittest.skip("not test")
    @testinfo()
    def test_addr_backward(self):
        def fn(m_0, v1_0, v2_0, dtype, is_grad_conti=True):
            m = m_0.to("mlu")
            v1 = v1_0.to("mlu")
            v2 = v2_0.to("mlu")
            out_cpu = torch.addr(
                m_0.float(), v1_0.float(), v2_0.float(), beta=1, alpha=1
            )
            out_mlu = torch.addr(m, v1, v2, beta=1, alpha=1)
            grad = torch.randn(out_cpu.shape)
            grad_mlu = grad.to("mlu")
            if is_grad_conti is False:
                grad = self.to_non_dense(grad, dim=None, distance=2)
            out_cpu.backward(grad)

            out_grad_cpu_m = copy.deepcopy(m_0.grad)
            out_grad_cpu_v1 = copy.deepcopy(v1_0.grad)
            out_grad_cpu_v2 = copy.deepcopy(v2_0.grad)
            m_0.grad.zero_()
            v1_0.grad.zero_()
            v2_0.grad.zero_()

            if is_grad_conti is False:
                grad_mlu = self.to_non_dense(grad_mlu, dim=None, distance=2)
            out_mlu.backward(grad_mlu)

            out_grad_mlu_m = copy.deepcopy(m_0.grad)
            out_grad_mlu_v1 = copy.deepcopy(v1_0.grad)
            out_grad_mlu_v2 = copy.deepcopy(v2_0.grad)
            m_0.grad.zero_()
            v1_0.grad.zero_()
            v2_0.grad.zero_()
            er = 0.003
            if dtype == torch.float16:
                er = 0.3
            self.assertTensorsEqual(
                out_grad_cpu_m.float(), out_grad_mlu_m.float().cpu(), er, use_MSE=True
            )
            self.assertTensorsEqual(
                out_grad_cpu_v1.float(), out_grad_mlu_v1.float().cpu(), er, use_MSE=True
            )
            self.assertTensorsEqual(
                out_grad_cpu_v2.float(), out_grad_mlu_v2.float().cpu(), er, use_MSE=True
            )

        for h, w in [(100, 110), (1, 20), (200, 2)]:
            for dtype in [torch.float32, torch.float16, torch.double, torch.half]:
                m_0 = torch.randn(h, w, dtype=dtype, requires_grad=True)
                v1_0 = torch.randn(h, dtype=dtype, requires_grad=True)
                v2_0 = torch.randn(w, dtype=dtype, requires_grad=True)
                fn(m_0, v1_0, v2_0, dtype)
                fn(m_0, v1_0, v2_0, dtype, is_grad_conti=False)

    # @unittest.skip("not test")
    @testinfo()
    def test_addr_empty(self):
        m = torch.randn(1, 0)
        a = torch.randn(1)
        b = torch.randn(0)
        m_mlu = m.mlu()
        a_mlu = a.mlu()
        b_mlu = b.mlu()
        out = torch.addr(m, a, b)
        out_mlu = torch.addr(m_mlu, a_mlu, b_mlu)
        self.assertTensorsEqual(out.float(), out_mlu.float().cpu(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addr_invalid_shape(self):
        m = torch.randn(2, 3).to("mlu")
        a = torch.randn(2, 3).to("mlu")
        b = torch.randn(2, 3).to("mlu")
        ref_msg = "addr: Expected 1-D argument vec1, but got 2-D"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addr(m, a, b)

    # @unittest.skip("not test")
    @testinfo()
    def test_addr_invalid_dtype(self):
        m = torch.arange(2).int().to("mlu")
        a = torch.arange(2).int().to("mlu")
        b = torch.arange(2).int().to("mlu")
        ref_msg = f"MLU addr don't support tensor dtype Int."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addr(m, a, b)

    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_addr_bfloat16(self):
        mat = torch.randn((44, 32)).bfloat16().float()
        a = torch.randn((44,)).bfloat16().float()
        b = torch.randn((32,)).bfloat16().float()
        mat_cpu = torch.nn.Parameter(mat)
        a_cpu = torch.nn.Parameter(a)
        b_cpu = torch.nn.Parameter(b)
        mat_mlu = torch.nn.Parameter(mat.mlu().bfloat16())
        a_mlu = torch.nn.Parameter(a.mlu().bfloat16())
        b_mlu = torch.nn.Parameter(b.mlu().bfloat16())
        out_cpu = torch.addr(mat_cpu, a_cpu, b_cpu)
        out_mlu = torch.addr(mat_mlu, a_mlu, b_mlu)
        grad = torch.randn(out_cpu.shape)
        grad_mlu = grad.mlu().bfloat16()
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
        self.assertTensorsEqual(
            mat_cpu.grad.float(), mat_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            a_cpu.grad.float(), a_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            b_cpu.grad.float(), b_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
