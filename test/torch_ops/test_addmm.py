from __future__ import print_function
import logging
import unittest
import sys
import os
from itertools import product

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase, TEST_BFLOAT16  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)


class TestAddmmOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_addmm_contiguous(self):
        dtype_list = [torch.float, torch.half]
        shape_list = [
            ((10, 25), (10, 50), (50, 25)),
            ((1, 234), (123, 648), (648, 234)),
            ((123, 1), (123, 648), (648, 234)),
            ((0, 50), (0, 20), (20, 50)),
            ((20), (10, 33), (33, 20)),
            ((1, 1), (128, 1024), (1024, 379)),
            ((), (64, 128), (128, 256)),
        ]
        for dtype in dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn(m1_shape, dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, dtype)
                m1_mlu = self.to_mlu_dtype(m1, dtype)
                m2_mlu = self.to_mlu_dtype(m2, dtype)
                res_mlu = torch.addmm(M_mlu, m1_mlu, m2_mlu)
                res_cpu = torch.addmm(M, m1, m2)
                self.assertTensorsEqual(
                    res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True
                )

        # Test 0-strided
        for dtype in dtype_list:
            M = torch.randn((10, 1), dtype=torch.float).expand(10, 25)
            m1 = torch.randn((10, 1), dtype=torch.float).expand(10, 50)
            m2 = torch.randn((50, 25), dtype=torch.float)
            M_mlu = self.to_mlu_dtype(M, dtype)
            m1_mlu = self.to_mlu_dtype(m1, dtype)
            m2_mlu = self.to_mlu_dtype(m2, dtype)
            res_cpu = torch.addmm(M, m1, m2)
            res_mlu = torch.addmm(M_mlu, m1_mlu, m2_mlu)
            self.assertTensorsEqual(res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True)

        # Test alpha and beta not equal 1
        beta_list = [-0.5, 1.7, 1, 22]
        alpha_list = [-1.7, 0.4, 1, 33]
        for dtype in dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                for beta in beta_list:
                    for alpha in alpha_list:
                        M = torch.randn(M_shape, dtype=torch.float)
                        m1 = torch.randn(m1_shape, dtype=torch.float)
                        m2 = torch.randn(m2_shape, dtype=torch.float)
                        M_mlu = self.to_mlu_dtype(M, dtype)
                        m1_mlu = self.to_mlu_dtype(m1, dtype)
                        m2_mlu = self.to_mlu_dtype(m2, dtype)
                        res_cpu = torch.addmm(
                            input=M, beta=beta, mat1=m1, mat2=m2, alpha=alpha
                        )
                        res_mlu = torch.addmm(
                            input=M_mlu,
                            beta=beta,
                            mat1=m1_mlu,
                            mat2=m2_mlu,
                            alpha=alpha,
                        )
                        self.assertTensorsEqual(
                            res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True
                        )
        # for case [b] + [a, 0] x [0, b],
        # CPU result's shape is [a, b], GPU and MLU result's shape is [b]
        # ((13), (4, 0), (0, 13)),
        M_shape = (13,)
        m1_shape = (4, 0)
        m2_shape = (0, 13)
        M = torch.randn(M_shape, dtype=torch.float)
        m1 = torch.randn(m1_shape, dtype=torch.float)
        m2 = torch.randn(m2_shape, dtype=torch.float)
        res_cpu = torch.addmm(input=M, beta=beta, mat1=m1, mat2=m2, alpha=alpha)
        res_mlu = torch.addmm(
            input=M.mlu(), beta=beta, mat1=m1.mlu(), mat2=m2.mlu(), alpha=alpha
        )
        self.assertTensorsEqual(res_cpu[1], res_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addmm_not_dense(self):
        dtype_list = [torch.float, torch.half]
        shape_list = [
            ((10, 25), (10, 50), (50, 25)),
            ((1, 234), (123, 648), (648, 234)),
            ((0, 50), (0, 20), (20, 50)),
        ]
        for dtype in dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn((m1_shape[0], 2 * m1_shape[-1]), dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, dtype)[:, : int(M_shape[-1] / 2)]
                m1_mlu = self.to_mlu_dtype(m1, dtype)[:, : m1_shape[-1]]
                m2_mlu = self.to_mlu_dtype(m2, dtype)[:, : int(m2_shape[-1] / 2)]
                M = M[:, : int(M_shape[-1] / 2)]
                m1 = m1[:, : m1_shape[-1]]
                m2 = m2[:, : int(m2_shape[-1] / 2)]
                res_mlu = torch.addmm(M_mlu, m1_mlu, m2_mlu)
                res_cpu = torch.addmm(M, m1, m2)
                self.assertTensorsEqual(
                    res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_addmm_inplace_contiguous(self):
        dtype_list = [torch.float, torch.half]
        shape_list = [
            ((10, 25), (10, 50), (50, 25)),
            ((2, 6), (2, 3), (3, 6)),
            ((22, 58), (22, 45), (45, 58)),
            ((0, 50), (0, 20), (20, 50)),
        ]
        for dtype in dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn(m1_shape, dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, dtype)
                m1_mlu = self.to_mlu_dtype(m1, dtype)
                m2_mlu = self.to_mlu_dtype(m2, dtype)
                M_mlu.addmm_(m1_mlu, m2_mlu)
                M.addmm_(m1, m2)
                self.assertTensorsEqual(M, M_mlu.cpu().float(), 0.003, use_MSE=True)

        # Test 0-strided
        for dtype in dtype_list:
            M = torch.randn((10, 1), dtype=torch.float).expand(10, 25).clone()
            m1 = torch.randn((10, 1), dtype=torch.float).expand(10, 50).clone()
            m2 = torch.randn((50, 25), dtype=torch.float)
            M_mlu = self.to_mlu_dtype(M, dtype)
            m1_mlu = self.to_mlu_dtype(m1, dtype)
            m2_mlu = self.to_mlu_dtype(m2, dtype)
            M.addmm_(m1, m2)
            M_mlu.addmm_(m1_mlu, m2_mlu)
            self.assertTensorsEqual(M, M_mlu.cpu().float(), 0.003, use_MSE=True)

        # Test alpha and beta not equal 1
        beta_list = [-0.5, 1.7, 1, 22]
        alpha_list = [-1.7, 0.4, 1, 33]
        for dtype in dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                for beta in beta_list:
                    for alpha in alpha_list:
                        M = torch.randn(M_shape, dtype=torch.float)
                        m1 = torch.randn(m1_shape, dtype=torch.float)
                        m2 = torch.randn(m2_shape, dtype=torch.float)
                        M_mlu = self.to_mlu_dtype(M, dtype)
                        m1_mlu = self.to_mlu_dtype(m1, dtype)
                        m2_mlu = self.to_mlu_dtype(m2, dtype)
                        M.addmm_(mat1=m1, mat2=m2, beta=beta, alpha=alpha)
                        M_mlu.addmm_(mat1=m1_mlu, mat2=m2_mlu, beta=beta, alpha=alpha)
                        self.assertTensorsEqual(
                            M, M_mlu.cpu().float(), 0.003, use_MSE=True
                        )

    # @unittest.skip("not test")
    @testinfo()
    def test_addmm_inplace_not_dense(self):
        dtype_list = [torch.float, torch.half]
        shape_list = [
            ((10, 25), (10, 50), (50, 25)),
            ((22, 58), (22, 45), (45, 58)),
            ((0, 50), (0, 20), (20, 50)),
        ]
        for dtype in dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn((m1_shape[0], 2 * m1_shape[-1]), dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, dtype)[:, : int(M_shape[-1] / 2)]
                m1_mlu = self.to_mlu_dtype(m1, dtype)[:, : m1_shape[-1]]
                m2_mlu = self.to_mlu_dtype(m2, dtype)[:, : int(m2_shape[-1] / 2)]
                M = M[:, : int(M_shape[-1] / 2)]
                m1 = m1[:, : m1_shape[-1]]
                m2 = m2[:, : int(m2_shape[-1] / 2)]
                M_mlu.addmm_(m1_mlu, m2_mlu)
                M.addmm_(m1, m2)
                self.assertTensorsEqual(M, M_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addmm_out(self):
        dtype_list = [torch.float, torch.half]
        shape_list = [
            ((10, 25), (10, 50), (50, 25)),
            ((1, 234), (123, 648), (648, 234)),
            ((123, 1), (123, 648), (648, 234)),
            ((0, 50), (0, 20), (20, 50)),
            ((20,), (10, 33), (33, 20)),
            ((1, 1), (128, 1024), (1024, 379)),
            ((), (64, 128), (128, 256)),
        ]
        out_shapes = [(100, 10), (1,), (20, 20, 60, 100)]
        for dtype in dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                for out_shape in out_shapes:
                    M = torch.randn(M_shape, dtype=torch.float)
                    m1 = torch.randn(m1_shape, dtype=torch.float)
                    m2 = torch.randn(m2_shape, dtype=torch.float)
                    out_cpu = torch.randn(out_shape, dtype=torch.float)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), dtype)
                    M_mlu = self.to_mlu_dtype(M, dtype)
                    m1_mlu = self.to_mlu_dtype(m1, dtype)
                    m2_mlu = self.to_mlu_dtype(m2, dtype)
                    torch.addmm(M, m1, m2, out=out_cpu)
                    torch.addmm(M_mlu, m1_mlu, m2_mlu, out=out_mlu)
                    self.assertTensorsEqual(
                        out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                    )

        # test out equals input and out is slice
        for dtype in dtype_list:
            M = torch.randn((10, 25), dtype=torch.float)
            m1 = torch.randn((10, 50), dtype=torch.float)
            m2 = torch.randn((50, 25), dtype=torch.float)
            M_mlu = self.to_mlu_dtype(M, dtype)
            m1_mlu = self.to_mlu_dtype(m1, dtype)
            m2_mlu = self.to_mlu_dtype(m2, dtype)

            torch.addmm(M, m1, m2, out=M[0:1, 0:1])
            torch.addmm(M_mlu, m1_mlu, m2_mlu, out=M_mlu[0:1, 0:1])
            self.assertTensorsEqual(M, M_mlu.cpu().float(), 0.003, use_MSE=True)

        # Test 0-strided
        for dtype in dtype_list:
            for out_shape in out_shapes:
                M = torch.randn((10, 1), dtype=torch.float).expand(10, 25)
                m1 = torch.randn((10, 1), dtype=torch.float).expand(10, 50)
                m2 = torch.randn((50, 25), dtype=torch.float)
                out_cpu = torch.randn(out_shape, dtype=torch.float)
                out_mlu = self.to_mlu_dtype(torch.randn(out_shape), dtype)
                M_mlu = self.to_mlu_dtype(M, dtype)
                m1_mlu = self.to_mlu_dtype(m1, dtype)
                m2_mlu = self.to_mlu_dtype(m2, dtype)
                torch.addmm(M, m1, m2, out=out_cpu)
                torch.addmm(M_mlu, m1_mlu, m2_mlu, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

        beta_list = [-0.5, 1.7, 1, 22]
        alpha_list = [-1.7, 0.4, 1, 33]
        for dtype in dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                for beta, alpha, out_shape in product(
                    beta_list, alpha_list, out_shapes
                ):
                    M = torch.randn(M_shape, dtype=torch.float)
                    m1 = torch.randn(m1_shape, dtype=torch.float)
                    m2 = torch.randn(m2_shape, dtype=torch.float)
                    out_cpu = torch.randn(out_shape, dtype=torch.float)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), dtype)
                    M_mlu = self.to_mlu_dtype(M, dtype)
                    m1_mlu = self.to_mlu_dtype(m1, dtype)
                    m2_mlu = self.to_mlu_dtype(m2, dtype)
                    torch.addmm(
                        input=M, beta=beta, mat1=m1, mat2=m2, alpha=alpha, out=out_cpu
                    )
                    torch.addmm(
                        input=M_mlu,
                        beta=beta,
                        mat1=m1_mlu,
                        mat2=m2_mlu,
                        alpha=alpha,
                        out=out_mlu,
                    )
                    self.assertTensorsEqual(
                        out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_addmm_exception(self):
        input = torch.randn(1, 1).half().to("mlu")
        m1 = torch.randn(1, 1).half().to("mlu")
        m2 = torch.randn(1, 1).to("mlu")
        msg = f"self and mat2 must have the same dtype"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.addmm(input, m1, m2)

        input = torch.randn(1, 1).int().to("mlu")
        m1 = torch.randn(1, 1).int().to("mlu")
        m2 = torch.randn(1, 1).int().to("mlu")
        msg = f"not implemented for 'Int'"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.addmm(input, m1, m2)

    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_addmm_bfloat16(self):
        mat = torch.randn((10, 25), dtype=torch.bfloat16).float()
        m1 = torch.randn((10, 50), dtype=torch.bfloat16).float()
        m2 = torch.randn((50, 25), dtype=torch.bfloat16).float()
        mat_cpu = torch.nn.Parameter(mat)
        a_cpu = torch.nn.Parameter(m1)
        b_cpu = torch.nn.Parameter(m2)
        mat_mlu = torch.nn.Parameter(mat.mlu().bfloat16())
        a_mlu = torch.nn.Parameter(m1.mlu().bfloat16())
        b_mlu = torch.nn.Parameter(m2.mlu().bfloat16())
        out_cpu = torch.addmm(
            input=mat_cpu, beta=0.4, mat1=a_cpu, mat2=b_cpu, alpha=1.7
        )
        out_mlu = torch.addmm(
            input=mat_mlu, beta=0.4, mat1=a_mlu, mat2=b_mlu, alpha=1.7
        )
        # TODO(CNNLCORE-14101): backward not support bfloat16
        # grad = torch.randn_like(out_cpu)
        # grad_mlu = grad.mlu().bfloat16()
        # out_cpu.backward(grad)
        # out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
        # self.assertTensorsEqual(mat_mlu.grad, mat_mlu.grad.cpu().float(), 0.003, use_MSE=True)
        # self.assertTensorsEqual(a_cpu.grad, a_mlu.grad.cpu().float(), 0.003, use_MSE=True)
        # self.assertTensorsEqual(b_cpu.grad, b_mlu.grad.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_addmm_activation(self):
        dtype_list = [torch.float, torch.half]
        shape_list = [
            ((10, 25), (10, 50), (50, 25)),
            ((1, 234), (123, 648), (648, 234)),
            ((123, 1), (123, 648), (648, 234)),
            ((0, 50), (0, 20), (20, 50)),
            ((20), (10, 33), (33, 20)),
            ((1, 1), (128, 1024), (1024, 379)),
            ((), (64, 128), (128, 256)),
        ]
        # test relu
        for dtype in dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn(m1_shape, dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, dtype)
                m1_mlu = self.to_mlu_dtype(m1, dtype)
                m2_mlu = self.to_mlu_dtype(m2, dtype)
                res_mlu = torch._addmm_activation(M_mlu, m1_mlu, m2_mlu, use_gelu=False)
                res_cpu = torch._addmm_activation(M, m1, m2, use_gelu=False)
                self.assertTensorsEqual(
                    res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True
                )
        # test gelu
        for dtype in dtype_list:
            for M_shape, m1_shape, m2_shape in shape_list:
                M = torch.randn(M_shape, dtype=torch.float)
                m1 = torch.randn(m1_shape, dtype=torch.float)
                m2 = torch.randn(m2_shape, dtype=torch.float)
                M_mlu = self.to_mlu_dtype(M, dtype)
                m1_mlu = self.to_mlu_dtype(m1, dtype)
                m2_mlu = self.to_mlu_dtype(m2, dtype)
                res_mlu = torch._addmm_activation(M_mlu, m1_mlu, m2_mlu, use_gelu=True)
                res_cpu = torch._addmm_activation(M, m1, m2, use_gelu=True)
                self.assertTensorsEqual(
                    res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    unittest.main()
