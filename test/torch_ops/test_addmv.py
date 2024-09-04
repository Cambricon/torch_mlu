from __future__ import print_function

import sys
import os
from itertools import product
import unittest
import logging
import random
import itertools
import math
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase, read_card_info  # pylint: disable=C0413

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)
torch.manual_seed(1234)


class TestAddmvOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_linear_algebra_scalar_raises(self):
        device = "mlu"
        m = torch.randn(5, 5, device=device)
        v = torch.randn(5, device=device)
        s = torch.tensor(7, device=device)
        self.assertRaises(RuntimeError, lambda: torch.mv(m, s))
        self.assertRaises(RuntimeError, lambda: torch.addmv(v, m, s))

    # @unittest.skip("not test")
    @testinfo()
    def test_addmv_empty(self):
        def fn(torchfn, *args, test_out=False, **kwargs):
            def call_torch_fn(*args, **kwargs):
                return torchfn(
                    *tuple(
                        torch.randn(shape, device=device)
                        if isinstance(shape, tuple)
                        else shape
                        for shape in args
                    ),
                    **kwargs,
                )

            result = call_torch_fn(*args, **kwargs)
            if not test_out:
                return result
            else:
                out = torch.full_like(result, math.nan)
                out1 = call_torch_fn(*args, **kwargs, out=out)  # pylint: disable=W0612
                return out

        # mv, addmv
        device = "mlu"
        self.assertEqual((0,), fn(torch.mv, (0, 0), (0,)).shape)
        self.assertEqual((0,), fn(torch.mv, (0, 2), (2,)).shape)
        self.assertEqual(torch.zeros((3,), device=device), fn(torch.mv, (3, 0), (0,)))
        self.assertEqual(
            torch.zeros((3,), device=device), fn(torch.mv, (3, 0), (0,), test_out=True)
        )
        self.assertEqual((0,), fn(torch.addmv, (0,), (0, 0), (0,)).shape)
        t = torch.randn((3,), device=device)
        self.assertEqual(t, fn(torch.addmv, t, (3, 0), (0,)))
        self.assertEqual(t, fn(torch.addmv, t, (3, 0), (0,), test_out=True))

    def _test_addmm_addmv(
        self, f, t, m, v, *, alpha=None, beta=None, transpose_out=False, err=3e-3
    ):
        dtype = t.dtype
        numpy_dtype = dtype
        if dtype in {torch.bfloat16}:
            numpy_dtype = torch.float
        if dtype.is_complex:
            alpha = 0.9 + 0.3j if alpha is None else alpha
            beta = 0.5 + 0.6j if beta is None else beta
        else:
            alpha = 1.2 if alpha is None else alpha
            beta = 0.8 if beta is None else beta
        res1 = f(t, m, v, alpha=alpha, beta=beta)
        res2 = torch.full_like(res1, math.nan)
        if transpose_out:
            res2 = res2.t().clone(memory_format=torch.contiguous_format).t()
        f(t, m, v, alpha=alpha, beta=beta, out=res2)
        res3 = alpha * (
            m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy()
        )
        if beta != 0:
            res3 += (beta * t).to(numpy_dtype).cpu().numpy()
        res3 = torch.from_numpy(res3).to(dtype)
        self.assertTensorsEqual(
            res1.float().cpu(), res2.float().cpu(), err, use_MSE=True
        )
        self.assertTensorsEqual(
            res1.float().cpu(), res3.float().cpu(), err, use_MSE=True
        )

    # TODO(daitian): failed on CI test.
    @unittest.skip("not test")
    @testinfo()
    def test_addmv_ori(self):
        device = "mlu"
        dtype_list = [(torch.half, 3e-2), (torch.float, 3e-3)]
        for dtype, err in dtype_list:
            ts = [
                torch.randn(10, device=device).to(dtype),
                torch.randn(1, device=device).to(dtype).expand(10),
            ]
            vs = [
                torch.randn(100, device=device).to(dtype),
                torch.ones(1, device=device).to(dtype).expand(100),
            ]
            ms = [
                # 0d
                torch.ones((), device=device).to(dtype).expand(10, 100),
                # 1d
                torch.randn((1, 100), device=device).to(dtype).expand(10, 100),
                # this initialization reduces errors for low precision for broadcasted matrices
                # by making sure that intermediate and result values are exactly representable
                # in low precision type
                torch.randint(3, (10, 1), dtype=torch.float)
                .to(device)
                .to(dtype)
                .expand(10, 100),
                # 2d
                torch.randn((10, 100), device=device).to(dtype),
                torch.randn((100, 10), device=device).to(dtype).t(),
            ]
            for m, v, t in itertools.product(ms, vs, ts):
                self._test_addmm_addmv(torch.addmv, t, m, v, err=err)
            # Test beta=0, t=nan
            t = torch.full((10,), math.nan, device=device).to(dtype)
            for m, v in itertools.product(ms, vs):
                self._test_addmm_addmv(torch.addmv, t, m, v, beta=0, err=err)

            # test addmv row major clomajor incx incy lda
            o = 5
            s = 3
            a_data = torch.arange(1, o * s + 1, dtype=dtype).to(device).view(o, s)
            x_data = torch.arange(1, s + 1, 1, dtype=dtype).to(device)
            y_data = torch.ones(o, device=device, dtype=dtype)
            control = torch.tensor(
                [15.0, 33.0, 51.0, 69.0, 87.0],  # pylint: disable=W0612
                device=device,
                dtype=dtype,
            )

            def _test(row_major, incx, incy, lda_tail):
                if row_major:
                    a_storage = torch.full(
                        (o, s + lda_tail), float("nan"), device=device, dtype=dtype
                    )
                else:
                    a_storage = torch.full(
                        (s, o + lda_tail), float("nan"), device=device, dtype=dtype
                    ).permute(1, 0)
                a = a_storage[:o, :s].copy_(a_data)

                x_storage = torch.full(
                    (s, incx), float("nan"), device=device, dtype=dtype
                )
                x = x_storage[:, 0].copy_(x_data)

                y_storage = torch.full(
                    (o, incy), float("nan"), device=device, dtype=dtype
                )
                y = y_storage[:, 0].copy_(y_data)

                self._test_addmm_addmv(torch.addmv, y, a, x)

            for row_major, incx, incy, lda_tail in itertools.product(
                (False, True), (1, 2), (1, 2), (0, 1)
            ):
                _test(row_major, incx, incy, lda_tail)

    # @unittest.skip("not test")
    @testinfo()
    def test_addmv_alpha_beta_empty(self):
        device = "mlu"
        dtype_list = [(torch.half, 3e-2), (torch.float, 3e-3)]
        for dtype, _ in dtype_list:
            value = 11
            input = torch.full((2,), value, dtype=dtype, device=device)
            mat = torch.ones((2, 0), dtype=dtype, device=device)
            vec = torch.ones((0,), dtype=dtype, device=device)
            out = torch.empty((2,), dtype=dtype, device=device)

            alpha = 6
            beta = 3

            self.assertEqual(
                torch.full((2,), beta * value, dtype=dtype, device=device),
                torch.addmv(input=input, mat=mat, vec=vec, alpha=alpha, beta=beta),
            )
            self.assertEqual(
                torch.full((2,), beta * value, dtype=dtype, device=device),
                torch.addmv(
                    input=input, mat=mat, vec=vec, alpha=alpha, beta=beta, out=out
                ),
            )

    def _select_broadcastable_dims(self, dims_full=None):  # pylint: disable=R0201
        # select full dimensionality
        if dims_full is None:
            dims_full = []
            ndims = random.randint(1, 4)
            dims_full = [random.randint(1, 8) for _ in range(ndims)]
        else:
            ndims = len(dims_full)

        # select actual dimensions for ops:
        # larger: full ndims, individual sizes may be reduced
        # smaller: possibly reduced ndims, sizes may be reduced
        smaller_ndims = random.randint(1, ndims)
        dims_small = []
        dims_large = []
        for i in range(ndims - 1, -1, -1):
            j = random.randint(1, 3)
            if j == 1:  # no reduced singleton dimension
                ds = dims_full[i]
                dl = dims_full[i]
            elif j == 2:  # larger may have reduced singleton dimension
                ds = dims_full[i]
                dl = 1 if len(dims_small) < smaller_ndims else dims_full[i]
            elif j == 3:  # smaller may have reduced singleton dimension
                ds = 1
                dl = dims_full[i]
            dims_large = [dl] + dims_large
            if len(dims_small) < smaller_ndims:
                dims_small = [ds] + dims_small
        return (dims_small, dims_large, dims_full)

    # @unittest.skip("not test")
    @testinfo()
    def test_addmv_broadcast_fused(self):
        device = "mlu"
        n_dim = random.randint(1, 8)
        m_dim = random.randint(1, 8)
        (t0_dims_full, t1_dims, t2_dims) = ([n_dim], [n_dim, m_dim], [m_dim])
        (t0_dims_small, _, _) = self._select_broadcastable_dims(t0_dims_full)
        t0_small = torch.randn(*t0_dims_small, device=device).float()
        t1 = torch.randn(*t1_dims, device=device).float()
        t2 = torch.randn(*t2_dims, device=device).float()

        t0_full = t0_small.expand(*t0_dims_full).to(device)

        r0 = torch.addmv(t0_small, t1, t2)
        r1 = torch.addmv(t0_full, t1, t2)
        self.assertEqual(r0, r1)

    # @unittest.skip("not test")
    @testinfo()
    def test_addmv(self):
        dtype_list = [(torch.half, 3e-2), (torch.float, 3e-3)]
        M_shape = [10]
        mat_shape = [10, 20]
        vec_shape = [20]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype_err, func in product(dtype_list, func_list):
            dtype, err = dtype_err
            M_cpu = torch.randn(M_shape).to(dtype)
            mat_cpu = torch.randn(mat_shape).to(dtype)
            vec_cpu = torch.randn(vec_shape).to(dtype)
            M_mlu = M_cpu.to("mlu")
            mat_mlu = mat_cpu.to("mlu")
            vec_mlu = vec_cpu.to("mlu")
            out_cpu = torch.addmv(
                func(M_cpu).float(), func(mat_cpu).float(), func(vec_cpu).float()
            )
            out_mlu = torch.addmv(func(M_mlu), func(mat_mlu), func(vec_mlu))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )

        # test sliced output
        M_cpu = torch.randn(4)
        mat_cpu = torch.randn(4, 6)
        vec_cpu = torch.randn(6)
        out_cpu = torch.randn(10)
        M_mlu = M_cpu.to("mlu")
        mat_mlu = mat_cpu.to("mlu")
        vec_mlu = vec_cpu.to("mlu")
        out_mlu = out_cpu.to("mlu")
        torch.addmv(M_cpu, mat_cpu, vec_cpu, out=out_cpu[:4])
        torch.addmv(M_mlu, mat_mlu, vec_mlu, out=out_mlu[:4])
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
        )

        # test resize output
        out_cpu = torch.randn(10, 10)
        out_mlu = out_cpu.mlu()
        torch.addmv(M_cpu, mat_cpu, vec_cpu, out=out_cpu)
        torch.addmv(M_mlu, mat_mlu, vec_mlu, out=out_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_addmv_invalid_dtype(self):
        mat = torch.randn(2, 3).to(torch.int).mlu()
        vec = torch.randn(3).to(torch.int).mlu()
        ref_msg = f"\"MLU mm\" not implemented for 'Int'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.mv(mat, vec)
        M = torch.randn(2).to(torch.int).mlu()
        mat = torch.randn(2, 3).to(torch.int).mlu()
        vec = torch.randn(3).to(torch.int).mlu()
        ref_msg = f"\"MLU mm\" not implemented for 'Int'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.addmv(M, mat, vec)

    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_addmv_bfloat16(self):
        mat = torch.randn((3,), dtype=torch.bfloat16).float()
        a = torch.randn((3, 5), dtype=torch.bfloat16).float()
        b = torch.randn((5,), dtype=torch.bfloat16).float()
        mat_cpu = torch.nn.Parameter(mat)
        a_cpu = torch.nn.Parameter(a)
        b_cpu = torch.nn.Parameter(b)
        mat_mlu = torch.nn.Parameter(mat.mlu().bfloat16())
        a_mlu = torch.nn.Parameter(a.mlu().bfloat16())
        b_mlu = torch.nn.Parameter(b.mlu().bfloat16())
        out_cpu = torch.addmv(mat_cpu, a_cpu, b_cpu)
        out_mlu = torch.addmv(mat_mlu, a_mlu, b_mlu)
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
