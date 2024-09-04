from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product

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


def shape_gen(ns, ms, ps, t1=False, t2=False):
    shape_a = []
    shape_b = []
    for n, m, p in product(ns, ms, ps):
        if t1:
            shape_a.append((m, n))
        else:
            shape_a.append((n, m))
        if t2:
            shape_b.append((p, m))
        else:
            shape_b.append((m, p))
    return zip(shape_a, shape_b)


ns = [0, 4, 20, 2048]
ms = [0, 5, 45, 512]
ps = [0, 8, 64, 1999]
dtype_err = [
    (torch.half, 0.03),
    (torch.float, 0.003),
    (torch.double, 0.003),
]


class TestMMOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_mm(self):
        mem_func = [lambda x: x, self.to_non_dense]
        for (dt, err), (shape_a, shape_b) in product(dtype_err, shape_gen(ns, ms, ps)):
            for mem_func1, mem_func2 in product(mem_func, mem_func):
                x1 = torch.rand(shape_a).to(dt).float()
                x2 = torch.rand(shape_b).to(dt).float()
                x1_mlu = mem_func1(x1.to("mlu").to(dt))
                x2_mlu = mem_func2(x2.to("mlu").to(dt))
                y_cpu = torch.mm(x1, x2)
                y_mlu = torch.mm(x1_mlu, x2_mlu)
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_mm_out(self):
        mem_func = [lambda x: x, self.to_non_dense]
        for (dt, err), (shape_a, shape_b) in product(dtype_err, shape_gen(ns, ms, ps)):
            for mem_func1, mem_func2, mem_func3 in product(
                mem_func, mem_func, mem_func
            ):
                x1 = torch.rand(shape_a).to(dt).float()
                x2 = torch.rand(shape_b).to(dt).float()
                x1_mlu = mem_func1(x1.to("mlu").to(dt))
                x2_mlu = mem_func2(x2.to("mlu").to(dt))
                y_cpu = torch.zeros(15, 7)
                y_mlu = mem_func3(y_cpu.to("mlu").to(dt))
                torch.mm(x1, x2, out=y_cpu)
                torch.mm(x1_mlu, x2_mlu, out=y_mlu)
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_mm_exception(self):
        m1 = torch.randn(1, 1).half().to("mlu")
        m2 = torch.randn(1, 1).to("mlu")
        msg = f"expected scalar type Half but found Float"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.mm(m1, m2)

        m1 = torch.randn(1, 1).int().to("mlu")
        m2 = torch.randn(1, 1).int().to("mlu")
        msg = f"\"MLU mm\" not implemented for 'Int'"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.mm(m1, m2)

    # @unittest.skip("not test")
    @testinfo()
    def test_trans_mm(self):
        mem_func = [lambda x: x, self.to_non_dense]
        for t1, t2 in [(True, False), (False, True), (True, True)]:
            for (dt, err), (shape_a, shape_b) in product(
                dtype_err, shape_gen(ns, ms, ps, t1, t2)
            ):
                for mem_func1, mem_func2 in product(mem_func, mem_func):

                    def f(x, t):
                        return x.t() if t else x

                    x1 = torch.rand(shape_a).to(dt).float()
                    x2 = torch.rand(shape_b).to(dt).float()
                    x1_mlu = mem_func1(x1.to("mlu").to(dt))
                    x2_mlu = mem_func2(x2.to("mlu").to(dt))
                    y_cpu = torch.mm(f(x1, t1), f(x2, t2))
                    y_mlu = torch.mm(f(x1_mlu, t1), f(x2_mlu, t2))
                    self.assertTensorsEqual(
                        y_cpu, y_mlu.cpu().float(), err, use_MSE=True
                    )

    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_mm_bfloat16(self):
        # CPU side accumulate matmul using bfloat16, but MLU side and GPU side
        # is using float.
        a = torch.randn((4, 5), dtype=torch.bfloat16).float()
        b = torch.randn((5, 6), dtype=torch.bfloat16).float()
        a_cpu = torch.nn.Parameter(a)
        b_cpu = torch.nn.Parameter(b)
        a_mlu = torch.nn.Parameter(a.mlu().bfloat16())
        b_mlu = torch.nn.Parameter(b.mlu().bfloat16())
        out_cpu = torch.mm(a_cpu, b_cpu)
        out_mlu = torch.mm(a_mlu, b_mlu)
        grad = torch.randn(out_cpu.shape)
        grad_mlu = grad.mlu().bfloat16()
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            a_cpu.grad.float(), a_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            b_cpu.grad.float(), b_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
