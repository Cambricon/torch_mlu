from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
from numbers import Number

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
)  # pylint: disable=C0413,C0411

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
    (torch.float, 0.003),
]


class TestSparseMMOp(TestCase):
    def _gen_sparse(self, sparse_dim, nnz, with_size, dtype, device, coalesced):
        if isinstance(with_size, Number):
            with_size = [with_size] * sparse_dim

        x, i, v = self.genSparseTensor(
            with_size, sparse_dim, nnz, not coalesced, dtype=dtype, device=device
        )

        if not coalesced:
            self.assert_uncoalesced(x)

        return x, i, v

    # @unittest.skip("not test")
    @testinfo()
    def test_sparse_mm(self):
        mem_func = [lambda x: x, self.to_non_dense]
        for (dt, err), (shape_a, shape_b) in product(dtype_err, shape_gen(ns, ms, ps)):
            for mem_func1, mem_func2 in product(mem_func, mem_func):
                nnz = int(shape_a[0] * shape_a[1] * 0.2)
                x1, i, v = self._gen_sparse(2, nnz, shape_a, torch.float, "cpu", True)
                x1 = x1.to(dt).float()
                x2 = torch.rand(shape_b).to(dt).float()
                x1_mlu = torch.sparse_coo_tensor(
                    mem_func1(i),
                    mem_func1(v.to(dt)),
                    size=shape_a,
                    dtype=dt,
                    device="mlu",
                )
                x2_mlu = mem_func2(x2.to("mlu").to(dt))
                y_cpu = torch.sparse.mm(x1, x2)
                y_mlu = torch.sparse.mm(x1_mlu, x2_mlu)
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sparse_mm_out(self):
        mem_func = [lambda x: x, self.to_non_dense]
        for (dt, err), (shape_a, shape_b) in product(dtype_err, shape_gen(ns, ms, ps)):
            for mem_func1, mem_func2, mem_func3 in product(
                mem_func, mem_func, mem_func
            ):
                nnz = int(shape_a[0] * shape_a[1] * 0.2)
                x1, i, v = self._gen_sparse(2, nnz, shape_a, torch.float, "cpu", True)
                x1 = x1.to(dt).float()
                x2 = torch.rand(shape_b).to(dt).float()
                y_cpu = torch.zeros(15, 7)
                x1_mlu = torch.sparse_coo_tensor(
                    mem_func1(i),
                    mem_func1(v.to(dt)),
                    size=shape_a,
                    dtype=dt,
                    device="mlu",
                )
                x2_mlu = mem_func2(x2.to("mlu").to(dt))
                y_mlu = mem_func3(y_cpu.to("mlu").to(dt))
                torch.mm(x1, x2, out=y_cpu)
                torch.mm(x1_mlu, x2_mlu, out=y_mlu)
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_trans_sparse_mm(self):
        mem_func = [lambda x: x, self.to_non_dense]
        for t1, t2 in [(True, False), (False, True), (True, True)]:
            for (dt, err), (shape_a, shape_b) in product(
                dtype_err, shape_gen(ns, ms, ps, t1, t2)
            ):
                for mem_func1, mem_func2 in product(mem_func, mem_func):

                    def f(x, t):
                        return x.t() if t else x

                    nnz = int(shape_a[0] * shape_a[1] * 0.2)
                    x1, i, v = self._gen_sparse(
                        2, nnz, shape_a, torch.float, "cpu", True
                    )
                    x1 = x1.to(dt).float()
                    x2 = torch.rand(shape_b).to(dt).float()
                    x1_mlu = torch.sparse_coo_tensor(
                        mem_func1(i),
                        mem_func1(v.to(dt)),
                        size=shape_a,
                        dtype=dt,
                        device="mlu",
                    )
                    x2_mlu = mem_func2(x2.to("mlu").to(dt))
                    y_cpu = torch.mm(f(x1, t1), f(x2, t2))
                    y_mlu = torch.mm(f(x1_mlu, t1), f(x2_mlu, t2))
                    self.assertTensorsEqual(
                        y_cpu, y_mlu.cpu().float(), err, use_MSE=True
                    )


if __name__ == "__main__":
    unittest.main()
