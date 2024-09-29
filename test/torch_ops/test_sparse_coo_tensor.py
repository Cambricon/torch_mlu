from __future__ import print_function

import sys
import logging
import os
import unittest
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411
from numbers import Number

logging.basicConfig(level=logging.DEBUG)


class TestSparseCooTensor(TestCase):
    def _gen_sparse(self, sparse_dim, nnz, with_size, dtype, device, coalesced):
        if isinstance(with_size, Number):
            with_size = [with_size] * sparse_dim

        x, i, v = self.genSparseTensor(
            with_size, sparse_dim, nnz, not coalesced, dtype=dtype, device=device
        )

        if not coalesced:
            self.assert_uncoalesced(x)

        return x, i, v

    def assert_uncoalesced(self, x):
        assert not x.is_coalesced()
        existing_indices = set()
        for i in range(x._nnz()):
            index = str(x._indices()[:, i])
            if index in existing_indices:
                return True
            else:
                existing_indices.add(index)

    # @unittest.skip("not test")
    @testinfo()
    def test_basic(self):
        args_list = [
            # (sparse_dim, nnz, size)
            (1, 10, (100,)),
            (1, 100, (10, 100)),
            (2, 100, (10, 100)),
            (1, 100, (20, 20, 10)),
            (2, 100, (20, 20, 10)),
            (3, 100, (20, 20, 10)),
            (1, 200, (20, 20, 10, 10)),
            (1, 1, (20, 20, 10, 10)),
            (4, 40000, (20, 20, 10, 10)),
            (4, 400, (20, 20, 10, 5, 100)),
            (1, 0, (30, 40)),
            (2, 0, (30, 40)),
        ]
        dtype_list = [torch.float32, torch.half]
        for sparse_dim, nnz, size in args_list:
            for dtype in dtype_list:
                out_cpu, _, _ = self._gen_sparse(
                    sparse_dim, nnz, size, dtype, device="cpu", coalesced=False
                )
                out_mlu = torch.sparse_coo_tensor(
                    out_cpu._indices(),
                    out_cpu._values(),
                    size,
                    dtype=dtype,
                    device="mlu",
                    is_coalesced=False,
                )
                self.assertEqual(out_cpu, out_mlu.cpu(), 0.0)
                assert True == out_mlu.is_sparse

    # @unittest.skip("not test")
    @testinfo()
    def test_is_coalesced(self):
        args_list = [
            # (sparse_dim, nnz, size)
            (1, 10, (100,)),
            (1, 100, (10, 100)),
            (2, 100, (20, 20, 10)),
            (3, 100, (20, 20, 10)),
            (4, 40000, (20, 20, 10, 10)),
            (1, 0, (30, 40)),
        ]
        dtype_list = [torch.float32, torch.half]
        for sparse_dim, nnz, size in args_list:
            for dtype in dtype_list:
                out_cpu, _, _ = self._gen_sparse(
                    sparse_dim, nnz, size, dtype=dtype, device="cpu", coalesced=True
                )
                out_mlu = torch.sparse_coo_tensor(
                    out_cpu._indices(),
                    out_cpu._values(),
                    size,
                    dtype=dtype,
                    device="mlu",
                    is_coalesced=True,
                )

                self.assertEqual(out_cpu, out_mlu.cpu(), 0.0)
                assert True == out_mlu.is_sparse
                assert True == out_mlu.is_coalesced()

    # @unittest.skip("not test")
    @testinfo()
    def test_check_invariants(self):
        args_list = [
            # (sparse_dim, nnz, size)
            (1, 10, (100,)),
            (1, 100, (10, 100)),
            (2, 100, (20, 20, 10)),
            (3, 100, (20, 20, 10)),
            (4, 40000, (20, 20, 10, 10)),
            (1, 0, (30, 40)),
        ]
        dtype_list = [torch.float32, torch.half]
        # TODO(PYTORCH-12536): When flatten_indices is supported,
        # both is_coalesced and check_invariants can be set to True.
        coalesced = [False]
        for sparse_dim, nnz, size in args_list:
            for dtype in dtype_list:
                for is_coalesced in coalesced:
                    out_cpu, _, _ = self._gen_sparse(
                        sparse_dim,
                        nnz,
                        size,
                        dtype=dtype,
                        device="cpu",
                        coalesced=is_coalesced,
                    )
                    out_mlu = torch.sparse_coo_tensor(
                        out_cpu._indices(),
                        out_cpu._values(),
                        size,
                        dtype=dtype,
                        device="mlu",
                        is_coalesced=is_coalesced,
                        check_invariants=True,
                    )

                    self.assertEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_exception(self):
        size = [2, 3]
        indices = torch.tensor([[0, 1, 1, 1], [2, 0, 2, 2]])
        values = torch.tensor([3, 4, 5, 6])
        values_device = values.to("mlu")

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected all tensors to be on the same device, but found at least two devices, cpu and mlu:0!",
        ):
            output_tensor = torch.sparse_coo_tensor(indices, values_device, size)

        out_cpu = torch.sparse_coo_tensor(indices, values_device, size, device="cpu")
        out_mlu = torch.sparse_coo_tensor(indices, values_device, size, device="mlu")
        self.assertEqual(out_cpu, out_mlu.cpu(), 0.0)

        # TODO(PYTORCH-12536): When flatten_indices is supported,
        # both is_coalesced and check_invariants can be set to True.
        # with self.assertRaisesRegex(
        #     RuntimeError,
        #     "cannot set is_coalesced to true if indices correspond to uncoalesced COO tensor",
        # ):
        #     out_mlu = torch.sparse_coo_tensor(
        #         indices, values_device, size, device='mlu', is_coalesced=True, check_invariants=True)


if __name__ == "__main__":
    run_tests()
