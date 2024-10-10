from __future__ import print_function

import sys
import random
import os
import unittest
import logging
from itertools import product
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase


class TestCoalesceOp(TestCase):
    def compare_tensors(self, sparse_tensor_cpu, sparse_tensor_mlu, tolerance):
        self.assertTensorsEqual(
            sparse_tensor_cpu._values(),
            sparse_tensor_mlu.cpu()._values(),
            tolerance,
            use_MSE=True,
        )
        self.assertTensorsEqual(
            sparse_tensor_cpu._indices(), sparse_tensor_mlu.cpu()._indices(), 0
        )

    def make_tensor(self, size, device="cpu", dtype=torch.float32, low=-1, high=1):
        """Generate a tensor with random values between low and high."""
        return (high - low) * torch.rand(size, dtype=dtype, device=device) + low

    def generate_random_size(self, min_dim, max_dim, dim_count):
        """Generate a random size within the given dimension range."""
        return [random.randint(min_dim, max_dim) for _ in range(dim_count)]

    def generate_sparse_tensors(
        self,
        num_tensors,
        total_dim_range,
        min_dim,
        max_dim,
        nnz_range,
        is_uncoalesced,
        device,
        dtype,
    ):
        """
        Generate a specified number of random sparse tensors with random sizes and sparse dimensions.

        Parameters:
        - num_tensors: Number of tensors to generate.
        - total_dim_range: A tuple (min, max) specifying the range of total dimensions.
        - min_dim: Minimum value for the size of any dimension.
        - max_dim: Maximum value for the size of any dimension.
        - nnz_range: A tuple (min, max) specifying the range of non-zero elements.
        - is_uncoalesced: Boolean indicating whether to generate uncoalesced tensors.
        - device: Device on which to create the tensors.
        - dtype: Data type of the tensors.

        Returns:
        - List of generated sparse tensors.
        """
        tensors = []
        for _ in range(num_tensors):
            total_dim = random.randint(*total_dim_range)
            sparse_dim = random.randint(
                1, total_dim - 1
            )  # Ensure at least one dense dimension
            size = self.generate_random_size(min_dim, max_dim, total_dim)

            nnz = random.randint(*nnz_range)
            v_size = [nnz] + size[sparse_dim:]
            v = self.make_tensor(v_size, device=device, dtype=dtype, low=-100, high=100)

            # sparse_indices = random.sample(range(total_dim), sparse_dim)
            sparse_indices = list(range(sparse_dim))
            i = torch.zeros(sparse_dim, nnz, device=device, dtype=torch.long)

            for j in range(sparse_dim):
                unique_indices = torch.randint(
                    0, size[sparse_indices[j]], (nnz,), device=device
                )

                # Adjust repetitions based on the number of non-zero elements
                if is_uncoalesced:
                    # Calculate the number of repetitions
                    # Ensure that we have at least one repetition and do not exceed nnz
                    repetitions = max(1, min(nnz // 2, nnz - 1))
                    # Select random positions for the repetitions
                    repeat_indices = torch.randperm(nnz)[:repetitions]
                    # Generate a random value for the repetitions
                    repeat_value = torch.randint(
                        0, size[sparse_indices[j]], (1,), device=device
                    ).item()
                    # Set the selected positions to the repeat value
                    unique_indices[repeat_indices] = repeat_value

                i[j] = unique_indices

            x = torch.sparse_coo_tensor(
                i, v, torch.Size(size), dtype=dtype, device=device
            )
            if not is_uncoalesced:
                x = x.coalesce()  # Coalesce to remove duplicate indices if needed
            tensors.append(x)

        return tensors

    # @unittest.skip("not test")
    @testinfo()
    def test_Coalesce(self):
        num_tensors = 10
        total_dim_range = (2, 4)
        min_dim = 1
        max_dim = 20  # Size of any dimension will be between 1 and 20
        nnz_range = (1, 10)  # Range of non-zero elements per tensor
        is_uncoalesced = True  # Set to True to keep duplicate indices
        device = "cpu"
        dtype_list = [torch.float32, torch.float16]
        for dtype in dtype_list:
            tensors = self.generate_sparse_tensors(
                num_tensors,
                total_dim_range,
                min_dim,
                max_dim,
                nnz_range,
                is_uncoalesced,
                device,
                dtype,
            )
            for sparse_tensor in tensors:
                out_cpu = sparse_tensor.coalesce()
                out_mlu = sparse_tensor.to("mlu").coalesce()
                self.compare_tensors(out_cpu, out_mlu, 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_Coalesce_empty_i(self):
        shape = (1, 2)
        indices = torch.tensor([]).view(0, 5)
        values = torch.rand(5, 1, 2)
        dtype_list = [torch.float32, torch.float16]
        for dtype in dtype_list:
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, size=shape, dtype=dtype
            )
            out_cpu = sparse_tensor.coalesce()
            out_mlu = sparse_tensor.to("mlu").coalesce()
            self.compare_tensors(out_cpu, out_mlu, 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_Coalesce_empty_v(self):
        shape = (5, 3, 1, 0, 2)
        indices = torch.tensor([[0, 3, 1, 0], [1, 0, 1, 0]])
        values = torch.rand(4, 1, 0, 2)
        dtype_list = [torch.float32, torch.float16]
        for dtype in dtype_list:
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, size=shape, dtype=dtype
            )
            out_cpu = sparse_tensor.coalesce()
            out_mlu = sparse_tensor.to("mlu").coalesce()
            self.assertTensorsEqual(out_cpu._indices(), out_mlu.cpu()._indices(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_Coalesce_empty_nnz(self):
        shape = (7, 3, 4, 5, 2)
        indices = torch.tensor([]).view(2, 0)
        values = torch.tensor([]).view(0, 4, 5, 2)
        dtype_list = [torch.float32, torch.float16]
        for dtype in dtype_list:
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, size=shape, dtype=dtype
            )
            out_cpu = sparse_tensor.coalesce()
            out_mlu = sparse_tensor.to("mlu").coalesce()
            self.compare_tensors(out_cpu, out_mlu, 0.003)


if __name__ == "__main__":
    unittest.main()
