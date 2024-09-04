import sys
import os
import logging
import unittest
from itertools import product
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411


class TestUniqueConsecutive(TestCase):
    def _test_unique_consecutive(
        self, input_cpu, input_mlu, return_inverse, return_counts, dim
    ):
        if (
            return_inverse and return_counts
        ):  # set both return_ivnerse and return_counts
            output_cpu, inverse_indices_cpu, counts_cpu = torch.unique_consecutive(
                input_cpu,
                return_inverse=return_inverse,
                return_counts=return_counts,
                dim=dim,
            )
            output_mlu, inverse_indices_mlu, counts_mlu = torch.unique_consecutive(
                input_mlu,
                return_inverse=return_inverse,
                return_counts=return_counts,
                dim=dim,
            )
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(inverse_indices_cpu, inverse_indices_mlu.cpu(), 0)
            self.assertTensorsEqual(counts_cpu, counts_mlu.cpu(), 0)
            self.assertEqual(output_cpu.dtype, output_mlu.dtype)
            self.assertEqual(inverse_indices_cpu.dtype, inverse_indices_mlu.dtype)
            self.assertEqual(counts_cpu.dtype, counts_mlu.dtype)

        elif return_inverse:  # only set return_inverse
            output_cpu, inverse_indices_cpu = torch.unique_consecutive(
                input_cpu, return_inverse=return_inverse, dim=dim
            )
            output_mlu, inverse_indices_mlu = torch.unique_consecutive(
                input_mlu, return_inverse=return_inverse, dim=dim
            )
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(inverse_indices_cpu, inverse_indices_mlu.cpu(), 0)
            self.assertEqual(output_cpu.dtype, output_mlu.dtype)
            self.assertEqual(inverse_indices_cpu.dtype, inverse_indices_mlu.dtype)

        elif return_counts:  # only set return_counts
            output_cpu, counts_cpu = torch.unique_consecutive(
                input_cpu, return_counts=return_counts, dim=dim
            )
            output_mlu, counts_mlu = torch.unique_consecutive(
                input_mlu, return_counts=return_counts, dim=dim
            )
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(counts_cpu, counts_mlu.cpu(), 0)
            self.assertEqual(output_cpu.dtype, output_mlu.dtype)
            self.assertEqual(counts_cpu.dtype, counts_mlu.dtype)

        else:
            output_cpu = torch.unique_consecutive(
                input_cpu, return_counts=return_counts, dim=dim
            )
            output_mlu = torch.unique_consecutive(
                input_mlu, return_counts=return_counts, dim=dim
            )
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertEqual(output_cpu.dtype, output_mlu.dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_consecutive(self):
        # RuntimeError: "unique" not implemented for 'Half'
        # https://github.com/pytorch/pytorch/issues/30883
        dtype_list = [
            torch.float64,
            torch.float32,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
        shape_list = [(64,), (4, 64), (3, 4, 64), (100, 64, 7, 7)]
        return_inverse_list = [True, False]
        return_counts_list = [True, False]
        for shape in shape_list:
            dim_list = [None] + list(range(-len(shape), len(shape)))
            loop_var = [
                dtype_list,
                [shape],
                return_inverse_list,
                return_counts_list,
                dim_list,
            ]
            for param in product(*loop_var):
                torch.manual_seed(1)
                dtype, shape, return_inverse, return_counts, dim = param
                input_cpu = torch.randint(0, 64, shape).to(dtype)
                input_mlu = input_cpu.to("mlu")
                self._test_unique_consecutive(
                    input_cpu, input_mlu, return_inverse, return_counts, dim
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_consecutive_zero_element(self):
        return_inverse_list = [True, False]
        return_counts_list = [True, False]
        dim_list = [None, 0]
        loop_var = [return_inverse_list, return_counts_list, dim_list]
        for param in product(*loop_var):
            torch.manual_seed(1)
            return_inverse, return_counts, dim = param
            input_cpu = torch.Tensor([])
            input_mlu = input_cpu.to("mlu")
            self._test_unique_consecutive(
                input_cpu, input_mlu, return_inverse, return_counts, dim
            )

    # @unittst.skip("not test")
    @testinfo()
    def test_unique_consecutive_channel_last(self):
        dtype_list = [
            torch.float64,
            torch.float32,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
        shape_list = [(100, 64, 7, 7)]
        return_inverse_list = [True, False]
        return_counts_list = [True, False]
        dim_list = [None, 0]
        loop_var = [
            dtype_list,
            shape_list,
            return_inverse_list,
            return_counts_list,
            dim_list,
        ]
        for param in product(*loop_var):
            torch.manual_seed(1)
            dtype, shape, return_inverse, return_counts, dim = param
            input_cpu = (
                torch.randint(0, 64, shape)
                .to(dtype)
                .to(memory_format=torch.channels_last)
            )
            input_mlu = input_cpu.to("mlu")
            self._test_unique_consecutive(
                input_cpu, input_mlu, return_inverse, return_counts, dim
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_consecutive_not_dense(self):
        dtype_list = [
            torch.float64,
            torch.float32,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
        shape_list = [(64,), (4, 64), (3, 4, 64), (100, 64, 7, 7)]
        return_inverse_list = [True, False]
        return_counts_list = [True, False]
        dim_list = [None, 0]
        loop_var = [
            dtype_list,
            shape_list,
            return_inverse_list,
            return_counts_list,
            dim_list,
        ]
        for param in product(*loop_var):
            torch.manual_seed(1)
            dtype, shape, return_inverse, return_counts, dim = param
            input_cpu = torch.randint(0, 64, shape).to(dtype)
            input_mlu = input_cpu.to("mlu")
            self._test_unique_consecutive(
                input_cpu, input_mlu, return_inverse, return_counts, dim
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_consecutive_exception(self):
        x_mlu = torch.arange(1, 9, dtype=torch.uint8).to("mlu")
        ref_msg = r"\"unique_consecutive\" not implemented for"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.unique_consecutive(x_mlu, return_inverse=True, return_counts=True)

        shape = (0, 2, 0, 3)
        x_mlu = torch.randint(0, 64, shape).mlu()
        ref_msg = "Number of zero sized dimensions is more than one, so unique cannot be applied "
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.unique_consecutive(
                x_mlu, return_inverse=True, return_counts=True, dim=0
            )

        shape = (0, 2, 1, 3)
        x_mlu = torch.randint(0, 64, shape).mlu()
        ref_msg = "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.unique_consecutive(
                x_mlu, return_inverse=True, return_counts=True, dim=1
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_unique_consecutive_spec(self):
        x = torch.zeros((0, 0, 3), dtype=torch.float, device="mlu")
        _, inverse_indices, _ = torch.unique_consecutive(
            x, return_inverse=True, return_counts=True
        )
        expected_inverse_indices = torch.empty(
            (0, 0, 3), dtype=torch.long, device="mlu"
        )
        self.assertTrue(inverse_indices.shape == expected_inverse_indices.shape)


if __name__ == "__main__":
    unittest.main()
