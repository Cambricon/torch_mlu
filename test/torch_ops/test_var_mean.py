from __future__ import print_function

import sys
import os
import itertools
import unittest
import logging
import copy
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


# The var_mean operator calculates the variance for each row of the input tensor in a given
def to_non_dense_channels_last(data, dim=None, distance=2):
    if not type(data) == torch.Tensor:
        print(
            "[Warning]: It's not available to convert an unknown object to non-dense type"
        )
        return data
    # convert the last channel as default.
    convert_dim = data.dim()
    if dim is not None:
        convert_dim = dim
    if convert_dim > data.dim():
        print(
            f"[Warning]: The max available expand dim for a {data.dim()} Tensor"
            f" is {data.dim()}, but got specified dim as {dim}."
        )
        convert_dim = data.dim()
    a = data.unsqueeze(convert_dim)
    b = torch.cat([a for _ in range(distance)], convert_dim)
    return b.select(dim=convert_dim, index=0)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_var_mean_dim(self):
        type_list = [True, False]
        shape_list = [
            (3, 2, 128, 10, 6),
            (2, 128, 10, 6),
            (200, 1536, 202),
            (2, 100),
            (24,),
        ]
        for shape in shape_list:
            dim_len = len(shape)
            dim_lists = list(range(dim_len))
            for test_dim in dim_lists:
                for test_type in type_list:
                    x = torch.randn(shape, dtype=torch.float)
                    out_cpu = torch.var_mean(x, dim=test_dim, keepdim=test_type)
                    out_mlu = torch.var_mean(x.mlu(), dim=test_dim, keepdim=test_type)
                    self.assertTensorsEqual(
                        out_cpu[0], out_mlu[0].cpu(), 0.003, use_MSE=True
                    )
                    self.assertTensorsEqual(
                        out_cpu[1], out_mlu[1].cpu(), 0.003, use_MSE=True
                    )

            dim_lists_neg = list(itertools.permutations(range(-dim_len, 0), 1))
            for test_dim in dim_lists_neg:
                for test_type in type_list:
                    x = torch.randn(shape, dtype=torch.float)
                    out_cpu = torch.var_mean(x, dim=test_dim, keepdim=test_type)
                    out_mlu = torch.var_mean(x.mlu(), dim=test_dim, keepdim=test_type)
                    self.assertTensorsEqual(
                        out_cpu[0], out_mlu[0].cpu(), 0.003, use_MSE=True
                    )
                    self.assertTensorsEqual(
                        out_cpu[1], out_mlu[1].cpu(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_var_mean(self):
        shape_list = [(3, 2, 128, 10, 6), (2, 128, 10, 6), (2, 512, 8), (2, 100), (24,)]
        correction_list = [0, 1]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float32)
            for correction in correction_list:
                # (TODO) guolin var_mean only Calculates the var_meaniance
                # for each row of the input tensor in a given,not spport dims
                out_cpu = torch.var_mean(x, dim=0, correction=correction)
                out_mlu = torch.var_mean(self.to_mlu(x), dim=0, correction=correction)
                self.assertTensorsEqual(
                    out_cpu[0], out_mlu[0].cpu(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    out_cpu[1], out_mlu[1].cpu(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_var_mean_channel_last(self):
        shape_list = [(3, 2, 128, 10, 6), (2, 128, 10, 6), (2, 512, 8), (2, 100), (24,)]
        correction_list = [0, 1]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float32)
            x_mlu = x.mlu()
            x = self.convert_to_channel_last(x)
            x_mlu = self.convert_to_channel_last(x_mlu)
            for correction in correction_list:
                # (TODO) guolin var_mean only Calculates the var_meaniance for each
                # row of the input tensor in a given,not spport dims
                out_cpu = torch.var_mean(x, dim=0, correction=correction)
                out_mlu = torch.var_mean(x_mlu, dim=0, correction=correction)
                self.assertTensorsEqual(
                    out_cpu[0], out_mlu[0].cpu(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    out_cpu[1], out_mlu[1].cpu(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_var_mean_non_dense_channel_last(self):
        x = torch.randn(3, 5, 3, 3)
        out_cpu = torch.var_mean(x, (-1,))
        out_mlu = torch.var_mean(to_non_dense_channels_last(x.mlu(), 2), (-1,))
        self.assertTensorsEqual(out_cpu[0], out_mlu[0].cpu(), 0.003, use_MSE=True)
        self.assertTensorsEqual(out_cpu[1], out_mlu[1].cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_mean_no_contiguous(self):
        a = torch.randn(4, 4, dtype=torch.float)
        x = a[::2, ::2]
        out_cpu = torch.var_mean(x, dim=0, keepdim=False)
        a = a.to("mlu")
        x = a[::2, ::2]
        out_mlu = torch.var_mean(x, dim=0, keepdim=False)
        self.assertTensorsEqual(out_cpu[0], out_mlu[0].cpu(), 0.003, use_MSE=True)
        self.assertTensorsEqual(out_cpu[1], out_mlu[1].cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_mean_empty_input(self):
        shape = (2, 0, 4)
        x = torch.randn(shape, dtype=torch.float32)
        keepdim_list = [True, False]
        correction_list = [0, 1]
        # In this test case, empty tensors such as tensor([], device='mlu:0', size=(2, 0, 1)
        # or tensor([], device='mlu:0', size=(2, 0)) will be generated
        for keepdim, correction in itertools.product(keepdim_list, correction_list):
            out_cpu = torch.var_mean(x, dim=2, keepdim=keepdim, correction=correction)
            out_mlu = torch.var_mean(
                x.mlu(), dim=2, keepdim=keepdim, correction=correction
            )
            self.assertTensorsEqual(out_cpu[0], out_mlu[0].cpu(), 0, use_MSE=False)
            self.assertTensorsEqual(out_cpu[1], out_mlu[1].cpu(), 0, use_MSE=False)
        # In this test case, non-empty tensor filled with nan will be generated, such as
        # tensor([[[nan, nan, nan, nan]], [[nan, nan, nan, nan]]], device='mlu:0')
        for keepdim, correction in itertools.product(keepdim_list, correction_list):
            out_cpu = torch.var_mean(x, dim=1, keepdim=keepdim, correction=correction)
            out_mlu = torch.var_mean(
                x.mlu(), dim=1, keepdim=keepdim, correction=correction
            )
            self.assertEqual(out_cpu[0].shape, out_mlu[0].shape)
            self.assertEqual(out_cpu[1].shape, out_mlu[1].shape)
            self.assertTrue(torch.isnan(out_mlu[0]).all())
            self.assertTrue(torch.isnan(out_mlu[1]).all())
        # In this test case, dim will not be provided, and the output should be an empty tensor
        # filled with nan like tensor(nan, device='mlu:0')
        for correction in correction_list:
            out_cpu = torch.var_mean(x, correction=correction)
            out_mlu = torch.var_mean(x.mlu(), correction=correction)
            self.assertEqual(out_cpu[0].shape, out_mlu[0].shape)
            self.assertEqual(out_cpu[1].shape, out_mlu[1].shape)
            self.assertTrue(torch.isnan(out_mlu[0]).all())
            self.assertTrue(torch.isnan(out_mlu[1]).all())

    # @unittest.skip("not test")
    @testinfo()
    def test_var_mean_negative_dims(self):
        shape_list = [
            (3, 2, 128, 10, 6),
            (2, 128, 10, 6),
            (200, 1536, 202),
            (2, 100),
            (24,),
        ]
        keepdim_list = [True, False]
        correction_list = [0, 1]
        for items in itertools.product(shape_list, keepdim_list, correction_list):
            shape = items[0]
            keepdim = items[1]
            correction = items[2]
            for dim in range(-len(shape), -1):
                x = torch.randn(shape, dtype=torch.float32)
                out_cpu = torch.var_mean(
                    x, dim=dim, keepdim=keepdim, correction=correction
                )
                out_mlu = torch.var_mean(
                    x.mlu(), dim=dim, keepdim=keepdim, correction=correction
                )
                self.assertTensorsEqual(
                    out_cpu[0], out_mlu[0].cpu(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    out_cpu[1], out_mlu[1].cpu(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_var_mean_all(self):
        shape_list = [
            (3, 2, 128, 10, 6),
            (2, 128, 10, 6),
            (200, 1536, 202),
            (2, 100),
            (24,),
        ]
        keepdim_list = [True, False]
        correction_list = [0, 1]
        for items in itertools.product(shape_list, keepdim_list, correction_list):
            shape = items[0]
            keepdim = items[1]
            correction = items[2]
            x = torch.randn(shape, dtype=torch.float32)
            out_cpu = torch.var_mean(x, dim=[], keepdim=keepdim, correction=correction)
            out_mlu = torch.var_mean(
                x.mlu(), dim=[], keepdim=keepdim, correction=correction
            )
            self.assertTensorsEqual(out_cpu[0], out_mlu[0].cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(out_cpu[1], out_mlu[1].cpu(), 0.003, use_MSE=True)

        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float32)
            out_cpu = torch.var_mean(x)
            out_mlu = torch.var_mean(x.mlu())
            self.assertTensorsEqual(out_cpu[0], out_mlu[0].cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(out_cpu[1], out_mlu[1].cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_mean_multidims(self):
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6)]
        dtype_list = [torch.float, torch.half]
        dim_list = [[0, 2], [1, 3], [0, 1, 3], [-1, -2, 0], None]
        correction_list = [0, 1]
        keepdim_list = [True, False]
        for channels_last in [True, False]:
            for item in product(
                shape_list, dtype_list, dim_list, correction_list, keepdim_list
            ):
                x = torch.randn(item[0], dtype=item[1])
                if channels_last:
                    x = self.convert_to_channel_last(x)
                x_mlu = self.to_device(copy.deepcopy(x))

                out_cpu = torch.var_mean(
                    x, dim=item[2], correction=item[3], keepdim=item[4]
                )
                out_mlu = torch.var_mean(
                    x_mlu, dim=item[2], correction=item[3], keepdim=item[4]
                )
                self.assertTensorsEqual(
                    out_cpu[0].float(), out_mlu[0].cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    out_cpu[1].float(), out_mlu[1].cpu().float(), 0.003, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("41GB")
    def test_var_mean_large(self):
        shape_list = [(4, 1025, 1024 * 1024), (1, 4 * 1025 * 1024 * 1024)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float32)
            out_cpu = torch.var_mean(x, dim=0)
            out_mlu = torch.var_mean(self.to_mlu(x), dim=0)
            self.assertTensorsEqual(out_cpu[0], out_mlu[0].cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(out_cpu[1], out_mlu[1].cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_mean_out_exceptions(self):
        ref_msg = r"now correction only supports 0 and 1 but got 2"
        x = torch.randn(5, 5, dtype=torch.float, device="mlu")
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.var_mean(x, correction=2)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_var_mean_bfloat16(self):
        keepdim_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(keepdim_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.bfloat16)
                x_mlu = self.to_device(x)
                out_cpu = torch.var_mean(x, item[1], keepdim=item[0])
                out_mlu = torch.var_mean(x_mlu, item[1], keepdim=item[0])
                self.assertTensorsEqual(
                    out_cpu[0].float(), out_mlu[0].cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    out_cpu[1].float(), out_mlu[1].cpu().float(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
