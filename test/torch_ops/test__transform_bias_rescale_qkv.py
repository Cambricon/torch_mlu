from __future__ import print_function

import sys
import os

# pylint: disable=all
import unittest

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
)  # pylint: disable=C0413,C0411


class Test_TransformBiasRescaleQKV(TestCase):
    def _create_args(self, B, T, embed_dim, dtype, device="cpu"):
        qkv = torch.randn([B, T, 3 * embed_dim], dtype=dtype, device=device)
        qkv_bias = torch.randn([3 * embed_dim], dtype=dtype, device=device)
        return qkv, qkv_bias

    def _test__transform_bias_rescale_qkv(self, B, T, embed_dim, num_heads, dtype):
        qkv, qkv_bias = self._create_args(B, T, embed_dim, dtype, device="cpu")
        q_cpu, k_cpu, v_cpu = torch._transform_bias_rescale_qkv(
            qkv, qkv_bias, num_heads
        )
        q_mlu, k_mlu, v_mlu = torch._transform_bias_rescale_qkv(
            qkv.mlu(), qkv_bias.mlu(), num_heads
        )
        self.assertTrue(q_mlu.device.type == "mlu")
        self.assertTrue(q_mlu.is_contiguous())
        tol = 1e-6
        if dtype == torch.half:
            tol = 1e-3
        elif dtype == torch.bfloat16:
            tol = 2e-2
        self.assertTensorsEqual(q_cpu, q_mlu.cpu(), tol, use_MSE=True)
        self.assertTensorsEqual(k_cpu, k_mlu.cpu(), tol, use_MSE=True)
        self.assertTensorsEqual(v_cpu, v_mlu.cpu(), tol, use_MSE=True)

    @testinfo()
    def test_transform_bias_rescale_qkv_mlu(self):
        dtypes = [torch.float32, torch.half]
        if torch.mlu.get_device_properties(torch.mlu.current_device()).major >= 5:
            dtypes.append(torch.bfloat16)
        shapes = [
            [4, 16, 192, 4],
            [3, 32, 256, 8],
            [12, 32, 256, 8],
            [89, 32, 256, 8],
            [148, 32, 256, 8],
            [370, 32, 256, 8],
            [3, 16, 1536, 16],
            [113, 17, 1536, 16],
        ]
        for dtype in dtypes:
            for B, T, embed_dim, num_heads in shapes:
                self._test__transform_bias_rescale_qkv(
                    B, T, embed_dim, num_heads, dtype
                )

    @testinfo()
    def test_zero_num_elements(self):
        shapes = [
            [4, 32, 0, 8],
        ]
        dtypes = [torch.float32]
        for dtype in dtypes:
            for B, T, embed_dim, num_heads in shapes:
                qkv, qkv_bias = self._create_args(B, T, embed_dim, dtype, device="mlu")
                q_mlu, k_mlu, v_mlu = torch._transform_bias_rescale_qkv(
                    qkv, qkv_bias, num_heads
                )

    @testinfo()
    def test_max_size_of_lowest_dim(self):
        restriction = [
            # dtype, size_of_lowest_dim, test_passed
            [torch.float32, 18432, True],
            [torch.float32, 18480, False],
            [torch.float16, 32760, True],
            [torch.float16, 32880, False],
        ]
        B = 2
        T = 4
        num_heads = 8
        for dtype, dim_size, test_passed in restriction:
            if test_passed:
                qkv, qkv_bias = self._create_args(
                    B, T, int(dim_size / 3), dtype, device="mlu"
                )
                q_mlu, k_mlu, v_mlu = torch._transform_bias_rescale_qkv(
                    qkv, qkv_bias, num_heads
                )
            else:
                error_msg = "_transform_bias_rescale_qkv: MLU does not support qkv.size\(2\) exceed {} for {}.".format(
                    dim_size, dtype
                )
                qkv, qkv_bias = self._create_args(
                    B, T, int(dim_size / 3), dtype, device="mlu"
                )
                with self.assertRaisesRegex(RuntimeError, error_msg):
                    q_mlu, k_mlu, v_mlu = torch._transform_bias_rescale_qkv(
                        qkv, qkv_bias, num_heads
                    )

    @testinfo()
    def test_large_tensor(self):
        shape_and_dtypes = [
            # B, T, embed_dim, num_heads, dtype
            [1024, 32, 18432, 8, torch.float32],  # 2.25G
            [2048, 32, 18432, 8, torch.float16],  # 2.25G
        ]
        for B, T, embed_dim, num_heads, dtype in shape_and_dtypes:
            self._test__transform_bias_rescale_qkv(
                B, T, int(embed_dim / 3), num_heads, dtype
            )


if __name__ == "__main__":
    unittest.main()
