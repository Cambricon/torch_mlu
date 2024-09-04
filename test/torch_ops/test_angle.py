from __future__ import print_function
import os

import sys
import logging
import unittest
import torch
import copy
import numpy as np
from itertools import product
from torch.testing._internal.common_utils import torch_to_numpy_dtype_dict
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestAngleOp(TestCase):
    """
    Test Angle op
    """

    shape_list = [
        (512, 1024, 2, 2, 4),
        (10, 3, 32, 32),
        (2, 3, 4),
        (254, 254, 112, 1, 1, 3),
        (10),
        (),
    ]
    rtol = 1.3e-6
    atol = 1e-05

    # @unittest.skip("not test")
    @testinfo()
    def test_angle_spec_op(self):
        datas = [
            torch.tensor(
                [
                    complex(3, 4),
                    complex(float("nan"), 3),
                    complex(3, float("nan")),
                    complex(float("nan"), float("nan")),
                    complex(3, -float("inf")),
                    complex(3, float("inf")),
                    complex(-float("inf"), 4),
                    complex(-float("inf"), 4),
                    complex(float("inf"), float("inf")),
                    complex(-float("inf"), float("inf")),
                    complex(float("inf"), -float("inf")),
                    complex(float("inf"), float("nan")),
                    complex(-float("inf"), float("nan")),
                    complex(-float("inf"), float("nan")),
                    complex(-float("inf"), float("nan")),
                ]
            ),
            torch.tensor([-100, 100, float("inf"), -float("inf"), float("nan")]),
        ]
        for data in datas:
            if data.is_complex():
                out_cpu = torch.angle(data)
                out_mlu = torch.angle(data.to(torch.device("mlu")))
                self.assertEqual(out_cpu, out_mlu.cpu(), rtol=self.rtol, atol=self.atol)
            else:
                for dtype in [
                    torch.int8,
                    torch.uint8,
                    torch.int16,
                    torch.int32,
                    torch.float,
                    torch.double,
                ]:
                    data = data.type(dtype)
                    out_cpu = torch.angle(data)
                    out_mlu = torch.angle(data.to(torch.device("mlu")))
                    self.assertEqual(out_cpu, out_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_angle_complex_to_float(self, dtype=torch.complex64):
        from random import random

        random_vals = []
        for multiplier in [-1, 1, -10, 10, -100, 100]:
            for _ in range(10):
                random_vals.append(
                    complex(random() * multiplier, random() * multiplier)  # NOSONAR
                )  # NOSONAR
        for vals in [random_vals, []]:
            a = np.array(vals, dtype=torch_to_numpy_dtype_dict[dtype])
            t = torch.tensor(vals, dtype=dtype)
            fn_name = "angle"

            torch_fn = getattr(torch, fn_name)
            np_fn = getattr(np, fn_name)

            # Tests function
            np_result = torch.from_numpy(np_fn(a))  # NOSONAR
            torch_result = torch_fn(t.to("mlu")).cpu()  # NOSONAR

            # Tests float out
            float_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
            np_float_out = np_fn(a).astype(torch_to_numpy_dtype_dict[float_dtype])
            float_out = torch.empty_like(t, dtype=float_dtype).float().to("mlu")
            torch_fn(t.to("mlu"), out=float_out)
            self.assertEqual(torch.from_numpy(np_float_out), float_out.cpu())

            # Tests float out (resized out)
            float_out = torch.empty(1, dtype=float_dtype).to("mlu")
            torch_fn(t.to("mlu"), out=float_out)
            self.assertEqual(
                torch.from_numpy(np_float_out),
                float_out.cpu(),
                rtol=self.rtol,
                atol=self.atol,
            )

            # Tests complex out
            np_complex_out = np_fn(a).astype(torch_to_numpy_dtype_dict[dtype])
            complex_out = torch.empty_like(t).to("mlu")
            torch_fn(t.to("mlu"), out=complex_out)
            # Tests complex out (resized out)
            complex_out = torch.empty(0, dtype=dtype).to("mlu")
            torch_fn(t.to("mlu"), out=complex_out)
            self.assertEqual(
                torch.from_numpy(np_complex_out),
                complex_out.cpu(),
                rtol=self.rtol,
                atol=self.atol,
            )

            # Tests long out behavior (expected failure)
            long_out = torch.empty(0, dtype=torch.long).to("mlu")
            with self.assertRaises(RuntimeError):
                torch_fn(t.to("mlu"), out=long_out)

            # Note: angle does not have an in-place variant
            if fn_name == "angle":
                with self.assertRaises(AttributeError):
                    torch_inplace_method = getattr(
                        torch.Tensor, fn_name + "_"
                    )  # NOSONAR

    # @unittest.skip("not test")
    @testinfo()
    def test_angle(self):
        shape_list = [(512, 1024, 2, 2), (2, 3, 4), (254, 254, 112, 1, 1, 3)]
        dtype_list = [
            torch.double,
            torch.float,
            torch.complex64,
            torch.complex128,
            torch.int64,
            torch.int,
            torch.int16,
            torch.uint8,
            torch.int8,
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype, func in product(shape_list, dtype_list, func_list):
            x = torch.testing.make_tensor(shape, dtype=dtype, device="cpu")
            x_mlu_temp = copy.deepcopy(x).mlu()
            if dtype.is_floating_point or dtype.is_complex:
                x_cpu = torch.nn.Parameter(x)
                x_mlu = torch.nn.Parameter(x_mlu_temp)
            else:
                x_cpu = x
                x_mlu = x_mlu_temp
            out_cpu = torch.angle(func(x_cpu))
            out_mlu = torch.angle(func(x_mlu))
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 1e-05, use_MSE=True)
            # TODO(PYTORCH-9180): backward is not support now.
            # if dtype.is_floating_point or dtype.is_complex:
            #    grad_cpu = torch.testing.make_tensor(out_cpu.shape, dtype=out_cpu.dtype, device="cpu")
            #    grad_mlu = grad_cpu.mlu()
            #    out_cpu.backward(grad_cpu)
            #    out_mlu.backward(grad_mlu)
            #    self.assertEqual(x_cpu.grad.dtype, x_mlu.grad.dtype)
            #    self.assertTensorsEqual(x_cpu.grad, x_mlu.grad.cpu(), 1e-05, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_angle_out(self):
        shape_list = [(512, 1024, 2, 2), (2, 3, 4), (254, 254, 112, 1, 1, 3)]
        dtype_list = [
            torch.double,
            torch.float,
            torch.complex64,
            torch.complex128,
            torch.int64,
            torch.int,
            torch.int16,
            torch.uint8,
            torch.int8,
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype, func in product(shape_list, dtype_list, func_list):
            if not dtype.is_floating_point or not dtype.is_complex:
                output_dtype = torch.float
            else:
                output_dtype = dtype
            x = func(torch.testing.make_tensor(shape, dtype=dtype, device="cpu"))
            x = func(torch.testing.make_tensor(shape, dtype=dtype, device="cpu"))
            x_mlu = func(copy.deepcopy(x).mlu())
            out_cpu = torch.testing.make_tensor(shape, dtype=output_dtype, device="cpu")
            out_mlu = copy.deepcopy(out_cpu).mlu()
            ori_ptr = out_mlu.data_ptr()
            out_cpu = torch.angle(x, out=out_cpu)
            out_mlu = torch.angle(x_mlu, out=out_mlu)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertEqual(ori_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 1e-05, use_MSE=True)
            # resize output
            out_cpu = torch.empty((0,), dtype=output_dtype)
            out_mlu = copy.deepcopy(out_cpu).mlu()
            ori_ptr = out_mlu.data_ptr()
            out_cpu = torch.angle(x, out=out_cpu)
            out_mlu = torch.angle(x_mlu, out=out_mlu)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTrue(ori_ptr != out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 1e-05, use_MSE=True)

    # Test is_non_overlapping_and_dense
    # @unittest.skip("not test")
    @testinfo()
    def test_angle_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.testing.make_tensor(
                shape_list[i], dtype=torch.float, device="cpu"
            )
            out = torch.testing.make_tensor(
                shape_list[i], dtype=torch.float, device="cpu"
            )
            x_mlu = copy.deepcopy(x).mlu()
            out_mlu = copy.deepcopy(out).mlu()
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                permute_shape[i]
            )
            # test output
            torch.angle(x, out=out)
            torch.angle(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 1e-05, use_MSE=True)
            # test functional
            out = torch.angle(x)
            out_mlu = torch.angle(x_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 1e-05, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_angle_zero_element(self):
        # test zero element
        input_cpu = torch.randn(0, 2, 2, dtype=torch.float)
        input_mlu = copy.deepcopy(input_cpu).mlu()
        out_cpu = torch.angle(input_cpu)
        out_mlu = torch.angle(input_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 1e-05, use_MSE=True)

    # TODO(shangang): Fallback will skip this error.
    @unittest.skip("not test")
    @testinfo()
    def test_exception(self):
        mlu_x = torch.testing.make_tensor(
            (2, 3, 4), dtype=torch.half, device="cpu"
        ).mlu()
        with self.assertRaisesRegex(
            RuntimeError, r"MLU angle don't support tensor dtype Half."
        ):
            torch.angle(mlu_x)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_angle_bfloat16(self):
        input_cpu = torch.randn(1, 2, 2, dtype=torch.bfloat16)
        input_mlu = copy.deepcopy(input_cpu).mlu()
        out_cpu = torch.angle(input_cpu)
        out_mlu = torch.angle(input_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 1e-05, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("37GB")
    def test_angle_large(self):
        shape = (4, 1025, 1024, 1024)
        dtype = torch.float
        x = torch.testing.make_tensor(shape, dtype=dtype, device="cpu")
        x_mlu = x.to("mlu")
        out_cpu = torch.angle(x)
        out_mlu = torch.angle(x_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 1e-05, use_MSE=True)


if __name__ == "__main__":
    run_tests()
