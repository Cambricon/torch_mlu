from __future__ import print_function
from re import L

import sys
import os
from itertools import product
import unittest
import logging
import copy
import math
import torch
import numpy as np
from torch.testing._internal.common_utils import make_tensor

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestFmodOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_fmod(self):
        dtype_list = [
            (torch.half, 3e-3),
            (torch.int32, 3e-3),
            (torch.long, 3e-3),
            (torch.float, 3e-3),
        ]
        shape_list = [(10, 12, 10, 13), (2, 10, 15)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype_err, in_shape, func in product(dtype_list, shape_list, func_list):
            dtype, err = dtype_err
            x_cpu = (20 * torch.randn(in_shape)).to(dtype)
            y_cpu = (20 * torch.rand(in_shape)).to(dtype) + 1
            x_mlu = x_cpu.to("mlu")
            y_mlu = y_cpu.to("mlu")
            out_cpu = torch.fmod(func(x_cpu), func(y_cpu))
            out_mlu = torch.fmod(func(x_mlu), func(y_mlu))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )

        # test sliced output
        x_cpu = 20 * torch.randn(10, 25)
        y_cpu = 20 * torch.rand(10, 25) + 1
        out_cpu = torch.randn(10, 50)
        x_mlu = x_cpu.to("mlu")
        y_mlu = y_cpu.to("mlu")
        out_mlu = out_cpu.to("mlu")
        torch.fmod(x_cpu, y_cpu, out=out_cpu[:, :25])
        torch.fmod(x_mlu, y_mlu, out=out_mlu[:, :25])
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_fmod_ori(self):
        device = "mlu"
        zero_d = torch.randn((), device=device)
        one_d = torch.randn((1,), device=device)
        # fmod
        self.assertEqual((), torch.fmod(zero_d, zero_d).shape)
        self.assertEqual((), torch.fmod(zero_d, 2).shape)
        self.assertEqual((1,), torch.fmod(zero_d, one_d).shape)
        self.assertEqual((1,), torch.fmod(one_d, zero_d).shape)

        dtype = torch.float
        m1 = torch.Tensor(10, 10).uniform_(-10.0, 10.0).to(dtype=dtype, device=device)
        res1 = m1.clone()
        q = 2.1
        res1[:, 3].fmod_(q)
        res2 = m1.clone()
        for i in range(m1.size(1)):
            res2[i, 3] = math.fmod(res2[i, 3], q)
        self.assertEqual(res1, res2)

        # test fmod remainder
        # Use numpy as reference
        def _helper(x, mod):
            fns_list = [(torch.fmod, torch.Tensor.fmod_, np.fmod)]
            for fn, inplace_fn, ref_fn in fns_list:
                np_x = x.cpu().numpy()
                np_mod = mod.cpu().numpy() if torch.is_tensor(mod) else mod
                exp = ref_fn(np_x, np_mod)
                exp = torch.from_numpy(exp)
                res = fn(x, mod)

                self.assertEqual(res, exp, exact_dtype=False)
                # out
                out = torch.empty(0, device=device, dtype=res.dtype)
                fn(x, mod, out=out)
                self.assertEqual(out, exp, exact_dtype=False)
                self.assertEqual(out.size(), torch.Size([10, 10]))
                # in-place (Type cast runtime error)
                try:
                    inplace_fn(x, mod)
                    self.assertEqual(x, exp, exact_dtype=False)
                except RuntimeError as e:
                    self.assertRegex(
                        str(e),
                        "result type (Half|Float|Double) "
                        "can't be cast to the desired output "
                        "type (Byte|Char|Short|Int|Long)",
                    )

        x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
        # mod with same dtype as x
        mod = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
        # Exclude 0
        mod[mod == 0] = 1

        # Mods: Integer, Float, Tensor, Non-contiguous Tensor
        mods = [3, 2.3, mod, mod.t()]
        # mod with floating-point dtype
        if dtype in [torch.half, torch.float, torch.int]:
            mod_float = make_tensor(
                (10, 10), device=device, dtype=torch.float, low=-9, high=9
            )
            mod[mod == 0] = 1
            mods.append(mod_float)

        for dividend, mod in product([x, x.t()], mods):
            _helper(dividend, mod)

        # test large dividend
        dtype = torch.float
        alarge = 1e9
        pi = 3.14159265358979
        for avalue in [alarge, -alarge]:
            for bvalue in [pi, -pi]:
                a = torch.tensor([avalue], dtype=dtype, device=device)
                b = torch.tensor([bvalue], dtype=dtype, device=device)
                c = torch.remainder(a, b)
                d = torch.fmod(a, b)
                self.assertTrue(
                    (b[0] > 0) == (c[0] > 0)
                )  # remainder has same sign as divisor
                self.assertTrue(
                    (a[0] > 0) == (d[0] > 0)
                )  # fmod has same sign as dividend
                self.assertTrue(
                    abs(c[0]) < abs(b[0])
                )  # remainder is within range of divisor
                self.assertTrue(
                    abs(d[0]) < abs(b[0])
                )  # fmod is within range of divisor
                if (a[0] > 0) == (b[0] > 0):  # pylint: disable=C0325
                    self.assertTrue(c[0] == d[0])  # remainder is same as fmod
                else:
                    self.assertTrue(
                        abs(c[0] - d[0]) == abs(b[0])
                    )  # differ by one divisor

        for dtype in [torch.half, torch.float, torch.int]:
            # check floating-point tensor fmod to zero is nan on both CPU and GPU
            x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
            zero = torch.zeros_like(x)
            if dtype == torch.int:
                self.assertTrue(torch.all(torch.fmod(x, zero) == -1))
            else:
                self.assertTrue(torch.all(torch.fmod(x, 0.0).isnan()))
                self.assertTrue(torch.all(torch.fmod(x, zero).isnan()))

    # @unittest.skip("not test")
    @testinfo()
    def test_fmod_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = 20 * torch.randn(shape_list[i], dtype=torch.float)
            y = (20 * torch.rand(shape_list[i], dtype=torch.float)) + 1
            out = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            y_mlu = copy.deepcopy(y).to("mlu")
            out_mlu = copy.deepcopy(out).to("mlu")
            x, y, out = (
                x.permute(permute_shape[i]),
                y.permute(permute_shape[i]),
                out.permute(permute_shape[i]),
            )
            x_mlu, y_mlu, out_mlu = (
                x_mlu.permute(permute_shape[i]),
                y_mlu.permute(permute_shape[i]),
                out_mlu.permute(permute_shape[i]),
            )
            torch.fmod(x, y, out=out)
            torch.fmod(x_mlu, y_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_fmod_type(self):
        shape_list = [(1, 3, 16, 16)]
        type_list = [
            torch.double,
            torch.float,
            torch.half,
            torch.long,
            torch.int,
            torch.short,
            torch.bool,
        ]
        for shape in shape_list:
            for type in type_list:
                x_cpu = 20 * torch.randn(shape).to(type)
                y_cpu = (20 * torch.rand(shape).to(type)) + 1
                x_mlu = self.to_mlu(x_cpu)
                y_mlu = self.to_mlu(y_cpu)
                if type == torch.half:
                    x_cpu = x_cpu.float()
                    y_cpu = y_cpu.float()
                out_cpu = torch.fmod(x_cpu, y_cpu)
                out_mlu = torch.fmod(x_mlu, y_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_fmod_stride_channle_last_tensor(self):
        dtype_list = [(torch.half, 3e-3), (torch.int32, 3e-3), (torch.float, 3e-3)]

        shape_list = [(1, 128, 1, 64)]
        stride_list = [(128, 1, 8912, 128)]
        for dtype_err, in_shape, stride_shape in product(
            dtype_list, shape_list, stride_list
        ):
            dtype, err = dtype_err
            x_cpu = (20 * torch.randn(in_shape)).to(dtype)
            y_cpu = (20 * torch.rand(in_shape)).to(dtype) + 1
            x_mlu = x_cpu.to("mlu").to(memory_format=torch.channels_last)
            y_mlu = y_cpu.to("mlu").as_strided(in_shape, stride_shape)
            out_cpu = torch.fmod(
                x_cpu.to(memory_format=torch.channels_last),
                y_cpu.as_strided(in_shape, stride_shape),
            )
            out_mlu = torch.fmod(x_mlu, y_mlu)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_fmod_mix_device(self):
        x_cpu = torch.randn([])
        y_cpu = torch.randn(12, 13)
        out_cpu = torch.fmod(x_cpu, y_cpu)
        out_mlu = torch.fmod(x_cpu, y_cpu.to("mlu"))
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_fmod_bfloat16(self):
        parameter_list = [
            (torch.bfloat16, 3e-3, False),
            (torch.bfloat16, 3e-3, True),
        ]
        shape_list = [
            ((), ()),
            ((6, 0, 3), (6, 0, 1)),
            ((1, 3, 224, 224), (1, 3, 224, 1)),
            ((2, 2, 4, 2), (2)),
        ]
        for parameter, shape in product(parameter_list, shape_list):
            data_type, err, other_is_scalar = parameter
            shape1, shape2 = shape
            x = (20 * torch.randn(shape1, dtype=torch.float)).to(data_type)
            y = (
                (20 * torch.rand(shape2, dtype=torch.float)).to(data_type) + 1
                if other_is_scalar is False
                else 2.0
            )
            x_cpu = torch.nn.Parameter(x)
            y_cpu = y
            x_mlu = torch.nn.Parameter(x.mlu())
            y_mlu = y.mlu() if other_is_scalar is False else y
            out_cpu = torch.fmod(x_cpu, y_cpu)
            out_mlu = torch.fmod(x_mlu, y_mlu)
            grad = torch.randn(out_cpu.shape)
            grad_mlu = grad.mlu()
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )
            self.assertTensorsEqual(
                x_cpu.grad.float(), x_mlu.grad.cpu().float(), err, use_MSE=True
            )


if __name__ == "__main__":
    unittest.main()
