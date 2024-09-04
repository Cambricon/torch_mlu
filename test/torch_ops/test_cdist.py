from __future__ import print_function

import sys
import logging
import os
import copy

import unittest
from itertools import product
import torch
import torch_mlu  # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    read_card_info,
    skipBFloat16IfNotSupport,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestCdistOp(TestCase):
    def TensorGenerator(self, shape, dtype, func=lambda x: x):
        if dtype.is_floating_point:
            cpu_tensor = torch.randn(shape).to(torch.half).to(torch.float)
            mlu_tensor = func(cpu_tensor.to("mlu").to(dtype))
            cpu_tensor = func(cpu_tensor)
            return cpu_tensor, mlu_tensor
        elif dtype.is_complex:
            cpu_tensor = torch.randn(shape, dtype=dtype)
            mlu_tensor = func(cpu_tensor.to("mlu"))
            cpu_tensor = func(cpu_tensor)
            return cpu_tensor, mlu_tensor
        elif dtype == torch.bool:
            cpu_tensor = torch.randint(0, 2, shape, dtype=dtype)
            mlu_tensor = func(cpu_tensor.to("mlu"))
            cpu_tensor = func(cpu_tensor)
            return cpu_tensor, mlu_tensor
        else:
            cpu_tensor = torch.randint(100, shape, dtype=dtype)
            mlu_tensor = func(cpu_tensor.to("mlu"))
            cpu_tensor = func(cpu_tensor)
            return cpu_tensor, mlu_tensor

    # @unittest.skip("not test")
    @testinfo()
    def test_cdist_same_inputs(self):  # pylint: disable=R0201
        # Test to detect issues in cdist gradient calculation
        # When the distances are 0
        sizex = (1, 27, 32)
        # other p values will be supported in the future
        for p in [1]:
            device = "mlu"
            x = torch.randn(sizex, device=device, dtype=torch.float)
            dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.float)
            y = x.clone()
            x.requires_grad = True
            d = torch.cdist(x, y, p=p)
            d.backward(dist_grad)
            # Check that the backward passs does not contain invalid
            # values such as nan or inf
            assert torch.isfinite(x.grad).all()

    # @unittest.skip("not test")
    @testinfo()
    def test_cdist_empty(self):
        device = "mlu"
        x = torch.randn((0, 5), device=device)
        y = torch.randn((4, 5), device=device)
        self.assertEqual(torch.empty(0, 4, device=device), torch.cdist(x, y, p=1.0))

        x = torch.randn((2, 5), device=device)
        y = torch.randn((0, 5), device=device)
        self.assertEqual(torch.empty(2, 0, device=device), torch.cdist(x, y, p=1.0))

        x = torch.randn((2, 0), device=device)
        y = torch.randn((3, 0), device=device)
        self.assertEqual(torch.zeros(2, 3, device=device), torch.cdist(x, y, p=1.0))

        x = torch.randn((2, 0), device=device)
        y = torch.randn((0, 0), device=device)
        self.assertEqual(torch.empty(2, 0, device=device), torch.cdist(x, y, p=1.0))

    def _brute_cdist(self, x, y, p=1):  # pylint: disable=R0201
        r1 = x.shape[-2]
        r2 = y.shape[-2]
        if r1 == 0 or r2 == 0:
            return torch.empty(r1, r2, device=x.device)
        return torch.norm(x[..., None, :] - y[..., None, :, :], p=p, dim=-1)

    # @unittest.skip("not test")
    @testinfo()
    def test_cdist_norm(self):
        device = "mlu"
        for r1 in [3, 4, 5, 6]:
            for m in [2, 3, 4, 10]:
                for r2 in [4, 6, 7, 8]:
                    # other p values will be supported in the future
                    for p in [1]:
                        x = torch.randn(r1, m, device=device)
                        y = torch.randn(r2, m, device=device)
                        if p == 2:
                            for cm in [
                                "use_mm_for_euclid_dist",
                                "donot_use_mm_for_euclid_dist",
                            ]:
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                expected = self._brute_cdist(x, y, p=2)
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)

    # @unittest.skip("not test")
    @testinfo()
    def test_cdist_norm_batch(self):
        device = "mlu"
        for r1 in [3, 4, 5, 6]:
            for m in [2, 3, 4, 10]:
                for r2 in [4, 6, 7, 8]:
                    # other p values will be supported in the future
                    for p in [1]:
                        x = torch.randn(2, 3, 6, r1, m, device=device)
                        y = torch.randn(2, 3, 6, r2, m, device=device)
                        if p == 2:
                            for cm in [
                                "use_mm_for_euclid_dist",
                                "donot_use_mm_for_euclid_dist",
                            ]:
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                expected = self._brute_cdist(x, y, p=2)
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)

    # @unittest.skip("not test")
    @testinfo()
    def test_cdist_mlu_backward(self):
        device = "mlu"
        for l1 in [1, 511, 513]:
            for l2 in [1, 511, 513]:
                # other p values will be supported in the future
                for p in [1]:
                    x1 = torch.randn(4, l1, 32, device=device, requires_grad=True)
                    x2 = x1.clone().detach_().requires_grad_()
                    y1 = torch.randn(4, l2, 32, device=device, requires_grad=True)
                    y2 = y1.clone().detach_().requires_grad_()
                    if p == 2:
                        for cm in [
                            "use_mm_for_euclid_dist",
                            "donot_use_mm_for_euclid_dist",
                        ]:
                            z1 = torch.cdist(x1, y1, p=2, compute_mode=cm).mean()
                            z2 = self._brute_cdist(x2, y2, p=2).mean()
                            z1.backward()
                            z2.backward()
                            self.assertEqual(x1.grad, x2.grad, rtol=0, atol=0.001)
                            self.assertEqual(y1.grad, y2.grad, rtol=0, atol=0.001)
                    else:
                        z1 = torch.cdist(x1, y1, p=p).mean()
                        z2 = self._brute_cdist(x2, y2, p=p).mean()
                        self.assertEqual(x1.grad, x2.grad, rtol=0, atol=0.001)
                        self.assertEqual(y1.grad, y2.grad, rtol=0, atol=0.001)

    # @unittest.skip("not test")
    @testinfo()
    def test_cdist(self):
        shape_list = [(2, 3), (8, 224, 224), (1, 3, 16, 16), (128, 128, 1, 8, 3)]
        # other p values will be supported in the future
        p_list = [1]
        type_list = [torch.float, torch.double]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape, type, p, func in product(shape_list, type_list, p_list, func_list):
            cpu_x, mlu_x = self.TensorGenerator(shape, type, func)
            cpu_y, mlu_y = self.TensorGenerator(shape, type, func)
            cpu_result = torch.cdist(cpu_x, cpu_y, p=p)
            mlu_result = torch.cdist(mlu_x, mlu_y, p=p)
            self.assertTensorsEqual(cpu_result, mlu_result.cpu(), 3e-3, use_MSE=True)

            # backward needs contiguous input
            cpu_x = torch.randn(shape, dtype=type)
            x_t = copy.deepcopy(cpu_x)
            mlu_x = x_t.mlu()
            cpu_x.requires_grad = True
            mlu_x.requires_grad = True
            cpu_y = torch.randn(shape, dtype=type)
            y_t = copy.deepcopy(cpu_y)
            mlu_y = y_t.mlu()
            cpu_y.requires_grad = True
            mlu_y.requires_grad = True
            cpu_result = torch.cdist(cpu_x, cpu_y, p=p)
            mlu_result = torch.cdist(mlu_x, mlu_y, p=p)
            grad_cpu = torch.randn(cpu_result.shape, dtype=type)
            grad_mlu = grad_cpu.mlu()
            cpu_result.backward(grad_cpu)
            mlu_result.backward(grad_mlu)
            self.assertTensorsEqual(cpu_x.grad, mlu_x.grad.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(cpu_y.grad, mlu_y.grad.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cdist_exception(self):
        ref_msg = r"cnnl_cdist does not support half input for now, X1 got: Half"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x = torch.randn((2, 3, 4)).mlu().half()
            y = torch.randn((2, 3, 4)).mlu().half()
            torch.cdist(x, y, p=1)

        ref_msg = r"cnnl_cdist only supports p = 1.0 for now"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x = torch.randn((2, 3, 4)).mlu()
            y = torch.randn((2, 3, 4)).mlu()
            torch.cdist(x, y, p=2)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_cdist_bfloat16(self):
        # Test to detect issues in cdist gradient calculation
        # When the distances are 0
        sizex = (1, 27, 32)
        # other p values will be supported in the future
        device = "mlu"
        x = torch.randn(sizex, device=device, dtype=torch.bfloat16)
        dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.bfloat16)
        y = x.clone()
        x.requires_grad = True
        d = torch.cdist(x, y, p=1)
        d.backward(dist_grad)
        # Check that the backward passs does not contain invalid
        # values such as nan or inf
        assert torch.isfinite(x.grad).all()

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("53GB")
    def test_cdist_large(self):
        shape = (4, 1025, 1024, 1024)
        dtype = torch.float
        cpu_x = torch.randn(shape, dtype=dtype)
        x_t = copy.deepcopy(cpu_x)
        mlu_x = x_t.mlu()
        cpu_y = torch.randn(shape, dtype=dtype)
        y_t = copy.deepcopy(cpu_y)
        mlu_y = y_t.mlu()
        cpu_result = torch.cdist(cpu_x, cpu_y, p=1)
        mlu_result = torch.cdist(mlu_x, mlu_y, p=1)
        self.assertTensorsEqual(cpu_result, mlu_result.cpu(), 3e-3, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("53GB")
    def test_cdist_backward_large(self):
        # [CNNL] [Error]:CDIST_BACKWARD_API overflow max supported tensor num 2147483647,
        # now tensor's total num is 4299161600.
        ref_msg = r"CNNL error: CNNL_STATUS_NOT_SUPPORTED"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            shape1 = (4, 1024, 1025, 1024)
            shape2 = (4, 1024, 1, 1024)
            cpu_x = torch.randn(shape1, requires_grad=True)
            cpu_y = torch.randn(shape2, requires_grad=True)
            cpu_result = torch.cdist(cpu_x, cpu_y, p=1)
            mlu_result = torch.cdist(cpu_x.to("mlu"), cpu_y.to("mlu"), p=1)
            grad_cpu = torch.randn(cpu_result.shape)
            cpu_result.backward(grad_cpu)
            grad_x_cpu = copy.deepcopy(cpu_x.grad)
            cpu_x.grad.zero_()
            cpu_y.grad.zero_()
            mlu_result.backward(grad_cpu.to("mlu"))
            grad_x_mlu = copy.deepcopy(cpu_x.grad)
            self.assertTensorsEqual(grad_x_cpu, grad_x_mlu.cpu(), 3e-3, use_MSE=True)


if __name__ == "__main__":
    run_tests()
