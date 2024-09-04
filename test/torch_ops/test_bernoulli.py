# pylint:disable=R0912,W0703,W0632,W0212,R0201,W0612,W0702,C0411
from __future__ import print_function

import sys
import os
import copy
import unittest
import logging

from scipy import stats
from itertools import product
from functools import reduce
from operator import mul

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413, C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)
torch.manual_seed(2)


class TestBenoulliOp(TestCase):
    # The hypothesis cannot be rejected at the 5% level of significance
    # @unittest.skip("not test")
    @testinfo()
    def test_bernoulli(self):  # pylint: disable=R0201
        failure_count = 0
        for j in range(3):
            shape_list = [(2, 224, 224), (2048,), (2, 3, 224, 224), (1, 3, 5, 5)]
            pred_list = [0.0, 0.2, 0.5, 0.7, 1.0]
            memory_format_list = [torch.channels_last, torch.contiguous_format]
            param = [shape_list, pred_list, memory_format_list]

            for shape, pred, memory_format in product(*param):
                if len(shape) != 4 and memory_format == torch.channels_last:
                    continue

                a = torch.randn(shape, dtype=torch.float).to(
                    memory_format=memory_format
                )
                a_copy = copy.deepcopy(a)
                a_mlu = a_copy.to("mlu")
                cpu_data_ptr_orig = a.data_ptr()
                mlu_data_ptr_orig = a_mlu.data_ptr()
                a.bernoulli_(p=pred)
                a_mlu.bernoulli_(p=pred)
                self.assertEqual(cpu_data_ptr_orig, a.data_ptr())
                self.assertEqual(mlu_data_ptr_orig, a_mlu.data_ptr())
                # perform binomial test by sample proportion, refer to
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
                b = a_mlu.cpu()
                a_cpu = b.numpy()
                count = reduce(mul, shape)
                sucesses = int(a_cpu.sum())
                p_value = stats.binomtest(
                    sucesses, n=count, p=pred, alternative="less"
                ).pvalue
                self.assertTrue(torch.ne(b, 0).mul_(torch.ne(b, 1)).sum().item() == 0)
                if p_value < 0.005:
                    failure_count += 1
        self.assertTrue(failure_count < 3)

    # @unittest.skip("not test")
    @testinfo()
    def test_bernolli_exception(self):
        a = torch.randn(4).to("mlu")
        ref_msg = r"^bernoulli\_ expects p to be in \[0, 1\], but got p=2$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.bernoulli_(p=2)

    # @unittest.skip("not test")
    @testinfo()
    def test_bernoulli_channels_last(self):  # pylint: disable=R0201
        failure_count = 0
        for j in range(3):
            shape_list = [
                (2, 8, 16, 30),
            ]
            pred_list = [
                0.5,
            ]
            layout_list = ["channels_last", "not_dense"]
            param = [shape_list, pred_list, layout_list]
            for shape, pred, layout in product(*param):
                x = torch.randn(shape, dtype=torch.float)
                if layout == "channels_last":
                    a = x.to(memory_format=torch.channels_last)
                elif layout == "not_dense":
                    shape = (2, 8, 16, 20)
                    a = x[..., :20]
                a_copy = copy.deepcopy(a)
                a_mlu = a_copy.to("mlu")
                cpu_data_ptr_orig = a.data_ptr()
                mlu_data_ptr_orig = a_mlu.data_ptr()
                a.bernoulli_(p=pred)
                a_mlu.bernoulli_(p=pred)
                self.assertEqual(cpu_data_ptr_orig, a.data_ptr())
                self.assertEqual(mlu_data_ptr_orig, a_mlu.data_ptr())
                # perform binomial test by sample proportion, refer to
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
                b = a_mlu.cpu()
                a_cpu = b.numpy()
                count = reduce(mul, shape)
                sucesses = int(a_cpu.sum())
                p_value = stats.binomtest(
                    sucesses, n=count, p=pred, alternative="less"
                ).pvalue
                self.assertTrue(torch.ne(b, 0).mul_(torch.ne(b, 1)).sum().item() == 0)
                if p_value < 0.005:
                    failure_count += 1
        self.assertTrue(failure_count < 3)

    # @unittest.skip("not test")
    @testinfo()
    def test_bernoulli_tensor(self):  # pylint: disable=R0201
        failure_count = 0
        for j in range(3):
            shape_list = [(2, 224, 224), (2048,), (2, 3, 224, 224), (1, 3, 5, 5)]
            memory_format_list = [torch.channels_last, torch.contiguous_format]
            param = [shape_list, memory_format_list]

            for shape, memory_format in product(*param):
                if len(shape) != 4 and memory_format == torch.channels_last:
                    continue

                a = torch.empty(shape, dtype=torch.float).uniform_(0, 1)
                a = a.to(memory_format=memory_format)
                a_copy = copy.deepcopy(a)
                a_mlu = a_copy.to("mlu")
                cpu_data_ptr_orig = a.data_ptr()
                mlu_data_ptr_orig = a_mlu.data_ptr()
                cpu_output = torch.bernoulli(a)
                mlu_output = torch.bernoulli(a_mlu)
                self.assertEqual(cpu_data_ptr_orig, a.data_ptr())
                self.assertEqual(mlu_data_ptr_orig, a_mlu.data_ptr())
                # perform binomial test by sample proportion, refer to
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
                b = mlu_output.cpu()
                a_cpu = b.numpy()
                count = reduce(mul, shape)
                sucesses = int(a_cpu.sum())
                p_value = stats.binomtest(
                    sucesses, n=count, p=torch.mean(a).item(), alternative="less"
                ).pvalue
                self.assertTrue(torch.ne(b, 0).mul_(torch.ne(b, 1)).sum().item() == 0)
                if p_value < 0.005:
                    failure_count += 1
        self.assertTrue(failure_count < 3)

    # @unittest.skip("not test")
    @testinfo()
    def test_bernoulli_p(self):
        device = "mlu"
        for dtype in [torch.half, torch.float]:
            for trivial_p in ([0, 1], [1, 0, 1, 1, 0, 1]):
                x = torch.tensor(trivial_p, dtype=dtype, device=device)
                self.assertEqual(x.bernoulli().tolist(), trivial_p)

            def isBinary(t):
                return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

            p = torch.rand(5, 5, dtype=dtype, device=device)
            self.assertTrue(isBinary(p.bernoulli()))

            p = torch.rand(5, dtype=dtype, device=device).expand(5, 5)
            self.assertTrue(isBinary(p.bernoulli()))

            p = torch.rand(5, 5, dtype=dtype, device=device)
            torch.bernoulli(torch.rand_like(p), out=p)
            self.assertTrue(isBinary(p))

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_bernoulli_p_bfloat16(self):
        device = "mlu"
        for dtype in [torch.bfloat16]:
            for trivial_p in ([0, 1], [1, 0, 1, 1, 0, 1]):
                x = torch.tensor(trivial_p, dtype=dtype, device=device)
                self.assertEqual(x.bernoulli().tolist(), trivial_p)

            def isBinary(t):
                return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

            p = torch.rand(5, 5, dtype=dtype, device=device)
            self.assertTrue(isBinary(p.bernoulli()))

            p = torch.rand(5, dtype=dtype, device=device).expand(5, 5)
            self.assertTrue(isBinary(p.bernoulli()))

            p = torch.rand(5, 5, dtype=dtype, device=device)
            torch.bernoulli(torch.rand_like(p), out=p)
            self.assertTrue(isBinary(p))

    # @unittest.skip("not test")
    @testinfo()
    def test_bernoulli_self(self):
        def isBinary(t):
            return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

        device = "mlu"
        for dtype in [torch.half, torch.float]:
            t = torch.empty(10, 10, dtype=dtype, device=device)

            t.fill_(2)
            t.bernoulli_(0.5)
            self.assertTrue(isBinary(t))
            p = torch.rand(10, dtype=dtype, device=device).expand(10, 10)
            t.fill_(2)
            t.bernoulli_(p)
            self.assertTrue(isBinary(t))

            # test different dtypes of t and p
            t.bernoulli_(p.to(torch.half))

            t.fill_(2)
            torch.bernoulli(torch.rand_like(t, dtype=dtype), out=t)
            self.assertTrue(isBinary(t))

            t.fill_(2)
            t.bernoulli_(torch.rand_like(t, dtype=dtype))
            self.assertTrue(isBinary(t))

    # @unittest.skip("not test")
    @testinfo()
    def test_bernoulli_edge_cases(self):
        device = "mlu"
        for dtype in [torch.half, torch.float]:
            # Need to draw a lot of samples to cover every random floating point number.
            # probability of drawing "1" is 0
            a = torch.zeros(10000, 10000, dtype=dtype, device=device)
            num_ones = (torch.bernoulli(a) == 1).sum()
            self.assertEqual(num_ones, 0)

            # probability of drawing "1" is 1
            b = torch.ones(10000, 10000, dtype=dtype, device=device)
            num_zeros = (torch.bernoulli(b) == 0).sum()
            self.assertEqual(num_zeros, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_bernoulli_p_broadcast(self):
        device = "mlu"
        a = (
            torch.rand((12, 3, 226, 226), dtype=torch.float)
            .to(device)
            .to(memory_format=torch.channels_last)
        )
        t = torch.rand((3, 1, 226)).to(device)
        a.bernoulli_(t)

    # @unittest.skip("not test")
    @testinfo()
    def test_bernolli_generator_exception(self):
        a = torch.randn(4).to("mlu")
        g = torch.Generator()
        ref_msg = r"Expected a \'mlu\' device type for generator but found \'cpu\'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.bernoulli(a, generator=g)


if __name__ == "__main__":
    unittest.main()
