import sys
import os
import unittest
import logging

from itertools import product
import numpy as np

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)

logging.basicConfig(level=logging.DEBUG)
torch.manual_seed(2)


class TestMultinomialOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_multinomial1D(self):  # pylint: disable=R0201
        failure_count = 0
        for j in range(3):
            shape_list = [20, 10, 5]
            replacement = [True, False]
            param = [shape_list, replacement]
            for shape, replacement in product(*param):
                a = torch.rand(shape, dtype=torch.float).to("mlu")
                n_samples = np.random.randint(1, shape)
                b = torch.multinomial(a, n_samples, replacement)

    # @unittest.skip("not test")
    @testinfo()
    def test_multinomial2D(self):  # pylint: disable=R0201
        failure_count = 0
        for j in range(3):
            shape_list = [(40, 200), (1, 50048)]
            replacement = [True, False]
            param = [shape_list, replacement]
            for shape, replacement in product(*param):
                a = torch.rand(shape, dtype=torch.float).to("mlu")
                n_samples = np.random.randint(1, shape[-1])
                b = torch.multinomial(a, 1, replacement)

    # @unittest.skip("not test")
    @testinfo()
    def test_multinomial1D_out(self):  # pylint: disable=R0201
        failure_count = 0
        for j in range(3):
            shape_list = [20, 10, 5]
            replacement = [True, False]
            param = [shape_list, replacement]
            for shape, replacement in product(*param):
                a = torch.rand(shape, dtype=torch.float).to("mlu")
                output = torch.rand(1).long().to("mlu")
                n_samples = np.random.randint(1, shape)
                b = torch.multinomial(a, n_samples, replacement, out=output)

    # @unittest.skip("not test")
    @testinfo()
    def test_multinomial2D_not_dense(self):  # pylint: disable=R0201
        failure_count = 0
        for j in range(3):
            shape_list = [
                (40, 200),
            ]
            replacement = [True, False]
            param = [shape_list, replacement]
            for shape, replacement in product(*param):
                a = torch.rand(shape, dtype=torch.float).to("mlu")[..., :2]
                n_samples = np.random.randint(1, shape[-1])
                b = torch.multinomial(a, 1, replacement)

    # @unittest.skip("not test")
    @testinfo()
    def test_multinomial_constraints(self):
        device = "mlu"
        x = torch.empty(1, 2, 3, dtype=torch.double, device=device)
        self.assertRaisesRegex(
            RuntimeError,
            "prob_dist must be 1 or 2 dim",
            lambda: torch.multinomial(x, 2),
        )
        x = torch.empty(1, 2, dtype=torch.long, device=device)
        self.assertRaisesRegex(
            RuntimeError,
            "multinomial only supports floating-point dtypes for input",
            lambda: torch.multinomial(x, 2),
        )
        x = torch.empty(1, 2, dtype=torch.double, device=device)
        y = torch.empty(1, 2, dtype=torch.double, device=device)
        self.assertRaisesRegex(
            RuntimeError,
            "multinomial expects Long tensor out",
            lambda: torch.multinomial(x, 2, out=y),
        )
        x = torch.empty(2, dtype=torch.double, device=device)
        self.assertRaisesRegex(
            RuntimeError,
            "cannot sample n_sample <= 0 samples",
            lambda: torch.multinomial(x, 0),
        )
        x = torch.empty(2, dtype=torch.double, device=device)
        self.assertRaisesRegex(
            RuntimeError,
            "cannot sample n_sample <= 0 samples",
            lambda: torch.multinomial(x, -1),
        )
        x = torch.empty(2, dtype=torch.double, device=device)
        self.assertRaisesRegex(
            RuntimeError,
            "cannot sample n_sample > prob_dist",
            lambda: torch.multinomial(x, 3, False),
        )
        x = torch.empty(16777217, dtype=torch.double, device=device)
        self.assertRaisesRegex(
            RuntimeError,
            "number of categories cannot exceed",
            lambda: torch.multinomial(x, 3),
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("46GB")
    def test_multinomial2D_large(self):  # pylint: disable=R0201
        for j in range(1):
            n_samples = 10
            shape_list = [(320, 16777216)]
            replacement = [True]
            param = [shape_list, replacement]
            for shape, replacement in product(*param):
                print("replacement: ", replacement)
                a = torch.rand(shape, dtype=torch.float).to("mlu")
                b = torch.multinomial(a, n_samples, replacement)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_multinomial2D_bfloat16(self):  # pylint: disable=R0201
        failure_count = 0
        for j in range(3):
            shape_list = [(40, 200), (1, 50048)]
            replacement = [True, False]
            param = [shape_list, replacement]
            for shape, replacement in product(*param):
                a = torch.rand(shape, dtype=torch.bfloat16).to("mlu")
                n_samples = np.random.randint(1, shape[-1])
                b = torch.multinomial(a, 1, replacement)


if __name__ == "__main__":
    run_tests()
