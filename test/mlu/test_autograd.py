from __future__ import print_function

import unittest
import logging
import pickle
from itertools import chain
import collections
import os
import sys
import torch
from typing import Optional

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import TestCase  # pylint: disable=C0411

logging.basicConfig(level=logging.DEBUG)


class TestAutoGrad(TestCase):
    # @unittest.skip("not test")
    def test_graph_save_on_cpu_mlu(self):
        def f(x):
            a = x + 1
            return a * a

        # with grad
        a = torch.ones(1, requires_grad=True, device="mlu")
        y = f(a)
        memory_with_grad = torch.mlu.memory_allocated()

        del a
        del y

        # without grad
        a = torch.ones(1, requires_grad=True, device="mlu")
        with torch.no_grad():
            y = f(a)
        memory_without_grad = torch.mlu.memory_allocated()

        self.assertGreater(memory_with_grad, memory_without_grad)

        del a
        del y

        # with hooks
        with torch.autograd.graph.save_on_cpu():
            a = torch.ones(1, requires_grad=True, device="mlu")
            y = f(a)
            memory_with_hooks = torch.mlu.memory_allocated()
            self.assertEqual(memory_with_hooks, memory_without_grad)

        # with hooks
        with torch.autograd.graph.save_on_cpu(True):
            a = torch.ones(1, requires_grad=True, device="mlu")
            y = f(a)
            memory_with_hooks = torch.mlu.memory_allocated()
            self.assertEqual(memory_with_hooks, memory_without_grad)


if __name__ == "__main__":
    unittest.main()
