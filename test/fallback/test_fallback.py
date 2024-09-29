from __future__ import print_function

import logging
import unittest
import sys
import os
import subprocess

os.environ["ENABLE_FALLBACK_TO_CPU"] = "1"

import torch
import torch_mlu  # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestFallback(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_ops_not_implemented_with_fallback_on(self):
        # remove when softshrink is supported
        input = torch.randn(2, 3, 4, 5, device="cpu")
        input_cpu = torch.nn.Parameter(input)
        input_mlu = torch.nn.Parameter(input.mlu())
        output_cpu = torch.nn.functional.softshrink(input_cpu, 0.5)
        output_mlu = torch.nn.functional.softshrink(input_mlu, 0.5)
        grad = torch.randn(output_cpu.shape, device="cpu")
        grad_mlu = grad.mlu()
        output_cpu.backward(grad)
        output_mlu.backward(grad_mlu)
        self.assertEqual(input_cpu.grad, input_mlu.grad.cpu())
        self.assertEqual(output_cpu, output_mlu.cpu())

        # remove when nanmedian is supported
        input_cpu = torch.randn(31)
        value, indices = input_cpu.nanmedian(0, True)
        input_mlu = input_cpu.mlu()
        value_mlu, indices_mlu = input_mlu.nanmedian(0, True)
        self.assertEqual(value, value_mlu.cpu())
        self.assertEqual(indices, indices_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_ops_not_implemented_with_fallback_off(self):
        script = f"""
import os
os.environ["ENABLE_FALLBACK_TO_CPU"] = "0"

import torch
import torch_mlu

input_mlu = torch.randn(31).mlu()
value_mlu, indices_mlu = input_mlu.nanmedian(0, True)

"""

        try:
            subprocess.check_output(
                [sys.executable, "-W", "all", "-c", script],
                stderr=subprocess.STDOUT,
                # Opening the subprocess with the default CWD makes `import torch` or `import torch_mlu`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            msg = f"The operator 'aten::nanmedian.dim_values' is not currently implemented for the MLU device"
            self.assertTrue(msg in e.stdout.decode("utf-8"))

    # @unittest.skip("not test")
    @testinfo()
    def test_ops_fail_on_mlu_with_fallback_on(self):
        script = f"""
import os
os.environ["ENABLE_MLU_FAIL_FALLBACK"] = "1"

import torch
import torch.nn.functional as F
import torch_mlu

m = torch.nn.ReflectionPad1d((1, 1))
x_cpu = torch.randn((2, 3, 4), dtype=torch.cfloat)
x_mlu = x_cpu.mlu()
output_cpu = m(x_cpu)
output_mlu = m(x_mlu)
assert torch.equal(output_mlu.cpu(), output_cpu)

x_cpu = torch.tensor(True).cfloat()
index_cpu = torch.tensor([0])
x_mlu = x_cpu.mlu()
index_mlu = index_cpu.mlu()
output_cpu = x_cpu.unsqueeze(0)[index_cpu]
output_mlu = x_mlu.unsqueeze(0)[index_mlu]
assert torch.equal(output_mlu.cpu(), output_cpu)

"""

        subprocess.check_output(
            [sys.executable, "-W", "all", "-c", script],
            stderr=subprocess.STDOUT,
            # Opening the subprocess with the default CWD makes `import torch` or `import torch_mlu`
            # fail, so just set CWD to this script's directory
            cwd=os.path.dirname(os.path.realpath(__file__)),
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_ops_fail_on_mlu_with_fallback_off(self):
        script = f"""
import os
os.environ["ENABLE_MLU_FAIL_FALLBACK"] = "0"

import torch
import torch_mlu

m = torch.nn.ReflectionPad1d((1, 1))
x_mlu = torch.randn((2, 3, 4), dtype=torch.cfloat).mlu()
output_mlu = m(x_mlu)

"""

        try:
            subprocess.check_output(
                [sys.executable, "-W", "all", "-c", script],
                stderr=subprocess.STDOUT,
                # Opening the subprocess with the default CWD makes `import torch` or `import torch_mlu`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            msg = f"not implemented for 'ComplexFloat'"
            self.assertTrue(msg in e.stdout.decode("utf-8"))

    # @unittest.skip("not test")
    @testinfo()
    def test_ops_no_fallback_implementation(self):
        # warp_ctc_loss has no fallback implementation
        T = 50
        C = 20
        N = 16
        S = 30
        S_min = 10
        probs = torch.randn((T, N, C), requires_grad=False)
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
        target = torch.randint(
            low=1, high=C, size=(sum(target_lengths),), dtype=torch.long
        )
        with self.assertRaisesRegex(
            RuntimeError, "warp_ctc_loss only support sum mode"
        ):
            torch.ops.torch_mlu.warp_ctc_loss(
                probs.mlu(),
                target.mlu(),
                input_lengths.mlu(),
                target_lengths.mlu(),
                0,
                2,
                True,
                0,
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_sparse_ops_not_implemented_with_fallback_on(self):
        # remove when sparse_sparse_matmul is supported
        def gen_sparse(sparse_dim, nnz, size, dtype):
            v_size = [nnz] + list(size[sparse_dim:])
            v = torch.randn(v_size)
            i = torch.rand(sparse_dim, nnz)
            i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
            i = i.to(torch.long)
            return torch.sparse_coo_tensor(i, v, torch.Size(size), dtype=dtype)

        m1 = gen_sparse(2, 10, (5, 6), torch.float)
        m2 = gen_sparse(2, 5, (6, 3), torch.float)
        output_cpu = torch.matmul(m1, m2)
        output_mlu = torch.matmul(m1.to("mlu"), m2.to("mlu"))
        self.assertEqual(output_cpu, output_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_sparse_ops_not_implemented_with_fallback_off(self):
        script = f"""
import os
os.environ["ENABLE_FALLBACK_TO_CPU"] = "0"

import torch
import torch_mlu

def gen_sparse(sparse_dim, nnz, size, dtype):
    v_size = [nnz] + list(size[sparse_dim:])
    v = torch.randn(v_size)
    i = torch.rand(sparse_dim, nnz)
    i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
    i = i.to(torch.long)
    return torch.sparse_coo_tensor(i, v, torch.Size(size), dtype=dtype)

m1 = gen_sparse(2, 10, (5, 6), torch.float)
m2 = gen_sparse(2, 5, (6, 3), torch.float)
output_mlu = torch.matmul(m1.to('mlu'), m2.to('mlu'))
"""
        try:
            subprocess.check_output(
                [sys.executable, "-W", "all", "-c", script],
                stderr=subprocess.STDOUT,
                # Opening the subprocess with the default CWD makes `import torch` or `import torch_mlu`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
            self.assertTrue(
                False,
                "Unimplemented sparse ops should not pass when fallback is turned off.",
            )
        except subprocess.CalledProcessError as e:
            msg = f"The operator 'aten::_sparse_sparse_matmul' is not currently implemented for the MLU device."
            self.assertTrue(msg in e.stdout.decode("utf-8"))

    # @unittest.skip("not test")
    @testinfo()
    def test_sparse_ops_fail_on_mlu_with_fallback_on(self):
        script = f"""
import os
os.environ["ENABLE_MLU_FAIL_FALLBACK"] = "1"

import torch
import torch_mlu

dims = (5, 5, 2, 2)
def gen_sparse(nnz):
    i = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
                   torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
    v = torch.randn(nnz, dims[2], dims[3])
    return torch.sparse_coo_tensor(i, v, dims).coalesce()

s = gen_sparse(15)
mask = gen_sparse(5)
s_mlu = s.to('mlu')
mask_mlu = mask.to('mlu')

output_cpu = s.sparse_mask(mask)
output_mlu = s_mlu.sparse_mask(mask_mlu)
not_equal = torch.any(output_cpu - output_mlu.cpu())
if not_equal:
    raise ValueError("The cpu result and fallback result are not equal.")

"""
        subprocess.check_output(
            [sys.executable, "-W", "all", "-c", script],
            stderr=subprocess.STDOUT,
            # Opening the subprocess with the default CWD makes `import torch` or `import torch_mlu`
            # fail, so just set CWD to this script's directory
            cwd=os.path.dirname(os.path.realpath(__file__)),
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_sparse_ops_fail_on_mlu_with_fallback_off(self):
        script = f"""
import os
os.environ["ENABLE_MLU_FAIL_FALLBACK"] = "0"

import torch
import torch_mlu

dims = (5, 5, 2, 2)
def gen_sparse(nnz):
    i = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
                   torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
    v = torch.randn(nnz, dims[2], dims[3])
    return torch.sparse_coo_tensor(i, v, dims).coalesce()

s = gen_sparse(15)
mask = gen_sparse(5)

output_mlu = s.to('mlu').sparse_mask(mask.to('mlu'))
"""
        try:
            subprocess.check_output(
                [sys.executable, "-W", "all", "-c", script],
                stderr=subprocess.STDOUT,
                # Opening the subprocess with the default CWD makes `import torch` or `import torch_mlu`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
            self.assertTrue(
                False,
                "Failed sparse ops should not pass when fail_fallback is turned off.",
            )
        except subprocess.CalledProcessError as e:
            msg = f"DispatchStub: missing PrivateUse1 kernel"
            self.assertTrue(msg in e.stdout.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
