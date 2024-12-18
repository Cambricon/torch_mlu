from itertools import product
import copy
import unittest
import torch
import torch_mlu
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import TEST_LARGETENSOR, largeTensorTest


class TestFusedOptimizer(unittest.TestCase):
    def setUp(self, max_abs_diff=1e-3, max_rel_diff=1, iters=7):
        self.max_abs_diff = max_abs_diff
        self.max_rel_diff = max_rel_diff
        self.iters = iters
        torch.manual_seed(9876)
        os.environ["CNNL_ACC_SQRT"] = "1"
        os.environ["TORCH_MLU_SQRT_HIGH_PRECISION"] = "ON"

    def tearDown(self):
        pass

    def gen_param_optim(self, tensors, options):
        ref_param = []
        tst_param = []
        for tensor in tensors:
            ref_param.append(torch.nn.Parameter(tensor.clone()))
            tst_param.append(torch.nn.Parameter(tensor.clone()))
        options["fused"] = False
        options["foreach"] = False
        tst_options = copy.deepcopy(options)
        tst_options["fused"] = True
        ref_optim = torch.optim.AdamW(ref_param, **options)
        tst_optim = torch.optim.AdamW(tst_param, **tst_options)
        return (ref_param, tst_param, ref_optim, tst_optim)

    def gen_grad(self, ref_param, tst_param):
        for p_ref, p_tst in zip(ref_param, tst_param):
            p_ref.grad = torch.rand_like(p_ref)
            p_tst.grad = p_ref.grad

    def gen_mixed_grad(self, ref_param, tst_param, scale=1.0):
        half_grads = []
        for p_ref, p_tst in zip(ref_param, tst_param):
            half_grads.append(torch.rand_like(p_ref).half())
            p_ref.grad = half_grads[-1].float() / scale
        return half_grads

    def get_max_diff(self, ref_param, tst_param):
        max_abs_diff = max_rel_diff = 0
        for p_ref, p_tst in zip(ref_param, tst_param):
            max_abs_diff_p = (p_ref - p_tst).abs().max().item()
            max_rel_diff_p = ((p_ref - p_tst) / (p_ref + 3e-06)).abs().max().item()

            if max_abs_diff_p > max_abs_diff:
                max_abs_diff = max_abs_diff_p
            if max_rel_diff_p > max_rel_diff:
                max_rel_diff = max_rel_diff_p

        return max_abs_diff, max_rel_diff

    def gen_single_type_test(
        self,
        nelem=278011,
        param_type=torch.float,
        device="mlu",
        tensor_nums=1,
        *,
        skip_assert: bool = False
    ):
        # nelem = 20000 ok
        # Some ref and test optimizers may require different set of options.
        # This is a quick workaround to add that functionality while making
        # minimum changes in existing code.
        # If there is no "tst_options" field provided, safe to initialize
        # the test optimizer with the parameters of reference optimizer.
        for options in self.options_list:
            tensors = []
            for _ in range(tensor_nums):
                tensors.append(torch.rand(nelem).to(dtype=param_type, device=device))
            ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(
                tensors, options
            )

            for i in range(self.iters):
                self.gen_grad(ref_param, tst_param)
                ref_optim.step()
                tst_optim.step()
                if skip_assert:
                    return
                max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)
                self.assertLessEqual(max_abs_diff, self.max_abs_diff)
                self.assertLessEqual(max_rel_diff, self.max_rel_diff)
                torch.mlu.synchronize()


class TestFusedAdamWHighPrecison(TestFusedOptimizer):
    def setUp(self):
        super().setUp()
        self.options_list = [
            {
                "lr": 0.001,
                "betas": (0.6, 0.9),
                "eps": 3e-06,
                "weight_decay": 0.1,
                "amsgrad": False,
                "maximize": False,
            },
            {
                "lr": 0.01,
                "betas": (0.9, 0.95),
                "eps": 3e-06,
                "weight_decay": 0.12,
                "amsgrad": True,
                "maximize": False,
            },
            {
                "lr": 0.1,
                "betas": (0.9, 0.99),
                "eps": 3e-06,
                "weight_decay": 0,
                "amsgrad": True,
                "maximize": True,
            },
        ]

    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    # NOTE(mkozuki): Current threshold values look too small for BFloat16.
    # TODO(mkozuki): Refactor `TestFusedOptimizer`
    def test_half(self):
        self.gen_single_type_test(param_type=torch.float16, skip_assert=True)

    @unittest.skipIf(not torch.mlu.is_bf16_supported(), "BF16 is not supported")
    def test_bfloat16(self):
        self.gen_single_type_test(param_type=torch.bfloat16, skip_assert=True)

    @unittest.skipIf(torch.mlu.device_count() < 2, "more than 1 GPU required")
    def test_multi_device(self):
        devices = ("mlu:0", "mlu:1")
        for current_dev, tensor_dev in product(devices, devices):
            with torch.mlu.device(current_dev):
                self.gen_single_type_test(param_type=torch.float, device=tensor_dev)

    def test_adamw_one_element(self):
        self.gen_single_type_test(nelem=1, param_type=torch.float)


if __name__ == "__main__":
    unittest.main()
