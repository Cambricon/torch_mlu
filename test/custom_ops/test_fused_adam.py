import logging
import sys
import os
import unittest
from itertools import product

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase, TEST_BFLOAT16


class TestFusedAdam(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFusedAdam, self).__init__(*args, **kwargs)
        self.options = {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }
        self.iters = 10
        self.ori_optim = torch.optim.Adam
        self.fused_optim = torch_mlu.optimizers.FusedAdam

    def fused_adam_dtype(self, test_type=torch.float, shape=[40, 40], len=2, err=1e-3):
        tensors_native = []
        tensors_fused = []
        tensors_golden = []
        fused_opt_params = []
        golden_opt_params = []
        opt_params = []

        # clamp and addcmul_ do not have the impl of half
        test_type = torch.float if test_type == torch.half else test_type
        for i in range(len):
            tensor_t = torch.clamp(
                torch.rand(shape, dtype=test_type), min=0.01, max=100.0
            )
            tensors_native.append(tensor_t)
            tensors_fused.append(tensor_t.to(test_type).to("mlu"))
            tensors_golden.append(tensor_t.to(torch.double))
        for tensor_native, tensor_fused, tensor_golden in zip(
            tensors_native, tensors_fused, tensors_golden
        ):
            opt_params.append(torch.nn.Parameter(tensor_native.clone()))
            fused_opt_params.append(torch.nn.Parameter(tensor_fused.clone()))
            golden_opt_params.append(torch.nn.Parameter(tensor_golden.clone()))
        fused_optimizer = torch_mlu.optimizers.FusedAdam(
            fused_opt_params, **self.options
        )
        optimizer = torch.optim.Adam(opt_params, **self.options)
        golden_optimizer = torch.optim.Adam(golden_opt_params, **self.options)

        for _ in range(self.iters):
            for p_opt, p_fused_opt, p_golden_opt in zip(
                opt_params, fused_opt_params, golden_opt_params
            ):
                p_opt.grad = torch.rand_like(p_opt)
                p_fused_opt.grad = p_opt.grad.to(test_type).to("mlu")
                p_golden_opt.grad = p_opt.grad.to(torch.double)
            optimizer.step()
            fused_optimizer.step()
            golden_optimizer.step()
            for p_ref, p_tst, p_golden in zip(
                opt_params, fused_opt_params, golden_opt_params
            ):
                if p_ref.numel() == 0 and p_tst.numel() == 0:
                    continue
                self.assertTensorsEqual(
                    p_ref.cpu().float(), p_tst.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_fused_adam(self):
        shape_list = [
            (0),
            (1),
            (15, 20),
            (2, 3, 4),
            (8, 1, 2, 3),
            (2, 1, 2, 1, 4),
            (64, 16, 32),
        ]
        dtype_err_list = [(torch.float, 1e-3), (torch.half, 1e-2), (torch.double, 1e-3)]
        wd_list = [0, 0.01]
        len_list = [2, 10, 40]
        loop_var = [shape_list, dtype_err_list, wd_list, len_list]
        for shape, dtype_err, wd, len in product(*loop_var):
            dtype, err = dtype_err
            self.options["weight_decay"] = wd
            self.fused_adam_dtype(test_type=dtype, shape=shape, len=len, err=err)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_fused_adam_bf16(self):
        shape_list = [
            (0),
            (1),
            (15, 20),
            (2, 3, 4),
            (8, 1, 2, 3),
            (2, 1, 2, 1, 4),
            (64, 16, 32),
        ]
        # TODO (dengshun): 1e-2 is too small for BFloat16.
        dtype_err_list = [(torch.bfloat16, 2e-2)]
        wd_list = [0, 0.01]
        len_list = [2, 10, 40]
        loop_var = [shape_list, dtype_err_list, wd_list, len_list]
        for shape, dtype_err, wd, len in product(*loop_var):
            dtype, err = dtype_err
            self.options["weight_decay"] = wd
            self.fused_adam_dtype(test_type=dtype, shape=shape, len=len, err=err)


if __name__ == "__main__":
    unittest.main()
