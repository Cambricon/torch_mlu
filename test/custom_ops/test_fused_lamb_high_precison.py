from __future__ import print_function
import logging
import sys
import os
from itertools import product
import unittest

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411


class RefLAMB(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        amsgrad=False,
        adam_w_mode=True,
        grad_averaging=True,
        set_grad_none=True,
        max_grad_norm=1.0,
        use_nvlamb=False,
    ):
        if amsgrad:
            raise RuntimeError("FusedLAMB does not support the AMSGrad variant.")
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            max_grad_norm=max_grad_norm,
        )
        super(RefLAMB, self).__init__(params, defaults)
        # Skip buffer
        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int).to(
            self.param_groups[0]["params"][0].device
        )

        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        self.use_nvlamb = use_nvlamb

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad = None
        else:
            super(RefLAMB, self).zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # create separate grad lists for fp32 and fp16 params
        g_all_32, g_all_16 = [], []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dtype == torch.float32:
                    g_all_32.append(p.grad.data)
                elif p.dtype == torch.float16:
                    g_all_16.append(p.grad.data)
                else:
                    raise RuntimeError("FusedLAMB only support fp16 and fp32.")

        device = self.param_groups[0]["params"][0].device
        g_norm_32, g_norm_16 = torch.zeros(1, device=device), torch.zeros(
            1, device=device
        )
        # compute grad norm for two lists
        if len(g_all_32) > 0:
            g_norm_32 = torch.ops.torch_mlu.fused_l2_norm(
                self._dummy_overflow_buf, g_all_32, False
            )[0]
        if len(g_all_16) > 0:
            g_norm_16 = torch.ops.torch_mlu.fused_l2_norm(
                self._dummy_overflow_buf, g_all_16, False
            )[0]

        # blend two grad norms to get global grad norm
        global_grad_norm = torch.ops.torch_mlu.fused_l2_norm(
            self._dummy_overflow_buf, [g_norm_32, g_norm_16], False
        )[0]
        max_grad_norm = self.defaults["max_grad_norm"]
        clipped_ratio = max_grad_norm / max(global_grad_norm, max_grad_norm)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.data *= clipped_ratio
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lamb does not support sparse gradients, consider SparseAdam instad."
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["m"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["v"] = torch.zeros_like(p.data)

                m_t, v_t = state["m"], state["v"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # m_t = beta1 * m + (1 - beta1) * g_t
                m_t.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
                v_t.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Debiasing
                m_t_hat = m_t / (1.0 - beta1 ** state["step"])
                v_t_hat = v_t / (1.0 - beta2 ** state["step"])

                update = m_t_hat / v_t_hat.sqrt().add(group["eps"])

                if group["weight_decay"] != 0:
                    update.add_(p.data, alpha=group["weight_decay"])

                trust_ratio = 1.0
                w_norm = p.data.pow(2).sum().sqrt()
                g_norm = update.pow(2).sum().sqrt()
                if w_norm > 0 and g_norm > 0:
                    trust_ratio = w_norm / g_norm

                state["w_norm"] = w_norm
                state["g_norm"] = g_norm
                state["trust_ratio"] = trust_ratio

                step_size = group["lr"]

                p.data.add_(update, alpha=-step_size * trust_ratio)

        return loss


class TestFusedLAMBHighPrecision(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFusedLAMBHighPrecision, self).__init__(*args, **kwargs)
        self.options = {"lr": 0.01, "betas": (0.6, 0.9), "eps": 3e-06}
        self.fused_optimizer = torch_mlu.optimizers.FusedLAMB
        self.origin_optimizer = RefLAMB
        self.iters = 10
        self.max_abs_diff = 1e-3
        self.max_rel_diff = 1
        os.environ["CNNL_ACC_SQRT"] = "1"
        os.environ["TORCH_MLU_SQRT_HIGH_PRECISION"] = "ON"

    def fused_lamb_dtype(
        self, test_type=torch.float, sz=[40, 40], len=2, is_mixed=False
    ):
        tensors = []
        fused_opt_params = []
        opt_params = []
        torch.manual_seed(9876)

        for i in range(len):
            tensors.append(
                torch.clamp(torch.rand(sz, dtype=torch.float).to("mlu"), min=0.1).to(
                    torch.half if is_mixed else test_type
                )
            )
        for tensor in tensors:
            fused_opt_params.append(torch.nn.Parameter(tensor.clone()))
            if is_mixed:
                opt_params.append(torch.nn.Parameter(tensor.clone().float()))
            else:
                opt_params.append(torch.nn.Parameter(tensor.clone()))
        fused_optimizer = self.fused_optimizer(
            fused_opt_params, use_nvlamb=True, **self.options
        )
        origin_optimizer = self.origin_optimizer(opt_params, **self.options)

        for _ in range(self.iters):
            for p_opt, p_fused_opt in zip(opt_params, fused_opt_params):
                p_opt.grad = torch.rand_like(p_opt)
                if is_mixed:
                    p_fused_opt.grad = p_opt.grad.half()
                else:
                    p_fused_opt.grad = p_opt.grad
            origin_optimizer.step()
            fused_optimizer.step()
            max_abs_diff = max_rel_diff = 0
            for p_ref, p_tst in zip(opt_params, fused_opt_params):
                if is_mixed:
                    p_ref = p_ref.half()
                max_abs_diff_p = (p_ref - p_tst).abs().max().item()
                max_rel_diff_p = ((p_ref - p_tst) / p_ref).abs().max().item()

                if max_abs_diff_p > max_abs_diff:
                    max_abs_diff = max_abs_diff_p
                if max_rel_diff_p > max_rel_diff:
                    max_rel_diff = max_rel_diff_p

            self.assertLessEqual(max_abs_diff, self.max_abs_diff)
            self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    # @unittest.skip("not test")
    @testinfo()
    def test_fused_lamb_fp32(self):
        # TODO: remove input data range restrictions.
        shape_list = [(1), (15, 20), (2, 3, 4), (8, 1, 2, 3), (2, 1, 2, 1, 4)]
        for sz in shape_list:
            for wd in [0, 0.01]:
                self.options["weight_decay"] = wd
                for len in [2, 40]:
                    self.fused_lamb_dtype(test_type=torch.float, sz=sz, len=len)

    # @unittest.skip("not test")
    @testinfo()
    def test_fused_lamb_amp(self):
        self.fused_optimizer = torch_mlu.optimizers.FusedLAMBAMP
        shape_list = [
            (4096, 1024),
            (1),
            (15, 20),
            (2, 3, 4),
            (8, 1, 2, 3),
            (2, 1, 2, 1, 4),
        ]
        wd_list = [0, 0.01]
        len_list = [2, 40]
        is_mixed_precision_list = [True, False]
        list_list = [shape_list, wd_list, len_list, is_mixed_precision_list]
        for sz, wd, len, is_mixed in product(*list_list):
            self.options["weight_decay"] = wd
            self.fused_lamb_dtype(
                test_type=torch.float, sz=sz, len=len, is_mixed=is_mixed
            )


if __name__ == "__main__":
    unittest.main()
