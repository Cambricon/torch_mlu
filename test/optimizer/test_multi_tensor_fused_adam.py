from itertools import product
from typing import Optional
import copy
import unittest
import torch
import torch_mlu
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import TEST_LARGETENSOR, largeTensorTest, TestCase

torch.manual_seed(12345)


class SimpleModel(torch.nn.Module):
    def __init__(self, num_layers, size):
        super().__init__()
        self.params = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.rand(1, size) + 1) for _ in range(num_layers)]
        )

    def forward(self, x):
        y = 0
        for i, param in enumerate(self.params):
            y += (i + 1) * param + x
        return y


def make_models(
    num_layers: int = 1,
    size: int = 10,
    lr: float = 0.1,
    beta1: float = 0.1,
    beta2: float = 0.2,
    eps: float = 0.25,
    weight_decay: float = 0.1,
    adam_w_mode: bool = True,
    model_dtype: torch.dtype = torch.float32,
    optim_dtype: Optional[torch.dtype] = None,
    store_param_remainders: bool = False,
    with_mlu_graph: bool = False,
):
    ref_model = SimpleModel(num_layers, size).to(device="mlu", dtype=model_dtype)
    optim_args = dict(lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    torch_optim_class = torch.optim.AdamW if adam_w_mode else torch.optim.Adam
    ref_optim = torch_optim_class(
        [
            {"params": list(ref_model.parameters())},
        ],
        **optim_args
    )
    x = torch.randn(1, device="mlu", dtype=model_dtype)
    dy = torch.randn((1, size), device="mlu", dtype=model_dtype)
    y = ref_model(x)
    ref_optim.zero_grad()
    y.backward(dy)
    tst_param = [copy.deepcopy(p) for p in list(ref_model.parameters())]
    tst_grad = [copy.deepcopy(p.grad) for p in list(ref_model.parameters())]
    tst_exp_avg = [
        torch.zeros_like(p, memory_format=torch.preserve_format, dtype=optim_dtype)
        for p in list(ref_model.parameters())
    ]
    tst_exp_avg_sq = [
        torch.zeros_like(p, memory_format=torch.preserve_format, dtype=optim_dtype)
        for p in list(ref_model.parameters())
    ]
    if store_param_remainders:
        tst_param_out = [
            torch.zeros_like(
                p, memory_format=torch.preserve_format, dtype=torch.bfloat16
            )
            for p in list(ref_model.parameters())
        ]
    else:
        tst_param_out = [
            torch.zeros_like(p, memory_format=torch.preserve_format, dtype=optim_dtype)
            for p in list(ref_model.parameters())
        ]
    ref_optim.step()
    grad_scale = torch.tensor([1.0], device="mlu", dtype=torch.float32)
    tst_model = dict(
        exp_avgs=tst_exp_avg,
        exp_avg_sqs=tst_exp_avg_sq,
        grads=tst_grad,
        params_out=tst_param_out,
        grad_scale=grad_scale,
    )
    if store_param_remainders:
        tst_model["params_in"] = tst_param
        tst_model["params_rem"] = [
            torch.ones_like(p, memory_format=torch.preserve_format, dtype=torch.int16)
            for p in list(ref_model.parameters())
        ]
    else:
        tst_model["params"] = tst_param
    lr = torch.tensor([lr], device="mlu", dtype=torch.float32) if with_mlu_graph else lr
    step = torch.tensor([1], device="mlu", dtype=torch.int32) if with_mlu_graph else 1
    tst_optim = torch.ops.torch_mlu.multi_tensor_fused_adam
    if with_mlu_graph:
        tst_optim = torch.ops.torch_mlu.multi_tensor_fused_adam_capturable
    if store_param_remainders:
        tst_optim = torch.ops.torch_mlu.multi_tensor_fused_adam_remainder
    tst_optim(
        **tst_model,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        step=step,
        mode=int(adam_w_mode),
        bias_correction=1,
        weight_decay=weight_decay
    )
    return ref_model, ref_optim, tst_model, tst_optim


class TestDistributedFusedAdam(TestCase):
    # @unittest.skip("not test")
    def test_multi_tensor_fused_adam(self):
        num_layers = [10, 20, 40]
        sizes = [10, 100, 1000, 10000, 50000]
        dtypes = [torch.float32, torch.float16]
        if torch.mlu.is_bf16_supported():
            dtypes = [torch.float32, torch.float16, torch.bfloat16]
        adam_w_modes = [True, False]
        is_capturable_list = [True, False]
        for num_layer in num_layers:
            for size in sizes:
                for dtype in dtypes:
                    for adam_w_mode in adam_w_modes:
                        for is_capturable in is_capturable_list:
                            prec = 0.003 if dtype == torch.float32 else 0.05
                            ref_model, ref_optim, tst_model, tst_optim = make_models(
                                num_layer,
                                size,
                                0.99,
                                0.1,
                                0.2,
                                0.001,
                                0.1,
                                adam_w_mode,
                                dtype,
                                dtype,
                                False,
                                is_capturable,
                            )
                            num = len(list(ref_model.parameters()))
                            for i in range(num):
                                ref_p = list(ref_model.parameters())[i]
                                ref_exp_avg = ref_optim.state[ref_p]["exp_avg"]
                                ref_exp_avg_sq = ref_optim.state[ref_p]["exp_avg_sq"]
                                self.assertTensorsEqual(
                                    ref_p.cpu().double(),
                                    tst_model["params"][i].cpu().double(),
                                    prec=prec,
                                    allow_inf=True,
                                    use_MSE=True,
                                )
                                self.assertTensorsEqual(
                                    ref_exp_avg.cpu().double(),
                                    tst_model["exp_avgs"][i].cpu().double(),
                                    prec=prec,
                                    allow_inf=True,
                                    use_MSE=True,
                                )
                                self.assertTensorsEqual(
                                    ref_exp_avg_sq.cpu().double(),
                                    tst_model["exp_avg_sqs"][i].cpu().double(),
                                    prec=prec,
                                    allow_inf=True,
                                    use_MSE=True,
                                )
                                self.assertTensorsEqual(
                                    ref_p.cpu().double(),
                                    tst_model["params_out"][i].cpu().double(),
                                    prec=prec,
                                    allow_inf=True,
                                    use_MSE=True,
                                )

    # @unittest.skip("not test")
    @unittest.skipIf(not torch.mlu.is_bf16_supported(), "BF16 is not supported")
    def test_multi_tensor_fused_adam_remainder(self):
        num_layers = [10, 20]
        sizes = [10, 100, 1000, 10000, 40000]
        adam_w_modes = [True, False]
        for num_layer in num_layers:
            for size in sizes:
                for adam_w_mode in adam_w_modes:
                    ref_model, ref_optim, tst_model, tst_optim = make_models(
                        num_layer,
                        size,
                        0.99,
                        0.1,
                        0.2,
                        0.001,
                        0.1,
                        adam_w_mode,
                        torch.bfloat16,
                        torch.float32,
                        True,
                        False,
                    )
                    num = len(list(ref_model.parameters()))
                    for i in range(num):
                        ref_p = list(ref_model.parameters())[i]
                        ref_exp_avg = ref_optim.state[ref_p]["exp_avg"]
                        ref_exp_avg_sq = ref_optim.state[ref_p]["exp_avg_sq"]
                        self.assertTensorsEqual(
                            ref_p.cpu().double(),
                            tst_model["params_out"][i].cpu().double(),
                            prec=0.05,
                            allow_inf=True,
                            use_MSE=True,
                        )
                        self.assertTensorsEqual(
                            ref_exp_avg.cpu().double(),
                            tst_model["exp_avgs"][i].cpu().double(),
                            prec=0.05,
                            allow_inf=True,
                            use_MSE=True,
                        )
                        self.assertTensorsEqual(
                            ref_exp_avg_sq.cpu().double(),
                            tst_model["exp_avg_sqs"][i].cpu().double(),
                            prec=0.05,
                            allow_inf=True,
                            use_MSE=True,
                        )

    # @unittest.skip("not test")
    @unittest.skipIf(not torch.mlu.is_bf16_supported(), "BF16 is not supported")
    def test_multi_tensor_fused_adamw_remainder_vs_cuda(self):
        param_in = [
            torch.tensor(
                [
                    1.2832,
                    1.6562,
                    1.2383,
                    1.7314,
                    1.6016,
                    1.3047,
                    1.2549,
                    1.6289,
                    1.9668,
                    1.7402,
                    1.4521,
                    1.4756,
                    1.7842,
                    1.1523,
                    1.6660,
                    1.3340,
                ],
                device="mlu",
                dtype=torch.bfloat16,
            )
        ]
        param_rem = [
            torch.tensor(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                device="mlu",
                dtype=torch.int16,
            )
        ]
        exp_avg = [
            torch.tensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                device="mlu",
                dtype=torch.float32,
            )
        ]
        exp_avg_sq = [
            torch.tensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                device="mlu",
                dtype=torch.float32,
            )
        ]
        grad = [
            torch.tensor(
                [
                    0.0465,
                    0.0105,
                    0.0012,
                    0.0889,
                    -0.0082,
                    -0.0473,
                    0.1197,
                    -0.0053,
                    0.0226,
                    -0.0463,
                    0.1036,
                    0.0319,
                    0.0429,
                    -0.0085,
                    -0.0052,
                    0.0577,
                ],
                device="mlu",
                dtype=torch.float16,
            )
        ]
        param_out = [
            torch.tensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                device="mlu",
                dtype=torch.bfloat16,
            )
        ]
        grad_scale = torch.tensor([1.0], device="mlu", dtype=torch.float32)
        cuda_ref_param_out = [
            torch.tensor(
                [
                    1.2500,
                    1.6328,
                    1.2266,
                    1.6875,
                    1.5859,
                    1.3047,
                    1.2109,
                    1.6094,
                    1.9375,
                    1.7422,
                    1.4062,
                    1.4531,
                    1.7500,
                    1.1406,
                    1.6484,
                    1.3047,
                ],
                device="mlu",
                dtype=torch.bfloat16,
            )
        ]
        cuda_ref_param_rem = [
            torch.tensor(
                [
                    23087,
                    23867,
                    22864,
                    27626,
                    23370,
                    24023,
                    16116,
                    12168,
                    27453,
                    -15075,
                    25584,
                    -22156,
                    -10161,
                    -3220,
                    8572,
                    -7244,
                ],
                device="mlu",
                dtype=torch.int16,
            )
        ]
        lr = 0.1
        beta1 = 0.1
        beta2 = 0.2
        eps = 0.25
        step = 1
        mode = 1
        bias_correction = 1
        weight_decay = 0.1
        torch.ops.torch_mlu.multi_tensor_fused_adam_remainder(
            param_in,
            param_rem,
            exp_avg,
            exp_avg_sq,
            grad,
            param_out,
            grad_scale,
            lr,
            beta1,
            beta2,
            eps,
            step,
            mode,
            bias_correction,
            weight_decay,
        )
        self.assertTensorsEqual(
            param_out[0], cuda_ref_param_out[0], prec=0.0001, use_MSE=True
        )
        self.assertTensorsEqual(
            param_rem[0].cpu(), cuda_ref_param_rem[0].cpu(), prec=0.0001, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
