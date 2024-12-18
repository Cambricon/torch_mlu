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


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(256, 120)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(120, 84)
        self.relu4 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(84, 10)
        self.relu5 = torch.nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.reshape(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


class ModelLinear(torch.nn.Module):
    def __init__(self):
        super(ModelLinear, self).__init__()
        self.linear1 = torch.nn.Linear(4096, 2048)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2048, 128)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(128, 32320)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(32320, 1024)
        self.relu4 = torch.nn.ReLU()

    def forward(self, x):
        y = self.linear1(x)
        y = self.relu1(y)
        y = self.linear2(y)
        y = self.relu2(y)
        y = self.linear3(y)
        y = self.relu3(y)
        y = self.linear4(y)
        y = self.relu4(y)
        return y


class TestFusedOptimizer(unittest.TestCase):
    def setUp(self, max_abs_diff=1e-3, max_rel_diff=1, iters=7):
        self.max_abs_diff = max_abs_diff
        self.max_rel_diff = max_rel_diff
        self.iters = iters
        torch.manual_seed(9876)

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

    def run_net_and_compare_weight(
        self, model, model_, input_size, grad_size, gradScalar=False
    ):
        for options in self.options_list:
            options["fused"] = False
            options["foreach"] = False
            tst_options = copy.deepcopy(options)
            tst_options["fused"] = True
            ref_optim = torch.optim.AdamW(model.parameters(), **options)
            tst_optim = torch.optim.AdamW(model_.parameters(), **tst_options)
            if gradScalar is not None:
                ref_scalar = torch.mlu.amp.GradScaler()
                tst_scalar = torch.mlu.amp.GradScaler()
            memory_format = torch.contiguous_format
            if len(input_size) == 4:
                memory_format = torch.channels_last

            for i in range(30):
                x = torch.rand(input_size).to(memory_format=memory_format).mlu()
                x_ = x.clone()
                gt = torch.rand(grad_size).mlu()
                gt_ = gt.clone()

                # Reference
                y = model(x)
                loss = ((gt - y) ** 2).mean()

                if gradScalar is not None:
                    ref_scalar.scale(loss).backward()
                    ref_scalar.step(ref_optim)
                    ref_scalar.update()
                else:
                    loss.backward()
                    ref_optim.step()

                # DUT
                y = model_(x_)
                loss_mlu = ((gt_ - y) ** 2).mean()

                if gradScalar is not None:
                    tst_scalar.scale(loss_mlu).backward()
                    tst_scalar.step(tst_optim)
                    tst_scalar.update()
                else:
                    loss_mlu.backward()
                    tst_optim.step()

                for module in zip(model.modules(), model_.modules()):
                    m = module[0]
                    m_ = module[1]
                    if isinstance(m, torch.nn.Conv2d) or isinstance(
                        m_, torch.nn.Linear
                    ):
                        torch.testing.assert_close(
                            m.weight.cpu(),
                            m_.weight.cpu(),
                            atol=1e-3,
                            rtol=1e-3,
                            equal_nan=True,
                        )
                        torch.testing.assert_close(
                            m.weight.grad.cpu(),
                            m_.weight.grad.cpu(),
                            atol=1e-3,
                            rtol=1e-3,
                            equal_nan=True,
                        )

                # Init for next iteration
                ref_optim.zero_grad()
                tst_optim.zero_grad()
                model_.load_state_dict(copy.deepcopy(model.state_dict()))


class TestFusedAdamW(TestFusedOptimizer):
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

    def test_multi_tensor_float(self):
        fix_tensor_num = 1000
        self.gen_single_type_test(
            nelem=1024, param_type=torch.float, device="mlu", tensor_nums=fix_tensor_num
        )

    def test_network(self):
        for gradScaler in [False, True]:
            model = Model().mlu()
            model_ = Model().mlu()
            model_.load_state_dict(copy.deepcopy(model.state_dict()))
            input_size = [32, 1, 28, 28]
            grad_size = [32, 10]
            self.run_net_and_compare_weight(
                model, model_, input_size, grad_size, gradScaler
            )
            model = ModelLinear().mlu()
            model_ = ModelLinear().mlu()
            model_.load_state_dict(copy.deepcopy(model.state_dict()))
            input_size = [16, 4096]
            grad_size = [16, 1024]
            self.run_net_and_compare_weight(
                model, model_, input_size, grad_size, gradScaler
            )

    def test_frozen_model(self):
        nelem = 1
        for adam_option in self.options_list:
            tensor = torch.rand(nelem, dtype=torch.float, device="mlu")
            ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(
                [tensor], adam_option
            )

            # Add an empty param group which may occur for pipeline parallel p-tuning
            tst_optim.add_param_group({"params": []})

            for i in range(self.iters):
                self.gen_grad(ref_param, tst_param)
                ref_optim.step()
                tst_optim.step()
                max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)

                self.assertLessEqual(max_abs_diff, self.max_abs_diff)
                self.assertLessEqual(max_rel_diff, self.max_rel_diff)

    @unittest.skipIf(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("44GB")
    def test_fused_adamw_large(self):
        tensor = torch.clamp(
            torch.rand(2147483660).to(dtype=torch.float, device="mlu"),
            min=0.01,
            max=100.0,
        )
        ref_param = [torch.nn.Parameter(tensor.clone())]
        options = {
            "lr": 0.001,
            "betas": (0.6, 0.9),
            "eps": 3e-06,
            "weight_decay": 0.1,
            "amsgrad": False,
            "maximize": False,
            "fused": True,
            "foreach": False,
        }
        fused_optim = torch.optim.AdamW(ref_param, **options)
        for i in range(self.iters):
            ref_param[0].grad = torch.rand_like(ref_param[0])
            fused_optim.step()
        torch.mlu.synchronize()


if __name__ == "__main__":
    unittest.main()
