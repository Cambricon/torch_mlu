import sys
import os
import torch
import torch.nn as nn
import torch.serialization as se
import torchvision.models as models
from torchvision.models.googlenet import InceptionAux
import copy
from torch.utils.checkpoint import checkpoint
import random

import unittest  # pylint: disable=C0411

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411


def opt_to(optimizer, device):
    for state_perparam in optimizer.state.values():
        for k, v in state_perparam.items():
            if isinstance(v, torch.Tensor):
                state_perparam[k] = v.to(device)
    return optimizer


class TestSaveAndLoad(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_save_and_load(self):
        def test_load_from_cpu():
            net = models.resnet18()
            net = net.to("mlu")
            optimizer = torch.optim.SGD(
                net.parameters(), 0, momentum=0.9, weight_decay=1e-4
            )
            checkpoint = torch.load("net_cpu.pth")
            net.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            return net

        def test_load_from_mlu():
            net = models.resnet18()
            net = net.to("mlu")
            optimizer = torch.optim.SGD(
                net.parameters(), 0, momentum=0.9, weight_decay=1e-4
            )
            checkpoint = torch.load("net_mlu.pth")
            net.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            return net

        def test_save_as_cpu():
            mlu = torch.device("mlu")
            net = models.resnet18()
            net.to(mlu)
            optimizer = torch.optim.SGD(
                net.parameters(), 0, momentum=0.9, weight_decay=1e-4
            )
            optimizer = opt_to(optimizer, mlu)
            checkpoint = {
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_path = "./"
            save_name = "net_cpu"
            torch.save(checkpoint, os.path.join(save_path, save_name + ".pth"))
            return net

        def test_save_as_mlu():
            mlu = torch.device("mlu")
            net = models.resnet18()
            net.to(mlu)
            optimizer = torch.optim.SGD(
                net.parameters(), 0, momentum=0.9, weight_decay=1e-4
            )
            optimizer = opt_to(optimizer, mlu)
            checkpoint = {
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_path = "./"
            save_name = "net_mlu"
            torch.save(checkpoint, os.path.join(save_path, save_name + ".pth"))
            return net

        net_cpu = test_save_as_cpu()
        net_mlu = test_save_as_mlu()
        net_from_cpu = test_load_from_cpu()
        net_from_mlu = test_load_from_mlu()
        self.assertTensorsEqual(
            net_cpu.conv1.weight.cpu(),
            net_from_cpu.conv1.weight.cpu(),
            0.0,
            use_MSE=True,
        )
        self.assertTensorsEqual(
            net_mlu.conv1.weight.cpu(),
            net_from_mlu.conv1.weight.cpu(),
            0.0,
            use_MSE=True,
        )

    @unittest.skip("wait for https://github.com/pytorch/vision/pull/8180")
    @testinfo()
    def test_desnet201_checkpoint_preserve_rng_state(self):
        class TestCheckPointGoogleNet(nn.Module):
            def __init__(self, model):
                super(TestCheckPointGoogleNet, self).__init__()
                self.model = model

            def forward(self, input):
                def create_custom_forward(module):
                    def custom_forward(input):
                        return module(input)

                    return custom_forward

                out = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.model), input, use_reentrant=True
                )
                return out

        input = torch.randn(5, 3, 224, 224, dtype=torch.float32).mlu()
        input.requires_grad = True
        checkpoint_input = copy.deepcopy(input)
        checkpoint_input.requires_grad = True
        state = torch.mlu.get_rng_state()
        densenet201 = models.densenet201(memory_efficient=True).mlu()
        output = densenet201(input)
        checkpoint_densenet = TestCheckPointGoogleNet(copy.deepcopy(densenet201))
        torch.mlu.set_rng_state(state)
        checkpoint_output = checkpoint_densenet(checkpoint_input)
        output.backward(torch.ones(output.size(), device="mlu"))
        checkpoint_output.backward(torch.ones(checkpoint_output.size(), device="mlu"))
        self.assertTensorsEqual(
            input.grad.cpu(), checkpoint_input.grad.cpu(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_checkpoint_preserve_rng_state(self):
        class TestCheckPointGoogleNet(nn.Module):
            def __init__(self, model):
                super(TestCheckPointGoogleNet, self).__init__()
                self.model = model

            def forward(self, input):
                def create_custom_forward(module):
                    def custom_forward(input):
                        return module(input)

                    return custom_forward

                out = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.model), input, use_reentrant=True
                )
                return out

        input = torch.randn(5, 3, 224, 224, dtype=torch.float32).mlu()
        input.requires_grad = True
        checkpoint_input = copy.deepcopy(input)
        checkpoint_input.requires_grad = True
        state = torch.mlu.get_rng_state()
        inception = InceptionAux(in_channels=3, num_classes=1000).mlu()
        output = inception(input)
        checkpoint_inception = TestCheckPointGoogleNet(copy.deepcopy(inception))
        torch.mlu.set_rng_state(state)
        checkpoint_output = checkpoint_inception(checkpoint_input)
        output.backward(torch.ones(output.size(), device="mlu"))
        checkpoint_output.backward(torch.ones(checkpoint_output.size(), device="mlu"))
        self.assertTensorsEqual(
            input.grad.cpu(), checkpoint_input.grad.cpu(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_checkpoint_rng_mlu(self):
        for _ in range(5):
            inp = torch.randn(20000, device="mlu").requires_grad_()
            phase1 = torch.nn.Dropout()
            phase2 = torch.nn.Dropout()

            def run_fn(input):
                return phase2(input)

            state = torch.mlu.get_rng_state()

            out = phase1(inp)
            out = checkpoint(run_fn, out, use_reentrant=True)
            out.sum().backward()
            grad_with_checkpointing = inp.grad

            torch.mlu.set_rng_state(state)

            inp.grad = None

            out = phase1(inp)
            out = run_fn(out)
            out.sum().backward()
            grad_no_checkpointing = inp.grad

            self.assertEqual(grad_with_checkpointing, grad_no_checkpointing)

    # @unittest.skip("not test")
    @testinfo()
    def test_checkpoint_non_tensor(self):
        def run_fn(tensor1, tensor2):
            if tensor2 is None:
                return tensor1
            return tensor1 + tensor2

        input_var = torch.randn(1, 100, requires_grad=True).mlu()
        out = checkpoint(run_fn, input_var, None, use_reentrant=True)
        out.sum().backward()

    # @unittest.skip("not test")
    @testinfo()
    def test_checkpoint_non_tensor_inputs_outputs(self):
        def foo(t1, t2, scale, t3):
            t4 = t1 + t2 * t3
            t5 = t1 * t2 + t3
            t4 *= scale
            t5 *= scale
            return scale, t4, None, True, t5, "bar", t1

        t1 = torch.rand(10, requires_grad=True).mlu()
        t2 = torch.rand(10, requires_grad=True).mlu()
        t3 = torch.rand(10).mlu()
        scale = random.randint(0, 10)
        res = checkpoint(foo, t1, t2, scale, t3, use_reentrant=True)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

        # Validate running backward.
        res[1].sum().backward(retain_graph=True)
        res[4].sum().backward(retain_graph=True)
        res[6].sum().backward()
        with self.assertRaisesRegex(
            RuntimeError, "Trying to backward through the graph a second time"
        ):
            res[6].sum().backward()
        t1_grad = t1.grad
        t2_grad = t2.grad

        # Reset grads, run without checkpoint and validate we receive same grads.
        t1.grad = None
        t2.grad = None
        res = foo(t1, t2, scale, t3)
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        self.assertEqual(t1.grad, t1_grad)
        self.assertEqual(t2.grad, t2_grad)

    # @unittest.skip("not test")
    @testinfo()
    def test_checkpoint_partial_grad(self):
        def run_fn(tensor1, tensor2):
            # tensor 2 is used for other application logic
            return tensor1, tensor2

        input_var = torch.randn(1, 4, requires_grad=True).mlu()
        input_var2 = torch.randn(1, 4, requires_grad=False).mlu()
        out = checkpoint(run_fn, input_var, input_var2, use_reentrant=True)
        out[0].sum().backward()

        def run_fn2(tensor1, tensor2):
            return tensor1

        input_var = torch.randn(1, 4, requires_grad=False).mlu()
        input_var2 = torch.randn(1, 4, requires_grad=True).mlu()
        with self.assertRaisesRegex(
            RuntimeError,
            r"none of output has requires_grad=True, this checkpoint\(\) is not necessary",
        ):
            out = checkpoint(run_fn2, input_var, input_var2, use_reentrant=True)
            out.sum().backward()

    # @unittest.skip("not test")
    def test_checkpoint_not_preserve_rng_state_and_without_reentrant(self):
        inp = torch.randn(2, device="mlu").requires_grad_()
        layer = torch.nn.Dropout()

        def run_fn(input):
            return layer(input)

        out = checkpoint(run_fn, inp, use_reentrant=False, preserve_rng_state=False)
        out.sum().backward()
        # This should run without error


if __name__ == "__main__":
    unittest.main()
