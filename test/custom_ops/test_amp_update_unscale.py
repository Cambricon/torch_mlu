import unittest
import logging
import os
import sys
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    TestCase,
    testinfo,
    largeTensorTest,
    TEST_LARGETENSOR,
)  # pylint: disable=C0411,C0413

TEST_MULTIMLU = torch.mlu.is_available() and torch.mlu.device_count() >= 2

logging.basicConfig(level=logging.DEBUG)


def amp_unscale_cpu_(grads, inv_scale):
    outs = []
    for grad in grads:
        outs.append(grad.mul_(inv_scale))
    return outs


class TestAmpOP(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_grad_unscale_found_inf_op(self, dtype=torch.float):
        inv_scale = torch.full((1,), 0.25, dtype=torch.float, device="mlu:0")
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device="mlu:0")

        size = 10
        g = torch.full((size, size), 4.0, dtype=dtype, device="mlu:0")
        ginf = g.clone()
        ginf[2, 2] = float("inf")
        gnan = g.clone()
        gnan[2, 2] = float("nan")
        gzero = torch.randn(0, device="mlu:0")

        cases = (
            ([g.clone()], False),
            ([ginf.clone()], True),
            ([gnan.clone()], True),
            ([g.clone(), g.clone()], False),
            ([g.clone().half(), g.clone().half()], False),
            ([gzero.clone(), g.clone()], False),
            ([g.clone(), g.clone().t()], False),
            ([g.clone(), g.clone()[:, :5]], False),
            ([g.clone()[:, :5], g.clone()[:, :5]], False),
            ([g.clone(), ginf.clone()], True),
            ([g.clone(), gnan.clone()], True),
            ([g.clone(), ginf.clone()[:, :5]], True),
            ([g.clone(), gnan.clone()[:, :5]], True),
            ([ginf.clone(), g.clone()[:, :5]], True),
            ([ginf.clone()[:, :5], g.clone()[:, :5]], True),
        )

        for grads, has_inf in cases:
            found_inf.zero_()
            torch._amp_foreach_non_finite_check_and_unscale_(
                grads, found_inf, inv_scale
            )
            if has_inf:
                self.assertEqual(found_inf, 1.0)
            else:
                self.assertEqual(found_inf, 0.0)
                for grad in grads:
                    self.assertTrue(
                        torch.allclose(
                            grad.cpu(), torch.ones_like(grad.cpu()), atol=1e-7
                        )
                    )

        def perfect_storm_grads(inject_inf):
            grads = [
                g.clone(),
                g.clone()[:, :5],
                g.to(dtype=torch.float16),
                g.to(dtype=torch.float16),
            ]
            if TEST_MULTIMLU:
                g_cpu = g.cpu()
                grads += [
                    g_cpu.to(device="mlu:1"),
                    g_cpu.to(device="mlu:1")[:, :5],
                    g_cpu.to(device="mlu:1", dtype=torch.float16),
                    g_cpu.to(device="mlu:1", dtype=torch.float16),
                ]
            if inject_inf >= 0:
                grads[inject_inf][2, 2] = float("inf")
            return grads

        scaler = torch.mlu.amp.GradScaler()
        dummy_params = [torch.empty_like(g) for g in perfect_storm_grads(-1)]
        dummy_opt = torch.optim.SGD(dummy_params, lr=1.0)

        for inject_inf in range(-1, len(dummy_params)):
            found_inf = torch.full((1,), 0.0, dtype=torch.float, device="mlu:0")
            grads = perfect_storm_grads(inject_inf)
            for i, p in enumerate(dummy_params):
                p.grad = grads[i]
            found_inf_per_device = scaler._unscale_grads_(
                dummy_opt, inv_scale, found_inf, True
            )
            if inject_inf < 0:
                self.assertTrue(
                    sum(v.item() for v in found_inf_per_device.values()) == 0
                )
                for grad in grads:
                    self.assertTrue(
                        torch.allclose(
                            grad.cpu(), torch.ones_like(grad.cpu()), atol=1e-7
                        )
                    )
            else:
                self.assertTrue(
                    sum(v.item() for v in found_inf_per_device.values()) == 1
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_amp_update_scale_op(self, device="mlu", dtype=torch.float):
        growth = 2.0
        backoff = 0.25
        growth_interval = 2
        scale = torch.full((1,), 4.0, dtype=dtype, device=device)
        growth_tracker = torch.full((1,), 0.0, dtype=torch.int32, device=device)
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device="mlu:0")

        # Simulates 2 consecutive unskipped iterations
        scale = torch._amp_update_scale_(
            scale, growth_tracker, found_inf, growth, backoff, growth_interval
        )
        self.assertEqual(growth_tracker, 1)
        self.assertEqual(scale, 4.0)
        scale = torch._amp_update_scale_(
            scale, growth_tracker, found_inf, growth, backoff, growth_interval
        )
        self.assertEqual(growth_tracker, 0)
        self.assertEqual(scale, 8.0)

        # Simulates a skipped iteration
        found_inf.fill_(1.0)
        scale = torch._amp_update_scale_(
            scale, growth_tracker, found_inf, growth, backoff, growth_interval
        )
        self.assertEqual(growth_tracker, 0)
        self.assertEqual(scale, 2.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_overlength_tensors(self):
        length = 3000
        grad_cpu = torch.randn(1, 2, 3, 2)
        grad_mlu = grad_cpu.mlu()
        grads_cpu = [grad_cpu] * length
        grads_mlu = [grad_mlu] * length

        found_inf = torch.mlu.FloatTensor([0.0])
        initial_scale = 2**8
        _scale = torch.mlu.FloatTensor([initial_scale]).double().reciprocal().float()
        torch._amp_foreach_non_finite_check_and_unscale_(grads_mlu, found_inf, _scale)
        amp_unscale_cpu_(grads_cpu, _scale.cpu())
        found_inf_real = torch.FloatTensor([0.0])
        self.assertTrue(torch.allclose(found_inf.cpu(), found_inf_real, atol=0.0))
        for i in range(length):
            grad_cpu = grads_cpu[i]
            grad_mlu = grads_mlu[i]
            self.assertTensorsEqual(
                grad_cpu, grad_mlu.cpu(), 3e-3, allow_inf=True, use_MSE=True
            )

    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    @testinfo()
    def test_large_tensor(self):
        num = 500
        grads_cpu = []
        grads_mlu = []
        for i in range(num):
            grad_cpu = torch.randn(1000, 6144)
            grad_mlu = grad_cpu.mlu()
            if i == int(num / 2):
                grad_cpu[0][0] = float("inf")
                grad_mlu[0][0] = float("inf")
            grads_cpu.append(grad_cpu)
            grads_mlu.append(grad_mlu)

        found_inf = torch.mlu.FloatTensor([0.0])
        initial_scale = 2**16
        _scale = torch.mlu.FloatTensor([initial_scale]).double().reciprocal().float()
        torch._amp_foreach_non_finite_check_and_unscale_(grads_mlu, found_inf, _scale)
        amp_unscale_cpu_(grads_cpu, _scale.cpu())
        found_inf_real = torch.FloatTensor([1.0])
        self.assertTrue(torch.allclose(found_inf.cpu(), found_inf_real, atol=0.0))
        for i in range(num):
            grad_cpu = grads_cpu[i]
            grad_mlu = grads_mlu[i]
            self.assertTensorsEqual(
                grad_cpu, grad_mlu.cpu(), 3e-3, allow_inf=True, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_amp_unscale_boundary(self):
        input_1 = torch.randn(1, torch.iinfo(torch.int32).max).mlu()
        input_1[0][1000] = float("nan")
        input_2 = torch.randn(1, 2, 3, 4).mlu()
        inputs = [input_1, input_2]
        inputs_cpu = [input_1.cpu(), input_2.cpu()]
        found_inf = torch.mlu.FloatTensor([0.0])
        initial_scale = 0.25
        _scale = torch.mlu.FloatTensor([initial_scale]).double().reciprocal().float()
        torch._amp_foreach_non_finite_check_and_unscale_(inputs, found_inf, _scale)
        amp_unscale_cpu_(inputs_cpu, _scale.cpu())
        found_inf_real = torch.FloatTensor([1.0])
        self.assertTrue(torch.allclose(found_inf.cpu(), found_inf_real, atol=0.0))
        for i, grad_mlu in enumerate(inputs):
            grad_cpu = inputs_cpu[i]
            self.assertTensorsEqual(
                grad_cpu, grad_mlu.cpu(), 3e-3, allow_inf=True, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_amp_unscale_mix_memory_format(self):
        input_1 = torch.randn(2, 3, 4, 5).mlu().to(memory_format=torch.channels_last)
        input_1[0][0][0][4] = float("nan")
        input_2 = torch.randn((10)).mlu()
        inputs = [input_1, input_2]
        inputs_cpu = [input_1.cpu(), input_2.cpu()]
        found_inf = torch.mlu.FloatTensor([0.0])
        initial_scale = 0.25
        _scale = torch.mlu.FloatTensor([initial_scale]).double().reciprocal().float()
        torch._amp_foreach_non_finite_check_and_unscale_(inputs, found_inf, _scale)
        amp_unscale_cpu_(inputs_cpu, _scale.cpu())
        found_inf_real = torch.FloatTensor([1.0])
        self.assertTrue(torch.allclose(found_inf.cpu(), found_inf_real, atol=0.0))
        for i, grad_mlu in enumerate(inputs):
            grad_cpu = inputs_cpu[i]
            self.assertTensorsEqual(
                grad_cpu, grad_mlu.cpu(), 3e-3, allow_inf=True, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_amp_unscale_exception(self):
        input = torch.randn(1, torch.iinfo(torch.int32).max + 1).mlu()
        found_inf = torch.mlu.FloatTensor([0.0])
        initial_scale = 2**16
        _scale = torch.mlu.FloatTensor([initial_scale]).double().reciprocal().float()
        ref_msg = "Not implemented large tensor."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch._amp_foreach_non_finite_check_and_unscale_([input], found_inf, _scale)


if __name__ == "__main__":
    unittest.main()
