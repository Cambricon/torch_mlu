import unittest
import logging
import pickle
from itertools import chain
import collections
import os
import sys
import torch
from typing import Optional

from torch.testing._internal.autocast_test_lists import AutocastTestLists

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import TestCase, testinfo  # pylint: disable=C0411

TEST_MULTIMLU = torch.is_mlu_available() and torch.mlu.device_count() >= 2
TEST_BFLOAT16 = torch.mlu.is_bf16_supported()
# DEVICE_TYPE = ct.is_using_floating_device()
DEVICE_TYPE = True

logging.basicConfig(level=logging.DEBUG)


class TestAmp(TestCase):
    def setUp(self):
        super(TestAmp, self).setUp()
        self.autocast_lists = AutocastTestLists(torch.device("mlu:0"))

    def tearDown(self):
        del self.autocast_lists
        super(TestAmp, self).tearDown()

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not TEST_MULTIMLU, "only one MLU detected")
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_a_autocast_half_to_float_inf(self):
        with torch.mlu.amp.autocast(enabled=False):
            input1 = torch.tensor(
                torch.finfo(torch.float16).max, dtype=torch.float16
            ).to("mlu")
            input2 = torch.tensor(
                -torch.finfo(torch.float16).max, dtype=torch.float16
            ).to("mlu")
            output1 = input1.to(torch.float32)
            output2 = input2.to(torch.float32)
            self.assertFalse(torch.isinf(output1).data.item())
            self.assertFalse(torch.isinf(output2).data.item())

        with torch.mlu.amp.autocast(enabled=True):
            input1 = torch.tensor(
                torch.finfo(torch.float16).max, dtype=torch.float16
            ).to("mlu")
            input2 = torch.tensor(
                -torch.finfo(torch.float16).max, dtype=torch.float16
            ).to("mlu")
            output1 = input1.to(torch.float32)
            output2 = input2.to(torch.float32)
            if torch.mlu.get_device_properties(torch.mlu.current_device()).major == 3:
                self.assertTrue(torch.isinf(output1).data.item())
                self.assertTrue(torch.isinf(output2).data.item())
            else:
                self.assertFalse(torch.isinf(output1).data.item())
                self.assertFalse(torch.isinf(output2).data.item())

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not TEST_MULTIMLU, "only one MLU detected")
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_grad_scaling_scale(self):
        scaler = torch.mlu.amp.GradScaler(init_scale=2.0)
        t0 = torch.full((1,), 4.0, dtype=torch.float32, device="mlu:0")
        t1 = torch.full((1,), 4.0, dtype=torch.float32, device="mlu:1")
        # Create some nested iterables of tensors on different devices.
        outputs = (
            t1.clone(),
            (t0.clone(), t1.clone()),
            [t0.clone(), (t1.clone(), t0.clone())],
        )
        outputs = scaler.scale(outputs)
        self.assertTrue(
            outputs[0] == 8.0
            and outputs[1][0] == 8.0
            and outputs[1][1] == 8.0
            and outputs[2][0] == 8.0
            and outputs[2][1][0] == 8.0
            and outputs[2][1][1] == 8.0
        )
        self.assertTrue(scaler._scale.device == t1.device)

    @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_grad_scaling_unscale_sparse(self, device="mlu", dtype=torch.float):
        scaler = torch.mlu.amp.GradScaler()

        inv_scale = torch.full((1,), 0.25, dtype=dtype, device=device)
        found_inf = torch.empty((1,), dtype=dtype, device=device)
        cur = found_inf.device

        i = torch.tensor([[0, 1, 1], [2, 0, 2]], device="mlu", dtype=torch.int64)
        v = torch.tensor([16.0, 32.0, 64.0], device="mlu", dtype=torch.float)
        s = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device="mlu", dtype=dtype)

        p = s.clone()
        assert p.is_sparse
        opt = torch.optim.SGD([p], lr=1.0)

        p.grad = s.clone()
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 0.0)
        self.assertTrue(torch.allclose(p.grad.to_dense(), (s / 4).to_dense()))

        v = torch.FloatTensor([16.0, 32.0, float("inf")])
        p.grad = torch.sparse_coo_tensor(
            i, v, torch.Size([2, 3]), device="mlu", dtype=dtype
        )
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 1.0)

        v = torch.FloatTensor([16.0, 32.0, float("nan")])
        p.grad = torch.sparse_coo_tensor(
            i, v, torch.Size([2, 3]), device="mlu", dtype=dtype
        )
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        self.assertEqual(found_inf, 1.0)

        p = s.clone().half()
        assert p.is_sparse
        opt = torch.optim.SGD([p], lr=1.0)

        p.grad = s.clone().half()
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, True)[cur]
        self.assertEqual(found_inf, 0.0)
        self.assertTrue(torch.allclose(p.grad.to_dense(), (s.half() / 4).to_dense()))

        i = torch.LongTensor([[0, 1, 0], [2, 0, 2]])
        v = torch.FloatTensor([64000.0, 32.0, 64000.0])
        p.grad = torch.sparse_coo_tensor(
            i, v, torch.Size([2, 3]), device="mlu", dtype=torch.float16
        )
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, True)[cur]
        self.assertEqual(found_inf, 1.0)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_grad_scaling_state_dict(self):
        for lazy_init_scale in True, False:
            s0 = torch.mlu.amp.GradScaler(
                init_scale=3.0, growth_factor=4.0, backoff_factor=0.5, growth_interval=2
            )
            s1 = torch.mlu.amp.GradScaler(
                init_scale=6.0, growth_factor=7.0, backoff_factor=0.8, growth_interval=1
            )

            # sets a random value for load_state_dict to overwrite
            s1._init_growth_tracker = 7

            if lazy_init_scale:
                # Dummy scale() call to ensure the scale tensor is lazily initialized.
                s1.scale(torch.full((1,), 4.0, dtype=torch.float32, device="mlu:0"))
                self.assertTrue(torch.is_floating_point(s1._scale) and s1._scale.is_mlu)

            s1.load_state_dict(s0.state_dict())

            self.assertEqual(s1.get_scale(), 3.0)
            self.assertEqual(s1.get_growth_factor(), 4.0)
            self.assertEqual(s1.get_backoff_factor(), 0.5)
            self.assertEqual(s1.get_growth_interval(), 2)
            self.assertEqual(s1._init_growth_tracker, 0)

    def _create_scaling_models_optimizers(self, device="mlu"):
        mod_control = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
        ).to(device=device)
        mod_scaling = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
        ).to(device=device)
        for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
            s.data.copy_(c.data)

        opt_control = torch.optim.SGD(mod_control.parameters(), lr=1.0)
        opt_scaling = torch.optim.SGD(mod_scaling.parameters(), lr=1.0)

        return mod_control, mod_scaling, opt_control, opt_scaling

    def _create_scaling_case(self, device="mlu", dtype=torch.float):
        data = [
            (
                torch.randn((8, 8), dtype=dtype, device=device),
                torch.randn((8, 8), dtype=dtype, device=device),
            ),
            (
                torch.randn((8, 8), dtype=dtype, device=device),
                torch.randn((8, 8), dtype=dtype, device=device),
            ),
            (
                torch.randn((8, 8), dtype=dtype, device=device),
                torch.randn((8, 8), dtype=dtype, device=device),
            ),
            (
                torch.randn((8, 8), dtype=dtype, device=device),
                torch.randn((8, 8), dtype=dtype, device=device),
            ),
        ]

        loss_fn = torch.nn.MSELoss().to("mlu")

        skip_iter = 2

        return self._create_scaling_models_optimizers(device=device) + (
            data,
            loss_fn,
            skip_iter,
        )

    def _run_scaling_case(self, run, unskipped, skipped, atol=1e-5):
        # Ensure scaling can be disabled without changing user control flow.
        for enabled in True, False:
            (
                mod_control,
                mod_scaling,
                opt_control,
                opt_scaling,
                data,
                loss_fn,
                skip_iter,
            ) = self._create_scaling_case()

            scaler = torch.mlu.amp.GradScaler(
                init_scale=128.0, growth_factor=2.0, enabled=enabled, growth_interval=1
            )

            _ = run(data, mod_control, opt_control, scaler, loss_fn, skip_iter, False)
            ret = run(data, mod_scaling, opt_scaling, scaler, loss_fn, skip_iter, True)

            # Allows run() to optionally return a different scaler instance.
            scaler = ret if ret else scaler

            if enabled:
                net_growth = (
                    scaler.get_growth_factor() ** unskipped if unskipped > 0 else 1.0
                )
                net_backoff = (
                    scaler.get_backoff_factor() ** skipped if skipped > 0 else 1.0
                )
                self.assertTrue(
                    scaler.get_scale() == (128.0 * net_growth * net_backoff)
                )
            else:
                self.assertTrue(scaler.get_scale() == 1.0)

            for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
                self.assertTrue(torch.allclose(c, s, atol=atol))

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_grad_scaling_autocast(self):
        try_pickle = False

        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                with torch.mlu.amp.autocast(enabled=try_scaling_api):
                    output = model(input)
                    loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.step(optimizer)
                    scaler.update()
                    if try_pickle:
                        scaler = pickle.loads(pickle.dumps(scaler))
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()
            return scaler

        # sets atol=1e-3 because we're comparing pure fp32 arithmetic vs a mixture of fp16 and fp32
        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-3)
        # this will be picked up by try_pickle within run():
        try_pickle = True
        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-3)

    # TODO(miaochen): mse_backward is unsupported
    @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_grad_scaling_autocast_fp16(self):
        try_pickle = False

        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            model = model.half()
            for i, (input, target) in enumerate(data):
                input = input.half()
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target.half())
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.step(optimizer)
                    scaler.update()
                    if try_pickle:
                        scaler = pickle.loads(pickle.dumps(scaler))
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()
            return scaler

        # sets atol=1e-3 because we're comparing pure fp32 arithmetic vs a mixture of fp16 and fp32
        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-3)
        # this will be picked up by try_pickle within run():
        try_pickle = True
        self._run_scaling_case(run, unskipped=3, skipped=1, atol=1e-3)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_grad_scaling_clipping(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm * scaler.get_scale()
                    )
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_grad_scaling_clipping_separate_unscale(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            max_norm = 0.2
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_grad_scaling_penalty(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            for i, (input, target) in enumerate(data):
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                if try_scaling_api:
                    grad_params = torch.autograd.grad(
                        scaler.scale(loss), model.parameters(), create_graph=True
                    )
                    inv_scale = 1.0 / scaler.get_scale()
                    grad_params = [p * inv_scale for p in grad_params]
                else:
                    grad_params = torch.autograd.grad(
                        loss, model.parameters(), create_graph=True
                    )

                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

                if try_scaling_api:
                    scaler.scale(loss).backward()
                    if i == skip_iter and scaler.is_enabled():
                        model[1].weight.grad.data.fill_(float("inf"))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()

        self._run_scaling_case(run, unskipped=3, skipped=1)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_grad_scaling_accumulation(self):
        def run(data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            iters_to_accumulate = 2
            for i, (input, target) in enumerate(data):
                output = model(input)
                loss = loss_fn(output, target)
                loss = loss / iters_to_accumulate
                if try_scaling_api:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % iters_to_accumulate == 0:
                    if try_scaling_api:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

        self._run_scaling_case(run, unskipped=2, skipped=0)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_grad_scaling_multiple(self):
        for enabled in True, False:
            (
                mod_control0,
                mod_scaling0,
                opt_control0,
                opt_scaling0,
                data,
                loss_fn,
                skip_iter,
            ) = self._create_scaling_case()
            (
                mod_control1,
                mod_scaling1,
                opt_control1,
                opt_scaling1,
            ) = self._create_scaling_models_optimizers()

            scaler = torch.mlu.amp.GradScaler(
                init_scale=128.0, growth_factor=2.0, enabled=enabled, growth_interval=1
            )

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input)
                    output1 = model1(input)
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1, target)
                    loss1 = loss_fn(0.6 * output0 - 0.4 * output1, target)

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            model1[1].weight.grad.data.fill_(float("inf"))

                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)
                        scaler.update()
                    else:
                        loss0.backward(retain_graph=True)
                        loss1.backward()
                        optimizer0.step()
                        if (not scaler.is_enabled()) or (i != skip_iter):
                            optimizer1.step()

            run(mod_control0, mod_control1, opt_control0, opt_control1, False)
            run(mod_scaling0, mod_scaling1, opt_scaling0, opt_scaling1, True)
            self.assertTrue(
                scaler.get_scale()
                == (
                    128.0
                    * scaler.get_growth_factor() ** 3
                    * scaler.get_backoff_factor() ** 1
                )
                if enabled
                else 1.0
            )

            for c, s in zip(
                chain(mod_control0.parameters(), mod_control1.parameters()),
                chain(mod_scaling0.parameters(), mod_scaling1.parameters()),
            ):
                self.assertTrue(torch.allclose(c, s, atol=1e-7))

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not TEST_MULTIMLU, "only one mlu detected")
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_grad_scaling_multimlu(self):
        dev0 = torch.device("mlu:0")
        dev1 = torch.device("mlu:1")

        for enabled in True, False:
            (
                mod_control0,
                mod_scaling0,
                opt_control0,
                opt_scaling0,
                data,
                loss_fn,
                skip_iter,
            ) = self._create_scaling_case()
            (
                mod_control1,
                mod_scaling1,
                opt_control1,
                opt_scaling1,
            ) = self._create_scaling_models_optimizers(device=dev1)

            scaler = torch.mlu.amp.GradScaler(
                init_scale=128.0, growth_factor=2.0, enabled=enabled, growth_interval=1
            )

            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input, target) in enumerate(data):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    output0 = model0(input)
                    output1 = model1(input.to(dev1))
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1.to(dev0), target)
                    loss1 = loss_fn(
                        0.6 * output0.to(dev1) - 0.4 * output1, target.to(dev1)
                    )

                    if try_scaling_api:
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()
                        if i == skip_iter and scaler.is_enabled():
                            model1[1].weight.grad.data.fill_(float("inf"))

                        scaler.unscale_(optimizer0)

                        scaler.step(optimizer0)
                        scaler.step(optimizer1)

                        if scaler.is_enabled():
                            self.assertTrue(
                                len(scaler._found_inf_per_device(optimizer0)) == 1
                            )
                            self.assertTrue(
                                len(scaler._found_inf_per_device(optimizer1)) == 1
                            )
                            self.assertTrue(
                                scaler._found_inf_per_device(optimizer0)[dev0].item()
                                == 0.0
                            )
                            self.assertTrue(
                                scaler._found_inf_per_device(optimizer1)[dev1].item()
                                == float(i == skip_iter)
                            )

                        scaler.update()
                    else:
                        loss0.backward(retain_graph=True)
                        loss1.backward()
                        optimizer0.step()
                        if (not scaler.is_enabled()) or (i != skip_iter):
                            optimizer1.step()

            run(mod_control0, mod_control1, opt_control0, opt_control1, False)
            run(mod_scaling0, mod_scaling1, opt_scaling0, opt_scaling1, True)

            self.assertTrue(
                scaler.get_scale()
                == (
                    128.0
                    * scaler.get_growth_factor() ** 3
                    * scaler.get_backoff_factor() ** 1
                )
                if enabled
                else 1.0
            )

            # Copy mod_control1 and mod_scaling1 back the device 0 for comparison
            mod_control1.to(dev0)
            mod_scaling1.to(dev0)

            for c, s in zip(
                chain(mod_control0.parameters(), mod_control1.parameters()),
                chain(mod_scaling0.parameters(), mod_scaling1.parameters()),
            ):
                self.assertTrue(torch.allclose(c, s, atol=1e-7))

    def _run_autocast(
        self, op, args, run_as_type, out_type=None, module=torch, add_kwargs=None
    ):
        # helper to cast args
        def cast(val, to_type):
            if isinstance(val, torch.Tensor):
                return val.to(to_type) if val.is_floating_point() else val
            elif isinstance(val, collections.abc.Iterable):
                return type(val)(cast(v, to_type) for v in val)
            else:
                return val

        if add_kwargs is None:
            add_kwargs = {}

        self.assertFalse(torch.is_autocast_enabled())
        with torch.mlu.amp.autocast():
            self.assertTrue(torch.mlu.is_autocast_enabled())

            out_type = out_type if out_type is not None else run_as_type
            output = out_method = None

            # Try module.* variant, if requested:
            if module is not None and hasattr(module, op):
                output = getattr(module, op)(*args, **add_kwargs)
                if isinstance(output, torch.Tensor):
                    self.assertTrue(
                        out_type == output.dtype,
                        "autocast for torch.{} produced {}, should produce {}".format(
                            op, output.dtype, out_type
                        ),
                    )

            # Try Tensor.* variant:
            if hasattr(torch.Tensor, op):
                out_method = getattr(args[0], op)(*args[1:], **add_kwargs)
                if isinstance(out_method, torch.Tensor):
                    self.assertTrue(
                        out_type == out_method.dtype,
                        "autocast for torch.{} produced {}, should produce torch.{}".format(
                            op, out_method.dtype, out_type
                        ),
                    )

            self.assertTrue(
                (output is not None) or (out_method is not None),
                "{} not found as an attribute on either Tensor or the requested module {}".format(
                    op, module
                ),
            )

            # If both torch.* and Tensor.* variants were found, check outputs are identical
            if (output is not None) and (out_method is not None):
                self.assertTrue(type(output) == type(out_method))
                comparison = (
                    torch.equal(output, out_method)
                    if isinstance(output, torch.Tensor)
                    else (output == out_method)
                )
                self.assertTrue(
                    comparison,
                    "torch.{0} result did not match Tensor.{0} result".format(op),
                )

            # Compare numerics to Python-side "autocasting" that (we expect) does the same thing
            # as the C++-side autocasting, and should be bitwise accurate.
            compare_ = output if output is not None else out_method
            with torch.mlu.amp.autocast(enabled=False):
                self.assertFalse(torch.is_autocast_enabled())

                if module is not None and hasattr(module, op):
                    control = getattr(module, op)(
                        *cast(args, run_as_type), **add_kwargs
                    )
                else:
                    control = getattr(args[0].to(run_as_type), op)(
                        *cast(args[1:], run_as_type), **add_kwargs
                    )
                self.assertTrue(type(compare_) == type(control))
                comparison = (
                    torch.equal(compare_, control)
                    if isinstance(control, torch.Tensor)
                    else (compare_ == control)
                )
                self.assertTrue(
                    comparison, "torch.{} result did not match control".format(op)
                )
            self.assertTrue(torch.is_autocast_enabled())
        self.assertFalse(torch.is_autocast_enabled())

    # TODO(miaochen): convolution_overrideable is unsupported
    @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_autocast(self):
        # test fp16 op
        for op_with_args in self.autocast_lists.torch_fp16:
            skip_test = False
            op, args = op_with_args[0], op_with_args[1]
            if len(op_with_args) == 3:
                skip_test = op_with_args[2]
            if not skip_test:
                self._run_autocast(op, args, torch.float16)

        # test fp32 op
        def args_maybe_kwargs(op_with_args):
            if len(op_with_args) == 2:
                return op_with_args[0], op_with_args[1], {}
            else:
                return op_with_args[0], op_with_args[1], op_with_args[2]

        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = args_maybe_kwargs(op_with_args)
            self._run_autocast(op, args, torch.float32, add_kwargs=maybe_kwargs)

        # test promote op
        for op, args in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast(op, args, torch.float32)

        # test nn_fp16 op
        for op, args in self.autocast_lists.nn_fp16:
            self._run_autocast(op, args, torch.float16, module=torch._C._nn)

        # test nn_fp32 op
        for op, args in self.autocast_lists.nn_fp32:
            self._run_autocast(op, args, torch.float32, module=torch._C._nn)

        # test methods_fp16 op
        for op, args in self.autocast_lists.methods_fp16:
            self._run_autocast(op, args, torch.float16, module=None)

        # test methods_fp32 op
        for op, args in self.autocast_lists.methods_fp32:
            self._run_autocast(op, args, torch.float32, module=None)

        # test banned op
        with torch.mlu.amp.autocast():
            for op, args, module in self.autocast_lists.banned:
                with self.assertRaises(RuntimeError):
                    getattr(module, op)(*args)

    @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_autocast_torch_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast(op, args, torch.float32, out_type=out_type)

    @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_autocast_methods_expect_builtin_promote(self):
        for op, args, out_type in self.autocast_lists.methods_expect_builtin_promote:
            self._run_autocast(op, args, torch.float32, module=None, out_type=out_type)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_autocast_ignored_types(self):
        with torch.mlu.amp.autocast():
            for ignore_type in (torch.double, torch.int32):
                a_ignore = torch.ones((8, 8), dtype=ignore_type, device="mlu:0")
                b_ignore = torch.ones((8, 8), dtype=ignore_type, device="mlu:0")

                # Tests if CastPolicy::fp16 ops ignore double and int
                # Currently, no ops belonging to this policy support integer inputs.
                if ignore_type is torch.double:
                    with torch.mlu.amp.autocast(enabled=False):
                        type_no_autocast = torch.mm(a_ignore, b_ignore).dtype
                    self.assertTrue(
                        torch.mm(a_ignore, b_ignore).dtype is type_no_autocast
                    )

                # Tests if CastPolicy::fp32_set_opt_dtype ops ignore double and int
                with torch.mlu.amp.autocast(enabled=False):
                    type_no_autocast = torch.sum(a_ignore).dtype
                self.assertTrue(torch.sum(a_ignore).dtype is type_no_autocast)

                # Tests if CastPolicy::fp32_append_dtype ops ignore double and int
                # Currently, no ops belonging to this policy support integer inputs.
                if ignore_type is torch.double:
                    with torch.mlu.amp.autocast(enabled=False):
                        type_no_autocast = torch.norm(a_ignore).dtype
                    self.assertTrue(torch.norm(a_ignore).dtype is type_no_autocast)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_autocast_custom_enabled(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.mlu.amp.custom_fwd
            def forward(ctx, a, b):
                self.assertTrue(a.dtype is torch.float32)
                self.assertTrue(b.dtype is torch.float32)
                self.assertTrue(torch.mlu.is_autocast_enabled())
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.mlu.amp.custom_bwd
            def backward(ctx, grad):
                self.assertTrue(torch.mlu.is_autocast_enabled())
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), a.t().mm(grad)

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="mlu", dtype=torch.float32, requires_grad=True)
        y = torch.randn((8, 8), device="mlu", dtype=torch.float32, requires_grad=True)

        with torch.mlu.amp.autocast():
            output = mymm(x, y)
            self.assertTrue(output.dtype is torch.float16)
            loss = output.sum()
        loss.backward()

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    def test_autocast_custom_cast_inputs(self):
        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.mlu.amp.custom_fwd(cast_inputs=torch.float32)
            def forward(ctx, a, container, expect_type):
                b = container[1][0]
                self.assertTrue(a.dtype is expect_type)
                self.assertTrue(b.dtype is expect_type)
                self.assertFalse(torch.is_autocast_enabled())
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.mlu.amp.custom_bwd
            def backward(ctx, grad):
                self.assertFalse(torch.is_autocast_enabled())
                _, b = ctx.saved_tensors
                return grad.mm(b.t()), None, None

        mymm = MyMM.apply

        x = torch.randn((8, 8), device="mlu", dtype=torch.float16, requires_grad=True)

        y = (
            0,
            {
                0: torch.randn(
                    (8, 8), device="mlu", dtype=torch.float16, requires_grad=False
                )
            },
        )

        with torch.mlu.amp.autocast():
            output = mymm(x, y, torch.float32)
            self.assertTrue(output.dtype is torch.float32)
            loss = output.sum()
        loss.backward()

        # Tests if custom_fwd becomes a no-op when mymm runs outside an autocast-enabled region.
        output = mymm(x, y, torch.float16)
        self.assertTrue(output.dtype is torch.float16)
        loss = output.sum()
        loss.backward()

    # @unittest.skip("not test")
    @testinfo()
    def test_autocast_fp32_set_opt_dtype_policy(self):
        def fn(a, b, c, d, dtype: Optional[int]):
            with torch.autocast("mlu"):
                x = torch.softmax(a, 0)
                y = torch.softmax(b, 0, None)
                z = torch.softmax(c, 0, torch.float64)
                w = torch.softmax(d, 0, dtype)
            return x, y, z, w

        a_fp16 = torch.rand((2, 2), dtype=torch.float16, device="mlu")
        b_fp16 = torch.rand((2, 2), dtype=torch.float16, device="mlu")
        c_fp16 = torch.rand((2, 2), dtype=torch.float16, device="mlu")
        d_fp16 = torch.rand((2, 2), dtype=torch.float16, device="mlu")

        x, y, z, w = fn(a_fp16, b_fp16, c_fp16, d_fp16, torch.float16)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(z.dtype, torch.float64)
        self.assertEqual(w.dtype, torch.float16)

    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_amp_bfloat16(self):
        left = torch.randn((3, 5))
        right = torch.randn((5, 4))
        left_mlu = left.mlu()
        right_mlu = right.mlu()
        with torch.mlu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            result_mlu = torch.mm(left_mlu, right_mlu)
            self.assertEqual(result_mlu.dtype, torch.bfloat16)
            self.assertEqual(result_mlu.device.type, torch.device("mlu").type)
            self.assertEqual(result_mlu.size(), (3, 4))
        result = torch.mm(left, right)
        self.assertTensorsEqual(
            result_mlu.cpu().float(), result.float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
