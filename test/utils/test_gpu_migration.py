import sys
import os
import unittest
import logging
import warnings
import expecttest
import functools
import types
import importlib
from pathlib import Path
import yaml
import re

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

import torch_mlu
import torch.nn as nn
import torch_mlu.utils.gpu_migration

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import run_tests, testinfo

logging.basicConfig(level=logging.DEBUG)


def process_ipc(
    outq,
    device,
    handle,
    storage_size_bytes,
    storage_offset_bytes,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    storage = torch.TypedStorage._new_shared_cuda(
        device,
        handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    )
    outq.put((storage.tolist(), storage.device.type))


def get_docstring_from_key(key):
    try:
        parts = key.split(".")
        obj = importlib.import_module(parts[0])

        for attr in parts[1:]:
            obj = getattr(obj, attr)

        return obj.__doc__ or f"No docstring found for {key}"
    except ModuleNotFoundError as e:
        return f"Module '{parts[0]}' not found: {e}"
    except AttributeError as e:
        return f"Attribute '{attr}' not found in '{key}': {e}"
    except Exception as e:
        return f"An error occurred while accessing {key}: {e}"


# This func is used to check whether 'device' arg is in Args/Kwargs in a torch native func
def contains_device_parameter(text):
    args_pattern = re.compile(
        r"(?<=Args:\n)([\s\S]*?)(?=\n\s*\n|Keyword args:|\Z)", re.DOTALL
    )

    keyword_args_pattern = re.compile(
        r"(?<=Keyword args:\n)([\s\S]*?)(?=\n\n\s*\w+:|\Z)", re.DOTALL
    )

    args_match = args_pattern.search(text)
    if args_match:
        args_text = args_match.group(0)
        if (
            "device" in args_text and "device_id" not in args_text
        ) or "use_cuda" in args_text:
            return True

    keyword_args_match = keyword_args_pattern.search(text)
    if keyword_args_match:
        keyword_args_text = keyword_args_match.group(0)
        if (
            "device" in keyword_args_text and "device_id" not in keyword_args_text
        ) or "use_cuda" in keyword_args_text:
            return True

    return False


# If you directly inherit common_utils.TestCase, you will encounter some device-related
# code hijacking failure problems, but in fact there is no need to hijack.
class TestGpuMigration(expecttest.TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_amp(self):
        self.assertEqual(torch.cuda.amp.autocast, torch.mlu.amp.autocast)
        self.assertEqual(torch.cuda.amp.common, torch.mlu.amp.common)
        self.assertEqual(torch.cuda.amp.GradScaler, torch.mlu.amp.GradScaler)
        grad_scaler = torch.amp.GradScaler(device="cuda")
        self.assertEqual(grad_scaler._device, "mlu")

        with torch.cuda.amp.autocast(enabled=True):
            m0 = torch.randn(2, 3).cuda()
            m1 = torch.randn(2, 4).cuda()
            m2 = torch.randn(4, 3).cuda()
            out = torch.addmm(m0, m1, m2)
            self.assertTrue(out.is_mlu)

        with torch.amp.autocast(device_type="cuda"):
            m0 = torch.randn(2, 3).cuda()
            m1 = torch.randn(2, 4).cuda()
            m2 = torch.randn(4, 3).cuda()
            out = torch.addmm(m0, m1, m2)
            self.assertTrue(out.is_mlu)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            assert torch.get_autocast_gpu_dtype() == torch.float32
            torch.set_autocast_gpu_dtype(torch.float16)
            assert torch.get_autocast_gpu_dtype() == torch.float16

        # test is_autocast_enabled
        def _cast_if_autocast_enabled(*args):
            if not torch.is_autocast_enabled():
                return args
            else:
                return torch.cuda.amp.autocast_mode._cast(
                    args, torch.get_autocast_gpu_dtype()
                )

        with torch.autocast(device_type="cuda"):
            a = torch.randn(1, 2, device="cuda")
            out = _cast_if_autocast_enabled(a)
            assert out[0].dtype == torch.float16

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(torch.cuda.device_count() < 2, "need at least 2 GPU")
    def test_device(self):
        x = torch.randn(10, device="cuda")
        self.assertTrue(x.is_mlu)

        device = torch.device(f"cuda:0")
        x = torch.randn(10, device=device)
        self.assertTrue(x.is_mlu)
        x = torch.randint(0, 2, (10,), device=device)
        self.assertTrue(x.is_mlu)

        y = x.to(1)
        self.assertTrue(y.is_mlu)
        self.assertTrue(y.device.index == 1)

        x = torch.randn(10).cuda()
        self.assertTrue(x.is_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_device_context(self):
        with torch.device("cuda"):
            x = torch.randn(10)
        self.assertTrue(x.is_mlu)
        with torch.device("mlu"):
            y = torch.zeros(10)
        self.assertTrue(y.is_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_set_autocast_dtype(self):
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        self.assertTrue(torch.get_autocast_dtype("mlu"), torch.bfloat16)
        self.assertTrue(torch.get_autocast_dtype("cuda"), torch.bfloat16)
        self.assertTrue(torch.mlu.get_autocast_dtype(), torch.bfloat16)

        torch.mlu.set_autocast_dtype(torch.float16)
        self.assertTrue(torch.get_autocast_dtype("mlu"), torch.float16)
        self.assertTrue(torch.get_autocast_dtype("cuda"), torch.float16)
        self.assertTrue(torch.mlu.get_autocast_dtype(), torch.float16)

        torch.set_autocast_dtype("mlu", torch.bfloat16)
        self.assertTrue(torch.get_autocast_dtype("mlu"), torch.bfloat16)
        self.assertTrue(torch.get_autocast_dtype("cuda"), torch.bfloat16)
        self.assertTrue(torch.mlu.get_autocast_dtype(), torch.bfloat16)

    # @unittest.skip("not test")
    @testinfo()
    def test_tensor(self):
        self.assertEqual(torch.Tensor.cuda, torch.Tensor.mlu)
        self.assertEqual(torch.Tensor.is_cuda, torch.Tensor.is_mlu)
        x = torch.tensor((), dtype=torch.int32, device="cuda")
        y = x.new_ones((2, 3), device="cuda")
        self.assertTrue(y.is_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_module(self):
        m = torch.nn.Linear(3, 10)
        m.cuda()
        self.assertTrue(m.weight.is_mlu)

        m = torch.nn.Linear(4, 10, device="cuda")
        self.assertTrue(m.weight.is_mlu)

        m = torch.nn.Linear(3, 10).to("cuda")
        self.assertTrue(m.weight.is_mlu)

        m = torch.nn.Linear(3, 10).to_empty(device="cuda")
        self.assertTrue(m.weight.is_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_cuda(self):
        y = torch.cuda.get_device_name(torch.cuda.current_device())
        self.assertIn("MLU", y)

        self.assertEqual(torch.cuda.FloatTensor, torch.mlu.FloatTensor)
        self.assertEqual(torch.cuda.FloatStorage, torch.mlu.FloatStorage)

        self.assertTrue(torch.cuda._is_compiled())
        self.assertEqual(
            torch._C._cuda_getCurrentRawStream, torch_mlu._MLUC._mlu_getCurrentRawStream
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_replace_aten_fn(self):
        out = torch.ops.aten.arange.start_step(
            0, 8192, 4096, dtype=torch.int32, device=torch.device("cuda")
        )
        self.assertTrue(out.is_mlu)
        input = torch.empty(2, 3)
        out = torch.ops.aten.ones_like(input, device=torch.device("cuda"))
        self.assertTrue(out.is_mlu)
        out = torch.ops.aten.ones_like.default(input, device=torch.device("cuda"))
        self.assertTrue(out.is_mlu)

    @unittest.skip("not test")  # PYTORCH-11776
    @testinfo()
    def test_cuda_graph(self):
        g = torch.cuda.CUDAGraph()

        # Placeholder input used for capture
        static_input = torch.empty((5,), device="cuda")

        # Warmup before capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_output = static_input * 2
        torch.cuda.current_stream().wait_stream(s)

        # Captures the graph
        with torch.cuda.graph(g, capture_error_mode="relaxed"):
            static_output = static_input * 2

        # Fills the graph's input memory with new data to compute on
        static_input.copy_(torch.full((5,), 3, device="cuda"))
        g.replay()
        # static_output holds the results
        self.assertEqual(static_output, torch.full((5,), 6.0, device="cuda"))

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipIf(torch.mlu.device_count() < 2, "need at least 2 MLU")
    def test_distributed(self):
        dist_url = "tcp://127.0.0.1:65501"
        world_size = 1
        rank = 0
        dist.init_process_group(
            backend="nccl", init_method=dist_url, world_size=world_size, rank=rank
        )
        self.assertTrue(dist.get_backend() == "cncl")
        self.assertTrue(dist.is_cncl_available())
        self.assertEqual(dist.ProcessGroupNCCL, dist.ProcessGroupCNCL)
        self.assertEqual(c10d.ProcessGroupNCCL, torch_mlu._MLUC.ProcessGroupCNCL)
        process_group = c10d._get_default_group()
        backend = process_group._get_backend(torch.device("cuda"))
        self.assertTrue(isinstance(backend, torch_mlu._MLUC.ProcessGroupCNCL))

        new_pg1 = dist.new_group(backend="nccl")
        self.assertTrue(dist.get_backend(new_pg1) == "cncl")
        new_pg2 = c10d.new_group(backend="nccl")
        self.assertTrue(dist.get_backend(new_pg2) == "cncl")

        dist.destroy_process_group()

    # @unittest.skip("not test")
    @testinfo()
    def test_fsdp(self):
        # test sharded_grad_scaler
        sharded_grad_scaler = ShardedGradScaler()
        self.assertTrue(sharded_grad_scaler._device == "mlu")
        sharded_grad_scaler = ShardedGradScaler(device="cuda")
        self.assertTrue(sharded_grad_scaler._device == "mlu")
        sharded_grad_scaler = ShardedGradScaler("cuda")
        self.assertTrue(sharded_grad_scaler._device == "mlu")

    # @unittest.skip("not test")
    @testinfo()
    def test_profiler(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(16, 33, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(243, 243),
            torch.nn.ReLU(),
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            x = torch.randn(40, 16, 18, 260).cuda()
            model.cuda()
            model(x)
        output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        self.assertIn("MLU Mem", output)
        output = prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=-1
        )
        self.assertIn("MLU Mem", output)

    # @unittest.skip("not test")
    @testinfo()
    def test_dataloader(self):
        class MyIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, start, end):
                super(MyIterableDataset).__init__()
                assert end > start, "this example code only works with end >= start"
                self.start = start
                self.end = end

            def __iter__(self):
                return iter(range(self.start, self.end))

        ds = MyIterableDataset(start=3, end=7)
        loader = torch.utils.data.DataLoader(ds, num_workers=0, pin_memory=True)
        self.assertIn("mlu", loader.pin_memory_device)

    # @unittest.skip("not test")
    @testinfo()
    def test_generator(self):
        g1 = torch.Generator(device="cuda")
        g1.manual_seed(12)
        self.assertIn("mlu", g1.device.type)
        g2 = torch.Generator(device="cuda")
        g2.set_state(g1.get_state())
        self.assertTrue(torch.allclose(g1.get_state(), g2.get_state(), 0.0, 0.0))
        self.assertTrue(isinstance(g2, torch.Generator))
        x = torch.randn(5, generator=torch.Generator(device="cuda"), device="cuda")
        self.assertIn("mlu", x.device.type)

    # @unittest.skip("not test")
    @testinfo()
    def test_random(self):
        state1 = torch.cuda.get_rng_state(device="cuda")
        state2 = torch.mlu.get_rng_state(device="mlu")
        self.assertTrue(torch.allclose(state1, state2, 0.0, 0.0))

    # @unittest.skip("not test")
    @testinfo()
    def test_storage(self):
        x = torch.randn(10).storage()
        y = x.cuda()
        self.assertTrue(y.is_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_jit_script(self):
        @torch.jit.script
        def foo(x, y):
            if x.max() > y.max():
                r = x
            else:
                r = y
            return r

        out = foo(torch.ones(2, 2).cuda(), torch.ones(2, 2).cuda())
        self.assertTrue(out.is_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_nvtx(self):
        self.assertEqual(torch.cuda.nvtx, torch.mlu.cnpx)
        torch.cuda.nvtx.range_push("foo")
        torch.cuda.nvtx.mark("bar")
        torch.cuda.nvtx.range_pop()
        range_handle = torch.cuda.nvtx.range_start("range_start")
        torch.cuda.nvtx.range_end(range_handle)

    # @unittest.skip("not test")
    @testinfo()
    def test_load(self):
        m = nn.Linear(3, 5).cuda()
        torch.save(m, "m.pth")
        m_load = torch.load("m.pth", map_location=torch.device("cuda", 0))
        self.assertTrue(m_load.weight.is_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_save_and_load(self):
        def _new_privateuse1_tag(obj):
            origin_privateuse1_tag = functools.partial(
                torch.serialization._backend_tag, "privateuse1"
            )
            backend_name = origin_privateuse1_tag(obj)
            if backend_name.startswith("mlu"):
                backend_name = backend_name.replace("mlu", "cuda")
            return backend_name

        # simulate cuda to write data to pkl file.
        torch.serialization.register_package(
            1,
            _new_privateuse1_tag,
            functools.partial(torch.serialization._deserialize, "privateuse1"),
        )
        m = nn.Linear(3, 5).mlu()
        torch.save(m, "m.pth")
        m_load = torch.load("m.pth")
        del torch.serialization._package_registry[0]
        self.assertTrue(m_load.weight.is_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_torch_cuda_nccl(self):
        self.assertEqual(torch.cuda.nccl, torch.mlu.cncl)

    @testinfo()
    def test_emit_nvtx(self):
        self.assertEqual(
            torch.autograd.profiler.emit_nvtx, torch.autograd.profiler.emit_cnpx
        )

        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return x

        def custom_layer(input_ten):
            return MyFunc.apply(input_ten)

        with torch.autograd.profiler.emit_nvtx(record_shapes=True) as prof:
            x = torch.randn(10, 10, requires_grad=True, device="cuda")
            y = torch.randn(10, 10, requires_grad=True, device="cuda")
            z = x + y
            s = custom_layer(z)
            q = s.sum()
            q.backward()

    @testinfo()
    def test_torch_backends_cuda(self):
        self.assertEqual(torch.backends.cuda, torch.backends.mlu)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=False, enable_mem_efficient=False
        ):
            size = (2, 3, 4)
            dtype = torch.float16
            device = "mlu"
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaisesRegex(
                RuntimeError,
                "No viable backend for scaled_dot_product_attention was found.",
                lambda: torch._fused_sdp_choice(q, k, v),
            )
            self.assertRaisesRegex(
                RuntimeError,
                "No viable backend for scaled_dot_product_attention was found.",
                lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
            )

    @testinfo()
    def test_model_transfer_deprecated_warning(self):
        with warnings.catch_warnings(record=True) as w:
            # # Cause all FutureWarning to always be triggered.
            warnings.simplefilter("always", FutureWarning)
            import torch_mlu.utils.model_transfer

        self.assertRegex(
            str(w[-1].message),
            r"`torch_mlu.utils.model_transfer` is deprecated. "
            "Please use `torch_mlu.utils.gpu_migration` instead.",
        )

    @testinfo()
    def test_apply_monkey_patches_multi_times(self):
        torch_mlu.utils.gpu_migration.apply_monkey_patches()
        torch_mlu.utils.gpu_migration.apply_monkey_patches()

    @testinfo()
    def test_storage_ipc_default(self):
        self.assertEqual(
            torch.UntypedStorage._release_ipc_counter_cuda,
            torch.UntypedStorage._release_ipc_counter_mlu,
        )
        self.assertEqual(
            torch.UntypedStorage._share_cuda_, torch.UntypedStorage._share_mlu_
        )
        self.assertEqual(
            torch.UntypedStorage._new_shared_cuda, torch.UntypedStorage._new_shared_mlu
        )
        self.assertEqual(
            torch.TypedStorage._release_ipc_counter,
            torch.TypedStorage._release_ipc_counter_mlu,
        )
        self.assertEqual(
            torch.TypedStorage._share_cuda_, torch.TypedStorage._share_mlu_
        )
        self.assertEqual(
            torch.TypedStorage._new_shared_cuda, torch.TypedStorage._new_shared_mlu
        )
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        outq = ctx.Queue()
        origin_storage = torch.randn(2, 2).cuda()._typed_storage()
        list_origin = origin_storage.untyped().tolist()
        device_origin = origin_storage.device.type
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = origin_storage._share_cuda_()

        p = ctx.Process(
            target=process_ipc,
            args=(
                outq,
                device,
                handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )
        p.start()
        process_storage, process_device = outq.get()
        p.join(2)
        self.assertEqual(torch.randn(2, 2).mlu().device.type, device_origin)
        self.assertEqual(list_origin, process_storage)
        self.assertEqual(device_origin, process_device)

    @testinfo()
    def test_cuda_default(self):
        # test torch.amp.GradScaler()
        default_scaler = torch.amp.GradScaler()
        default_scaler_cuda = torch.amp.GradScaler(device="cuda")
        self.assertEqual("mlu", default_scaler._device)
        self.assertEqual("mlu", default_scaler_cuda._device)

        # test torch.autograd.graph.save_on_cpu
        a = torch.randn(5, requires_grad=True, device="cuda")
        soc = torch.autograd.graph.save_on_cpu(pin_memory=True)
        origin_device, soc_tensor = soc.pack_hook(a)
        unpack_tensor = soc.unpack_hook((origin_device, soc_tensor))
        self.assertEqual("mlu", a.device.type)
        self.assertEqual(a.device, origin_device)
        self.assertEqual(a.device, unpack_tensor.device)
        self.assertTrue(torch.equal(a.detach(), unpack_tensor.detach()))
        self.assertEqual("cpu", soc_tensor.device.type)
        self.assertTrue(soc_tensor.is_pinned)

        # test torch.random.fork_rng
        before_fork_rng = torch.mlu.get_rng_state()
        with torch.random.fork_rng(enabled=True):
            torch.mlu.manual_seed(66)
            fork_rng = torch.mlu.get_rng_state()
        after_fork_rng = torch.mlu.get_rng_state()
        self.assertTrue(torch.equal(before_fork_rng, after_fork_rng))
        self.assertTrue(not torch.equal(before_fork_rng, fork_rng))

    @testinfo()
    def test_profiler(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(16, 33, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(243, 243),
            torch.nn.ReLU(),
        )
        test_apis = [
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
            ),
            torch.profiler.profile(
                use_cuda=True, record_shapes=True, profile_memory=True, with_flops=True
            ),
            torch.autograd.profiler.profile(
                use_cuda=True, record_shapes=True, profile_memory=True, with_flops=True
            ),
        ]
        for profiler_api in test_apis:
            with profiler_api as prof:
                x = torch.randn(40, 16, 18, 260).cuda()
                model.cuda()
                model(x)
            output = prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1
            )
            self.assertIn("MLU Mem", output)
            output = prof.key_averages().table(
                sort_by="self_cuda_memory_usage", row_limit=-1
            )
            self.assertIn("MLU Mem", output)

    @testinfo()
    def test_cuda_default_profiler(self):
        # torch.autograd.profiler.profile()
        self.assertRaisesRegex(
            TypeError,
            "profile.__init__\(\) takes from 1 to 2 positional arguments but 3 were given",
            lambda: torch.autograd.profiler.profile(True, True),
        )
        prof = torch.autograd.profiler.profile(use_cuda=True)
        self.assertEqual(prof.use_device, "mlu")
        self.assertEqual(prof.use_cuda, False)

        # torch.profiler.profile()
        prof = torch.profiler.profile(use_cuda=True)
        self.assertTrue(torch.profiler.ProfilerActivity.PrivateUse1 in prof.activities)

        # torch.autograd.profiler_legacy.profile()
        prof = torch.autograd.profiler_legacy.profile(use_cuda=True)
        self.assertEqual(prof.profiler_kind, torch.autograd.ProfilerState.PRIVATEUSE1)

    @testinfo()
    def test_pluggable_alloc(self):
        self.assertEqual(
            torch.cuda.memory.CUDAPluggableAllocator,
            torch.mlu.memory.MLUPluggableAllocator,
        )

    # Remove/modify the following cases after supported APIs used in below cases
    @testinfo()
    def test_update_dict_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            try:
                torch.cuda.memory_usage()
            except:
                pass
        self.assertRegex(
            str(w[-1].message),
            r"gpu_migration: memory_usage is not yet suppoorted on MLU",
        )
        with warnings.catch_warnings(record=True) as w:
            try:
                tmp = torch.cuda.DeferredCudaCallError()
            except:
                pass
        self.assertRegex(
            str(w[-1].message),
            r"gpu_migration: DeferredCudaCallError is not yet suppoorted on MLU",
        )

    # Use this test to check whether lists in gpu_migration contains all supported functions in api_support_list and require device
    # Notice: This func cannot cover all funcs reqire 'device' arg, funcs without native
    # doc will not be covered, therefore the lists in gpu_migration are still necessary
    @testinfo()
    def test_gpu_migration_lists(self):
        torch_fn_list = torch_mlu.utils.gpu_migration.migration.torch_fn_list
        tensor_fn_list = torch_mlu.utils.gpu_migration.migration.tensor_fn_list
        module_fn_list = torch_mlu.utils.gpu_migration.migration.module_fn_list
        default_cuda_args_list = (
            torch_mlu.utils.gpu_migration.migration.default_cuda_args_list
        )
        class_list = torch_mlu.utils.gpu_migration.migration.class_list
        distributed_fn_list = (
            torch_mlu.utils.gpu_migration.migration.distributed_fn_list
        )
        other_fn_list = torch_mlu.utils.gpu_migration.migration.other_fn_list

        # api_names here whose docstrings contain "device" but not has device arg/kwarg
        # or classes whose names are different from its full dir
        white_list = [
            "torch.nn.parallel.DistributedDataParallel",
            "torch.testing.assert_close",
            "torch.utils.checkpoint.checkpoint",
            "torch.utils.data.DataLoader",
            "torch.nn.utils.clip_grad_norm_",
        ]
        torch_mlu_path = Path(__file__).resolve().parent.parent.parent
        torch_api_path = (
            str(torch_mlu_path)
            + "/tools/autogen_torch_api/api_support_lists/torch_api.yaml"
        )
        with open(torch_api_path, "r", encoding="utf-8") as f:
            api_support_data = yaml.safe_load(f)
        for i, module in enumerate(api_support_data):
            module_name = list(module.keys())[0]
            for j, api in enumerate(module[module_name]):
                api_name = api[list(api.keys())[0]]
                supported = False
                for k in api:
                    if k == "supported":
                        supported = api[k]
                if (
                    supported
                    and not api_name.startswith("torch.cuda")
                    and not api_name.startswith("torch.backends.cuda")
                    and not api_name.startswith("torch.Tensor.cuda")
                    and not api_name.startswith("torch.profiler")
                ):
                    docstring = get_docstring_from_key(api_name)
                    requires_device = contains_device_parameter(docstring)
                    if requires_device:
                        inList = False
                        for l in [
                            torch_fn_list,
                            tensor_fn_list,
                            module_fn_list,
                            default_cuda_args_list,
                            distributed_fn_list,
                        ]:
                            if api_name.rsplit(".", 1)[-1] in l:
                                inList = True
                        for fn in other_fn_list:
                            if api_name == fn["name"]:
                                inList = True
                        for cls in class_list:
                            if (
                                f"{cls.__module__}.{cls.__name__}" == api_name
                                or api_name in white_list
                            ):
                                inList = True
                        self.assertEqual(
                            inList,
                            True,
                            msg=f"GPU Migration: {api_name} has a device arg/kwargs but not in func list, please add it to gpu migration manually.",
                        )

    @testinfo()
    def test_lazymodule_nn_function(self):
        a = torch.nn.LazyConvTranspose3d(
            out_channels=16, kernel_size=3, stride=2, padding=1
        )
        a.to(torch.float)
        a.cuda("cuda:0")
        a.cpu().mlu()
        self.assertTrue(
            torch.Tensor.to
            in torch.nn.parameter.UninitializedTensorMixin._allowed_methods
        )


if __name__ == "__main__":
    run_tests()
