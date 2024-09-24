import sys
import os
import unittest
import logging
import warnings
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch_mlu
import torch.nn as nn
import torch_mlu.utils.gpu_migration

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import run_tests, TestCase, testinfo

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


class TestGpuMigration(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_amp(self):
        self.assertEqual(torch.cuda.amp.autocast, torch.mlu.amp.autocast)
        self.assertEqual(torch.cuda.amp.common, torch.mlu.amp.common)
        self.assertEqual(torch.cuda.amp.GradScaler, torch.mlu.amp.GradScaler)

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

        from torch.cuda.amp import autocast

        with autocast(enabled=True):
            rnn = nn.LSTM(10, 20, 2).cuda()
            input = torch.randn(5, 3, 10).cuda().half()
            h0 = torch.randn(2, 3, 20).cuda()
            c0 = torch.randn(2, 3, 20).cuda()
            output, (hn, cn) = rnn(input, (h0, c0))

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

        # test sharded_grad_scaler
        self.assertEqual(
            torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler,
            torch_mlu.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler,
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_profiler(self):
        self.assertTrue(
            torch.profiler.ProfilerActivity.CUDA == torch.profiler.ProfilerActivity.MLU
        )
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
        self.assertEqual(g1.get_state(), g2.get_state())
        self.assertTrue(isinstance(g2, torch.Generator))
        x = torch.randn(5, generator=torch.Generator(device="cuda"), device="cuda")
        self.assertIn("mlu", x.device.type)

    # @unittest.skip("not test")
    @testinfo()
    def test_random(self):
        state1 = torch.cuda.get_rng_state(device="cuda")
        state2 = torch.mlu.get_rng_state(device="mlu")
        self.assertEqual(state1, state2)

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
            backend_name = torch.serialization._privateuse1_tag(obj)
            if backend_name.startswith("mlu"):
                backend_name = backend_name.replace("mlu", "cuda")
            return backend_name

        # simulate cuda to write data to pkl file.
        torch.serialization.register_package(
            1, _new_privateuse1_tag, torch.serialization._privateuse1_deserialize
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
            torch.mlu.UntypedStorage._release_ipc_counter_mlu,
        )
        self.assertEqual(
            torch.UntypedStorage._share_cuda_, torch.mlu.UntypedStorage._share_mlu_
        )
        self.assertEqual(
            torch.UntypedStorage._new_shared_cuda,
            torch.mlu.UntypedStorage._new_shared_mlu,
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
    def test_cuda_default_profiler(self):
        # torch.autograd.profiler.profile()
        self.assertRaisesRegex(
            TypeError,
            "__init__\(\) takes from 1 to 2 positional arguments but 3 were given",
            lambda: torch.autograd.profiler.profile(True, True),
        )
        prof = torch.autograd.profiler.profile(use_cuda=True)
        self.assertEqual(prof.use_mlu, True)
        self.assertEqual(prof.use_cuda, False)

        # torch.profiler.profile()
        prof = torch.profiler.profile(use_cuda=True)
        self.assertTrue(torch.profiler.ProfilerActivity.MLU in prof.activities)

        # torch.autograd.profiler_legacy.profile()
        msg = "MLU not support torch.autograd.profiler_legacy.profile\(\), and this class will be deprecated. Use torch.profiler instead"
        with self.assertRaisesRegex(NotImplementedError, msg):
            prof = torch.autograd.profiler_legacy.profile(use_cuda=True)


if __name__ == "__main__":
    run_tests()
