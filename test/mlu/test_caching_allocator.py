from __future__ import print_function
import os
import sys
import collections
import unittest
import gc
import tempfile
from random import randint
import json

import torch
import torch_mlu
from torch.cuda._memory_viz import profile_plot, _profile_to_snapshot
from torch.cuda._memory_viz import trace_plot
from torch.cuda._memory_viz import segment_plot
from torch.testing._internal.common_utils import IS_WINDOWS, IS_LINUX, parametrize

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase
import logging

logging.basicConfig(level=logging.DEBUG)

TEST_MLU = torch.mlu.is_available()
TEST_MULTIMLU = TEST_MLU and torch.mlu.device_count() >= 2


class TestCachingAllocator(TestCase):
    def _check_memory_stat_consistency(self):
        snapshot = torch.mlu.memory_snapshot()
        expected_each_device = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )

        for segment in snapshot:
            expandable = segment["is_expandable"]
            expected = expected_each_device[segment["device"]]
            pool_str = segment["segment_type"] + "_pool"

            if not expandable:
                expected["segment.all.current"] += 1
                expected["segment." + pool_str + ".current"] += 1

            expected["allocated_bytes.all.current"] += segment["allocated_size"]
            expected["allocated_bytes." + pool_str + ".current"] += segment[
                "allocated_size"
            ]

            expected["reserved_bytes.all.current"] += segment["total_size"]
            expected["reserved_bytes." + pool_str + ".current"] += segment["total_size"]

            expected["active_bytes.all.current"] += segment["active_size"]
            expected["active_bytes." + pool_str + ".current"] += segment["active_size"]

            expected["requested_bytes.all.current"] += segment["requested_size"]
            expected["requested_bytes." + pool_str + ".current"] += segment[
                "requested_size"
            ]

            sum_requested = 0
            is_split = len(segment["blocks"]) > 1
            for block in segment["blocks"]:
                if block["state"] == "active_allocated":
                    expected["allocation.all.current"] += 1
                    expected["allocation." + pool_str + ".current"] += 1

                if block["state"].startswith("active_"):
                    sum_requested += block["requested_size"]
                    expected["active.all.current"] += 1
                    expected["active." + pool_str + ".current"] += 1

                if block["state"] == "inactive" and is_split and not expandable:
                    expected["inactive_split.all.current"] += 1
                    expected["inactive_split." + pool_str + ".current"] += 1
                    expected["inactive_split_bytes.all.current"] += block["size"]
                    expected["inactive_split_bytes." + pool_str + ".current"] += block[
                        "size"
                    ]

            self.assertEqual(sum_requested, segment["requested_size"])

        for device, expected in expected_each_device.items():
            stats = torch.mlu.memory_stats(device)
            for k, v in expected.items():
                self.assertEqual(v, stats[k])

    @staticmethod
    def _test_memory_stats_generator(self, device=None, N=35):
        if device is None:
            device = torch.mlu.current_device()

        m0 = torch.mlu.memory_allocated(device)
        last_m_arr = [torch.mlu.memory_allocated(device)]
        max_m_arr = [torch.mlu.max_memory_allocated(device)]
        last_r_arr = [torch.mlu.memory_reserved(device)]
        max_r_arr = [torch.mlu.max_memory_reserved(device)]

        def alloc(*size):
            with torch.mlu.device(device):
                # NOTE: do **not** use methods that can have additional
                #       memory overhead, e.g., inplace random sampling methods.
                #       they can leave some memory occupied even after being
                #       deallocated, e.g., initialized RNG state, causing some
                #       memory checks below to fail.
                return torch.mlu.FloatTensor(
                    *size
                )  # torch.tensor(*size, dtype=torch.float32, device='mlu')

        def assert_change(comp=1, empty_cache=False, reset_peak=False):
            # comp > 0: increased
            # comp = 0: equal
            # comp < 0: decreased
            new_m = torch.mlu.memory_allocated(device)
            new_max_m = torch.mlu.max_memory_allocated(device)
            if comp > 0:
                self.assertGreater(new_m, last_m_arr[0])
            elif comp < 0:
                self.assertLess(new_m, last_m_arr[0])
            else:
                self.assertEqual(new_m, last_m_arr[0])
            self.assertLessEqual(new_m, new_max_m)
            self.assertGreaterEqual(new_max_m, max_m_arr[0])
            last_m_arr[0] = new_m
            max_m_arr[0] = new_max_m

            new_r = torch.mlu.memory_reserved(device)
            new_max_r = torch.mlu.max_memory_reserved(device)
            # emptying cache may happen (due to allocation or empty_cache), so
            # we can't assert new_c >= last_c
            self.assertLessEqual(new_r, new_max_r)
            self.assertGreaterEqual(new_max_r, max_r_arr[0])
            last_r_arr[0] = new_r
            max_r_arr[0] = new_max_r

            if empty_cache:
                torch.mlu.memory.empty_cache()
                new_r = torch.mlu.memory_reserved(device)
                new_max_r = torch.mlu.max_memory_reserved(device)
                self.assertLessEqual(new_r, last_r_arr[0])
                self.assertLessEqual(new_r, new_max_r)
                self.assertEqual(new_max_r, max_r_arr[0])
                last_r_arr[0] = new_r

            if reset_peak:
                torch.mlu.reset_peak_memory_stats(device)
                self.assertEqual(torch.mlu.memory_allocated(device), last_m_arr[0])
                self.assertEqual(torch.mlu.max_memory_allocated(device), last_m_arr[0])
                max_m_arr[0] = last_m_arr[0]
                self.assertEqual(torch.mlu.memory_reserved(device), last_r_arr[0])
                self.assertEqual(torch.mlu.max_memory_reserved(device), last_r_arr[0])
                max_r_arr[0] = last_r_arr[0]

        assert_change(0)
        assert_change(0, reset_peak=True)
        assert_change(0, empty_cache=True)
        assert_change(0, reset_peak=True)
        assert_change(0)
        yield

        tensors1 = [alloc(1), alloc(10, 20), alloc(200, 300, 2000)]
        m1 = torch.mlu.memory_allocated(device)
        assert_change(1)
        yield

        tensors2 = []

        # small chunks with allocation smaller than 1MB
        for i in range(1, int(N / 2) + 1):
            # small ones
            tensors2.append(alloc(i, i * 4))
            assert_change(1)
            yield

        # large chunks with allocation larger than 1MB
        for i in range(5, int(N / 2) + 5):
            # large ones
            tensors2.append(alloc(i, i * 7, i * 9, i * 11))
            assert_change(1, reset_peak=(i % 2 == 0))
            yield

        tensors2.append(alloc(0, 0, 0))
        assert_change(0)
        yield

        permute = []
        for i in torch.randperm(len(tensors2)):
            permute.append(tensors2[i])
            assert_change(0)
            yield

        del tensors2
        # now the memory of tensor2 is used by permute
        assert_change(0)
        yield
        tensors2 = permute
        assert_change(0)
        yield
        del permute
        # now the memory of permute is used by tensor2
        assert_change(0, reset_peak=True)
        yield

        for i in range(int(N / 2)):
            x = tensors2[i].numel()
            del tensors2[i]
            assert_change(-x)  # in case that tensors2[i] is empty
            yield

        for i in range(2, int(2 * N / 3) + 2):
            tensors2.append(alloc(i, i * 3, i * 8))
            assert_change(1)
            yield

        del tensors2
        assert_change(-1, reset_peak=True)
        assert_change(0)
        self.assertEqual(torch.mlu.memory_allocated(device), m1)
        yield True

        del tensors1
        assert_change(-1, reset_peak=True)
        self.assertEqual(torch.mlu.memory_allocated(device), m0)

        # test empty_cache and reset_peak
        assert_change(0, empty_cache=True)
        assert_change(0, reset_peak=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_memory_stats(self):
        gc.collect()
        torch.mlu.memory.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            self._check_memory_stat_consistency()

    # @unittest.skip("not test")
    @testinfo()
    def test_memory_allocation(self):
        gc.collect()
        torch.mlu.memory.empty_cache()
        mem = None
        size = 1
        prev = 0
        try:
            prev = torch.mlu.memory_allocated()
            mem = torch.mlu.caching_allocator_alloc(size)
            self.assertGreater(torch.mlu.memory_allocated(), prev)
        finally:
            if mem is not None:
                torch.mlu.caching_allocator_delete(mem)
                self.assertEqual(torch.mlu.memory_allocated(), prev)

    @unittest.skipIf(not TEST_MULTIMLU, "only one MLU detected")
    @testinfo()
    def test_memory_stats_multimlu(self):
        # advance a generator with a end flag
        def advance(gen, end):
            if not end:
                try:
                    next(gen)
                except StopIteration:
                    end = True
            return end

        # interlace
        torch.mlu.memory.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device="mlu:0", N=35)
        gen1 = self._test_memory_stats_generator(
            self, device=torch.device("mlu:1"), N=35
        )
        end0 = end1 = False
        while not (end0 and end1):
            end0 = advance(gen0, end0)
            end1 = advance(gen1, end1)

        # semi-random order
        torch.mlu.memory.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device=0, N=35)
        gen1 = self._test_memory_stats_generator(
            self, device=torch.device("mlu:1"), N=35
        )
        end0 = end1 = False

        while not (end0 and end1):
            end0 = advance(gen0, end0)
            if not end0:
                gen1_max_times = torch.LongTensor(1).random_(0, 3)[0]
            else:
                gen1_max_times = torch.inf
            t = 0
            while t < gen1_max_times and not end1:
                end1 = advance(gen1, end1)
                t += 1

    # @unittest.skip("not test")
    @testinfo()
    def test_out_of_memory(self):
        tensor = torch.zeros(1024, device="mlu")

        with self.assertRaisesRegex(RuntimeError, "Tried to allocate 500.00 GiB"):
            torch.empty(1024 * 1024 * 1024 * 500, dtype=torch.int8, device="mlu")

        with self.assertRaisesRegex(
            RuntimeError, "Tried to allocate more than 1EB memory"
        ):
            torch.empty(1024 * 1024 * 1024 * 8000000000, dtype=torch.int8, device="mlu")

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

    # @unittest.skip("not test")
    @testinfo()
    def test_set_per_process_memory_fraction(self):
        # test invalid fraction value.
        with self.assertRaisesRegex(TypeError, "Invalid type"):
            torch.mlu.set_per_process_memory_fraction(int(1))
        with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
            torch.mlu.set_per_process_memory_fraction(-0.1)
        with self.assertRaisesRegex(ValueError, "Invalid fraction value"):
            torch.mlu.set_per_process_memory_fraction(2.0)

        tensor = torch.empty(1024, device="mlu")
        torch.mlu.memory.empty_cache()
        total_memory = torch.mlu.get_device_properties(0).total_memory
        torch.mlu.set_per_process_memory_fraction(0.5, 0)

        # test 0.4 allocation is ok.
        application = int(total_memory * 0.4) - torch.mlu.max_memory_reserved()
        tmp_tensor = torch.empty(application, dtype=torch.int8, device="mlu")
        del tmp_tensor
        torch.mlu.memory.empty_cache()

        application = int(total_memory * 0.5)
        # it will get OOM when try to allocate more than half memory.
        with self.assertRaisesRegex(RuntimeError, "MLU out of memory."):
            torch.empty(application, dtype=torch.int8, device="mlu")

        # ensure out of memory error doesn't disturb subsequent kernel
        tensor.fill_(1)
        self.assertTrue((tensor == 1).all())

    # @unittest.skip("not test")
    @testinfo()
    def test_memory_snapshot(self):
        try:
            torch.mlu.memory.empty_cache()
            torch.mlu.memory._record_memory_history("state", stacks="python")
            # make x the second block in a segment
            torch.rand(2 * 311, 411, device="mlu")
            unused = torch.rand(310, 410, device="mlu")
            x = torch.rand(311, 411, device="mlu")

            # create a bunch of tensors that all will tile into the
            # same segment to  exercise the history merging code
            # 512B is the minimum block size,
            # so we allocate all the tensors to this size to make sure
            # they tile evenly
            tensors = [torch.rand(128, device="mlu") for _ in range(1000)]
            while tensors:
                del tensors[randint(0, len(tensors) - 1)]

            # exercise the history trimming code
            torch.rand(128 * 5, device="mlu")

            ss = torch.mlu.memory._snapshot()
            found_it = False
            for seg in ss["segments"]:
                self.assertTrue("frames" in seg)
                for b in seg["blocks"]:
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertTrue(
                            "test_caching_allocator" in b["frames"][0]["filename"]
                        )
                        found_it = True
                        self.assertEqual(x.untyped_storage().data_ptr(), b["address"])
            self.assertTrue(found_it)
            if not IS_WINDOWS:
                with tempfile.NamedTemporaryFile() as f:
                    torch.mlu.memory._save_segment_usage(f.name)
                    with open(f.name, "r") as f2:
                        self.assertTrue("test_caching_allocator.py" in f2.read())

            del unused
            del x
            torch.mlu.memory.empty_cache()
            ss = torch.mlu.memory._snapshot()
            self.assertTrue(
                ss["device_traces"][0][-1]["action"]
                in ("segment_free", "segment_unmap")
            )

        finally:
            torch.mlu.memory._record_memory_history(None)

    @unittest.skipIf(not IS_LINUX, "cpp contexts are linux only")
    @testinfo()
    def test_memory_snapshot_with_cpp(self):
        try:
            torch.mlu.memory.empty_cache()
            torch.mlu.memory._record_memory_history("state", stacks="all")
            x = torch.rand(311, 411, device="mlu")

            ss = torch.mlu.memory._snapshot()["segments"]
            found_it = False
            for seg in ss:
                for b in seg["blocks"]:
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertTrue("::rand" in str(b["frames"]))
                        found_it = True
            self.assertTrue(found_it)

        finally:
            torch.mlu.memory._record_memory_history(None)

    # @unittest.skip("not test")
    @testinfo()
    def test_memory_profiler_viz(self):
        with torch.profiler.profile(
            with_stack=True, profile_memory=True, record_shapes=True
        ) as prof:
            x = torch.rand(128, 128, device="mlu")
            x * x + x * x
        plot = profile_plot(prof)
        plot = json.dumps(_profile_to_snapshot(prof))
        self.assertTrue("test_caching_allocator.py" in plot)
        self.assertTrue("test_memory_profiler_viz" in plot)
        self.assertTrue("category" in plot)

    @unittest.skipIf(not IS_LINUX, "cpp contexts are linux only")
    @testinfo()
    def test_memory_plots(self):
        for context, stacks in (
            ("all", "all" if IS_LINUX else "python"),
            ("all", "python"),
            (None, "python"),
        ):
            try:
                torch.mlu.memory.empty_cache()
                torch.mlu.memory._record_memory_history(
                    "all", context=context, stacks=stacks
                )

                def run():
                    x = torch.rand(128, 128, device="mlu")
                    x * x + x * x

                run()
                cpp = stacks == "all"
                record_context = context is not None
                ss = torch.mlu.memory._snapshot()

                tplot = trace_plot(ss)
                splot = segment_plot(ss)
                text = json.dumps(ss)

                self.assertTrue(record_context == ("test_memory_plots" in text))
                self.assertTrue(cpp == ("::rand" in text))
                self.assertTrue(str(128 * 128 * 4) in text)

            finally:
                torch.mlu.memory._record_memory_history(None)

    @unittest.skipIf(not IS_LINUX, "cpp contexts are linux only")
    @testinfo()
    def test_memory_plots_free_stack(self):
        for context in ["alloc", "all", "state"]:
            try:
                torch.mlu.memory.empty_cache()
                torch.mlu.memory._record_memory_history(context=context)
                x = None

                def thealloc():
                    nonlocal x
                    x = torch.rand(3, 4, device="mlu")

                def thefree():
                    nonlocal x
                    del x

                thealloc()
                thefree()
                ss = json.dumps(torch.mlu.memory._snapshot())
                self.assertTrue(("thefree" in ss) == (context == "all"))
                self.assertTrue(("thealloc" in ss) == (context != "state"))
            finally:
                torch.mlu.memory._record_memory_history(None)

    # @unittest.skip("not test")
    @testinfo()
    def test_memory_snapshot_script(self):
        try:
            torch.mlu.memory.empty_cache()
            torch.mlu.memory._record_memory_history("state", stacks="python")

            @torch.jit.script
            def foo():
                return torch.rand(311, 411, device="mlu")

            x = foo()

            ss = torch.mlu.memory._snapshot()["segments"]
            found_it = False
            for seg in ss:
                for b in seg["blocks"]:
                    if b["requested_size"] == 311 * 411 * 4:
                        self.assertTrue(b["frames"][0]["name"] == "foo")
                        found_it = True
            self.assertTrue(found_it)

        finally:
            torch.mlu.memory._record_memory_history(None)

    @staticmethod
    def power2_div(size, div_factor):
        pow2 = 1
        while pow2 < size:
            pow2 = pow2 * 2
        if pow2 == size:
            return pow2
        step = pow2 / 2 / div_factor
        ret = pow2 / 2
        while ret < size:
            ret = ret + step
        return ret

    # @unittest.skip("not test")
    @testinfo()
    def test_allocator_settings(self):
        torch.mlu.memory.empty_cache()
        key_allocated = "active_bytes.all.allocated"
        key_requested = "requested_bytes.all.allocated"

        nelems = 21 * 1024 * 1024
        nbytes = 4 * nelems  # floats are 4 bytes

        nelems_big = 100 * 1024 * 1024
        nbytes_big = 4 * nelems_big  # floats are 4 bytes

        start_mem = torch.mlu.memory_stats()[key_allocated]
        torch.mlu.memory._set_allocator_settings("")
        # NOTE: Do not use torch.rand which may include extra memory cost.
        # x = torch.mlu.FloatTensor(nelems)
        x = torch.rand(nelems, device="mlu")

        # test roundup_power2_divisions single value syntax
        reg_mem = torch.mlu.memory_stats()[key_allocated]
        start_requested = torch.mlu.memory_stats()[key_requested]
        torch.mlu.memory._set_allocator_settings("roundup_power2_divisions:4")
        # y = torch.mlu.FloatTensor(nelems)
        y = torch.rand(nelems, device="mlu")

        pow2_div4_mem = torch.mlu.memory_stats()[key_allocated]
        current_requested = torch.mlu.memory_stats()[key_requested]

        self.assertTrue(reg_mem - start_mem == nbytes)
        self.assertTrue(pow2_div4_mem - reg_mem == self.power2_div(nbytes, 4))
        self.assertTrue(current_requested - start_requested == nbytes)

        torch.mlu.memory._set_allocator_settings("garbage_collection_threshold:0.5")
        torch.mlu.memory._set_allocator_settings(
            "garbage_collection_threshold:0.5,max_split_size_mb:70"
        )

        # should have reset the power2 divisions now
        torch.mlu.memory.empty_cache()
        start_mem = torch.mlu.memory_stats()[key_allocated]
        # z = torch.mlu.FloatTensor(nelems)
        z = torch.rand(nelems, device="mlu")
        reg_mem = torch.mlu.memory_stats()[key_allocated]
        self.assertTrue(reg_mem - start_mem == nbytes)

        # roundup_power2_divisions knob array syntax
        torch.mlu.memory.empty_cache()
        torch.mlu.memory._set_allocator_settings(
            "garbage_collection_threshold:0.5,roundup_power2_divisions:[64:8,128:2,256:2,512:2,1024:1,>:1]"
        )
        start_mem = torch.mlu.memory_stats()[key_allocated]
        w = torch.rand(nelems, device="mlu")

        pow2_div8_mem = torch.mlu.memory_stats()[key_allocated]
        self.assertTrue(pow2_div8_mem - start_mem == self.power2_div(nbytes, 8))

        torch.mlu.memory.empty_cache()
        start_mem = torch.mlu.memory_stats()[key_allocated]
        v = torch.rand(nelems_big, device="mlu")

        pow2_div2_mem = torch.mlu.memory_stats()[key_allocated]
        self.assertTrue(pow2_div2_mem - start_mem == self.power2_div(nbytes_big, 2))

        with self.assertRaises(RuntimeError):
            torch.mlu.memory._set_allocator_settings("foo:1,bar:2")

        with self.assertRaises(RuntimeError):
            torch.mlu.memory._set_allocator_settings("garbage_collection_threshold:1.2")

        with self.assertRaises(RuntimeError):
            torch.mlu.memory._set_allocator_settings("max_split_size_mb:2")

    # @unittest.skip("not test")
    @testinfo()
    def test_allocator_env_gpu_migration(self):
        import subprocess

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "roundup_power2_divisions:4"
        test_str = """
import torch
import torch_mlu
torch.mlu.memory.empty_cache()
key_allocated = "active_bytes.all.allocated"
key_requested = "requested_bytes.all.allocated"

nelems = 21 * 1024 * 1024
nbytes = 4 * nelems  # floats are 4 bytes

start_mem = torch.mlu.memory_stats()[key_allocated]
x = torch.rand(nelems, device="mlu")

pow2_div4_mem = torch.mlu.memory_stats()[key_allocated]
start_requested = torch.mlu.memory_stats()[key_requested]
torch.mlu.memory._set_allocator_settings("")
y = torch.rand(nelems, device="mlu")

reg_mem = torch.mlu.memory_stats()[key_allocated]
current_requested = torch.mlu.memory_stats()[key_requested]
print(reg_mem - pow2_div4_mem == nbytes)
print(current_requested - start_requested == nbytes)
print(pow2_div4_mem - start_mem)
print(nbytes)
        """
        res = subprocess.run(
            ["python", "-c", test_str], capture_output=True, env=env, text=True
        )
        output = res.stdout.split()
        self.assertTrue("False" not in output)
        self.assertTrue(int(output[2]) == self.power2_div(int(output[3]), 4))

    # @unittest.skip("not test")
    @testinfo()
    def test_roundup_power2_divisions_0(self):
        torch.mlu.memory.empty_cache()
        key_allocated = "active_bytes.all.allocated"

        nelems = 67 * 1024 * 1024  # 268 MB to fit [256, 512] interval.
        nbytes = 4 * nelems  # floats are 4 bytes

        start_mem = torch.mlu.memory_stats()[key_allocated]
        torch.mlu.memory._set_allocator_settings(
            "roundup_power2_divisions:[128:1,256:0,512:4]"
        )
        y = torch.rand(nelems, device="mlu")

        pow2_div0_mem = torch.mlu.memory_stats()[key_allocated]
        # divisions=0, do not roundup to power2 divsions. i.e. 268 MB.
        self.assertTrue(pow2_div0_mem - start_mem == nbytes)
        # self.assertTrue(pow2_div0_mem - start_mem == self.power2_div(nbytes, 4))

        torch.mlu.memory.empty_cache()
        nelems = 130 * 1024 * 1024  # 520 MB to fit [512, 1024] interval.
        nbytes = 4 * nelems  # floats are 4 bytes
        start_mem = torch.mlu.memory_stats()[key_allocated]
        y = torch.rand(nelems, device="mlu")

        pow2_div2_mem = torch.mlu.memory_stats()[key_allocated]
        # divsions=4, roundup to power2 divisons. i.e. 640 MB.
        self.assertTrue(pow2_div2_mem - start_mem == self.power2_div(nbytes, 4))

    # @unittest.skip("not test")
    @testinfo()
    def test_roundup_power2_divisions_1(self):
        def roundup_512(size):
            if size < 512:
                return 512
            else:
                return 512 * ((size + 512 - 1) // 512)

        torch.mlu.memory.empty_cache()
        key_allocated = "active_bytes.all.allocated"
        key_requested = "requested_bytes.all.allocated"

        nelems = 21 * 125 * 125
        nbytes = 4 * nelems  # floats are 4 bytes

        start_mem = torch.mlu.memory_stats()[key_allocated]
        torch.mlu.memory._set_allocator_settings("")
        # NOTE: Do not use torch.rand which may include extra memory cost.
        x = torch.rand(nelems, device="mlu")

        reg_mem = torch.mlu.memory_stats()[key_allocated]
        start_requested = torch.mlu.memory_stats()[key_requested]
        torch.mlu.memory._set_allocator_settings("roundup_power2_divisions:1")
        y = torch.rand(nelems, device="mlu")

        pow2_div1_mem = torch.mlu.memory_stats()[key_allocated]
        current_requested = torch.mlu.memory_stats()[key_requested]

        self.assertTrue(reg_mem - start_mem == roundup_512(nbytes))
        self.assertTrue(pow2_div1_mem - reg_mem == roundup_512(nbytes))
        self.assertTrue(current_requested - start_requested == nbytes)

    # @unittest.skip("not test")
    @testinfo()
    def test_raises_oom(self):
        # MLUCachingAllocator does early return when searching available blocks
        # if max_split_size_mb is not set
        # Setting this triggers more parts of the code
        torch.mlu.memory._set_allocator_settings("max_split_size_mb:1024")
        torch.mlu.memory.empty_cache()
        with self.assertRaises(torch.mlu.OutOfMemoryError):
            torch.empty(1024 * 1024 * 1024 * 1024, device="mlu")

    # @unittest.skip("not test")
    @testinfo()
    def test_raises_oom_no_max_split_size(self):
        with self.assertRaises(torch.mlu.OutOfMemoryError):
            torch.empty(1024 * 1024 * 1024 * 1024, device="mlu")

    # @unittest.skip("not test")
    @testinfo()
    def test_caching_allocator_record_stream_oom(self):
        """allocations delayed by a record_stream call should still be freed on
        an out-of-memory in cnrt_malloc_retry."""
        stream = torch.mlu.Stream()

        with torch.mlu.stream(stream):
            y = torch.zeros(40 * 1024 * 1024, device="mlu")

        for _ in range(100):
            x = torch.empty(40 * 1024 * 1024, device="mlu")
            with torch.mlu.stream(stream):
                y += x
            # delays re-use of `x` until after all operations in `stream`
            x.record_stream(stream)
            del x

        # we've made a mess by allocating up to the device capacity. free any
        # cached blocks in case it affects future tests.
        torch.mlu.memory.empty_cache()


if __name__ == "__main__":
    unittest.main()
