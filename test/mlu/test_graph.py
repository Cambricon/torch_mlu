import os
import sys
import contextlib
import gc
import threading
import unittest
import warnings
import subprocess
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import TestCase, testinfo


LARGE_BUFFER = 67108864  # 64MiB


def get_mlugraph_segments(pool_id):
    segments = torch.mlu.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] == pool_id]


def get_all_mlugraph_segments():
    segments = torch.mlu.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] != (0, 0)]


def mlugraphify(fn, inputs, pool=None):
    torch.mlu.synchronize()
    stream = torch.mlu.Stream()
    stream.wait_stream(torch.mlu.current_stream())
    with torch.mlu.stream(stream):
        fn(*inputs)
    stream.synchronize()
    torch.mlu.current_stream().wait_stream(stream)
    torch.mlu.synchronize()

    graph = torch.mlu.MLUGraph()
    with torch.mlu.graph(graph, stream=stream, pool=pool):
        static_outputs = fn(*inputs)

    return graph, static_outputs


# Some extra tests for MLU graph
class TestMLUGraph(TestCase):
    def test_allocate_in_thread_to_pool(self):
        def foo():
            return torch.rand([4], device="mlu")

        pool = torch.mlu.graph_pool_handle()
        graph, outputs = mlugraphify(foo, [], pool=pool)
        device = outputs[0].device.index
        del outputs

        @contextlib.contextmanager
        def _use_mlu_memory_pool_manager(device, mem_pool):
            """
            Context manager to use mlu graph pool for new allocations. If you use this manager
            all mlugraph tensors in use should be reflected in the allocator or they will be overwritten.
            existing_graph should already have been used in a capture, and the mem_pool must already exist.
            """
            torch.mlu.synchronize()
            stream = torch.mlu.Stream()
            stream.wait_stream(torch.mlu.current_stream())
            stream_context = torch.mlu.stream(stream)
            stream_context.__enter__()
            torch_mlu._MLUC._mlu_beginAllocateCurrentStreamToPool(device, mem_pool)
            try:
                yield
            finally:
                torch_mlu._MLUC._mlu_endAllocateCurrentStreamToPool(device, mem_pool)
                torch_mlu._MLUC._mlu_releasePool(device, mem_pool)
                stream_context.__exit__(None, None, None)

        segments = get_mlugraph_segments(pool)
        self.assertEqual(len(get_mlugraph_segments(pool)), 1)

        def use_pool():
            def alloc_three():
                a = torch.ones([LARGE_BUFFER // 4], device="mlu")
                b = torch.ones([LARGE_BUFFER // 4], device="mlu")
                c = a + b

            with _use_mlu_memory_pool_manager(device, pool):
                # three allocations
                for _ in range(10):
                    alloc_three()

            # three more allocations not in pool
            alloc_three()

        def no_pool():
            # two allocations
            for _ in range(10):
                a = torch.ones([LARGE_BUFFER // 4], device="mlu")
                b = torch.ones([LARGE_BUFFER // 4], device="mlu")
                del a, b

        graph_thread = threading.Thread(target=use_pool)
        no_graph_thread = threading.Thread(target=no_pool)
        graph_thread.start()
        no_graph_thread.start()

        graph_thread.join()
        no_graph_thread.join()

        self.assertEqual(len(get_mlugraph_segments(pool)), 4)

        del graph

        torch.mlu.synchronize()
        gc.collect()
        torch.mlu.empty_cache()

        self.assertEqual(len(get_mlugraph_segments(pool)), 0)

    def test_graph_memory_stats_and_use_result_after_destroy_graph(self):
        kSmallSize = 16777216
        kSmallBuffer = 33554432
        kLargeBuffer = 67108864
        kMinLargeAlloc = 33554432
        kRoundLarge = 2097152

        elem = 4

        # this was annoying to write but stresses the expectations pretty rigorously
        cases = (
            (512 // elem, 1, kSmallBuffer, kSmallBuffer, "small_pool"),
            (kSmallSize // elem, 2, 2 * kSmallBuffer, kSmallBuffer, "large_pool"),
            ((kSmallSize + 512) // elem, 1, kLargeBuffer, kLargeBuffer, "large_pool"),
            (
                (kMinLargeAlloc - 512) // elem,
                2,
                2 * kLargeBuffer,
                kLargeBuffer,
                "large_pool",
            ),
            (
                (kMinLargeAlloc + 512) // elem,
                3,
                3
                * (
                    kRoundLarge
                    * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge)
                ),
                kRoundLarge * ((kMinLargeAlloc + 512 + kRoundLarge - 1) // kRoundLarge),
                "large_pool",
            ),
        )

        stats_to_check = ("segment.", "reserved_bytes.", "active.", "active_bytes.")

        gc.collect()
        torch.mlu.empty_cache()

        s = torch.mlu.Stream()

        for (
            numel,
            delta_mluMallocs,
            delta_mluMalloc_bytes,
            delta_mluMalloc_bytes_post_del_g,
            pool_string,
        ) in cases:
            if pool_string == "small_pool":
                delta_active_blocks = 3  # one from "b" plus a sneaky two from MLUGraph's one-element rng seed and offset holders
                delta_active_bytes = (
                    numel * elem + 1024
                )  # + 1024 for MLUGraph's rng seed and offset holders each
            else:
                delta_active_blocks = 1  # We only check the large pool, which isn't affected by rng offset holder
                delta_active_bytes = numel * elem

            g = torch.mlu.MLUGraph()
            s.wait_stream(torch.mlu.current_stream())
            with torch.mlu.stream(s):
                # Allocation stat estimates assume input is created on the same stream as capture_begin()
                # (in other words, the same stream silo as the rng offset holder, which is not allocated from the
                # capture's private pool).
                a = torch.ones((numel,), device="mlu")

                precapture_stats = torch.mlu.memory_stats()

                g.capture_begin()
                b = a.clone()
                for _ in range(5):
                    b = b.clone() + 1
                g.capture_end()
            torch.mlu.current_stream().wait_stream(s)

            gc.collect()

            postcapture_stats = torch.mlu.memory_stats()

            expecteds = (
                delta_mluMallocs,
                delta_mluMalloc_bytes,
                delta_active_blocks,
                delta_active_bytes,
            )
            # Double checks replay and stats before and after a call to empty_cache
            for i in range(2):
                for stat, expected in zip(stats_to_check, expecteds):
                    stat = stat + pool_string + ".current"
                    current = postcapture_stats[stat] - precapture_stats[stat]
                    self.assertEqual(
                        current,
                        expected,
                        "Pre to post capture delta of "
                        + stat
                        + f" = {current}, expected = {expected}, numel = {numel}",
                    )

                g.replay()
                self.assertEqual(b.sum().item(), 6 * numel)
                if i == 0:
                    torch.mlu.empty_cache()

            del g
            gc.collect()
            torch.mlu.empty_cache()
            postdel_stats = torch.mlu.memory_stats()

            # Uses graph result b after graph has been deleted
            self.assertEqual(b.sum().item(), 6 * numel)

            # b should be the only live reference remaining from the graph's private pool
            expecteds = (1, delta_mluMalloc_bytes_post_del_g, 1, numel * elem)
            for stat, expected in zip(stats_to_check, expecteds):
                stat = stat + pool_string + ".current"
                current = postdel_stats[stat] - precapture_stats[stat]
                self.assertEqual(
                    current,
                    expected,
                    "Pre capture to post graph delete delta of "
                    + stat
                    + f" = {current}, expected = {expected}, numel = {numel}",
                )

            # del a, b before the next case is essential, otherwise overwriting a and b in the next case
            # can throw off its allocation/deallocation counts.
            del a, b
            # Tensors used across streams (a and b) were held until just now, so no need to call record_stream on them.
            torch.mlu.synchronize()
            torch.mlu.empty_cache()


if __name__ == "__main__":
    unittest.main()
