import unittest
import sys
import os
import numpy as np
import subprocess
import ctypes

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

import torch


class PluggableAllocatorTestCases(TestCase):
    # @unittest.skip("not test")
    def test_pluggable_allocator(self):
        from torch_mlu.utils.cpp_extension import include_paths, library_paths

        dummy_allocator_source = """
        #include <torch/extension.h>
        #include "aten/utils/exceptions.h"
        #include "framework/core/caching_allocator.h"
        #include "cnrt.h"

        extern "C" {
          int called_dummy_alloc = 0;
          int called_dummy_free = 0;
          bool is_init = false;
          bool enable_empty_cache = false;
          int memory_fraction = 0;
          int alloc_size = 0;
          bool enable_record_stream = false;
          bool enable_alloc_pool = false;
          bool end_alloc_pool = false;
          bool release_pool = false;

          void* dummy_alloc(size_t size, int device, void* stream) {
            called_dummy_alloc = 123;
            void* ptr;
            TORCH_CNRT_CHECK(cnrtMalloc(&ptr, size));
            return ptr;
          }

          void dummy_free(void* ptr, size_t size, int device, void* stream) {
            called_dummy_free = 321;
            TORCH_CNRT_CHECK(cnrtFree(ptr));
          }

          void init(int device_count) {
            is_init = true;
          }

          void empty_cache() {
            enable_empty_cache = true;
          }

          void memory_fragmentation_memory_fraction(double fraction, int device) {
            memory_fraction = 111;
          }

          void* base_alloc(void* ptr, size_t* size) {
            alloc_size = 222;
            return ptr;
          }

          void record_stream(void* ptr, cnrtQueue_t stream) {
            enable_record_stream = true;
          }

          void begin_alloc_to_pool(c10::DeviceIndex device,
                                   torch_mlu::MLUCachingAllocator::MempoolId_t mempool_id,
                                   std::function<bool(cnrtQueue_t)> filter) {

            enable_alloc_pool = true;
          }

          void end_alloc_to_pool(c10::DeviceIndex device,
                                 torch_mlu::MLUCachingAllocator::MempoolId_t mempool_id) {
            end_alloc_pool = true;
          }

          void release_alloc_pool(c10::DeviceIndex device,
                                  torch_mlu::MLUCachingAllocator::MempoolId_t mempool_id) {
            release_pool = true;
          }
        }
        """
        # currently, not support torch_mlu.utils.cpp_extension.load_inline.
        extra_ldflags = ["-ltorch_mlu", "-lcnrt"]
        for path in library_paths():
            extra_ldflags.append("-L" + path)
        dummy_allocator_libname = "dummy_allocator"
        dummy_allocator = torch.utils.cpp_extension.load_inline(
            name=dummy_allocator_libname,
            cpp_sources=dummy_allocator_source,
            extra_include_paths=include_paths(),
            extra_ldflags=extra_ldflags,
            is_python_module=False,
            keep_intermediates=False,
            verbose=True,
        )
        if dummy_allocator is None:
            dummy_allocator_path = torch.utils.cpp_extension._get_build_directory(
                dummy_allocator_libname, True
            )
            dummy_allocator = os.path.join(
                dummy_allocator_path,
                f"{dummy_allocator_libname}{torch.utils.cpp_extension.LIB_EXT}",
            )

        allocator = torch.mlu.memory.MLUPluggableAllocator(
            dummy_allocator,
            "dummy_alloc",
            "dummy_free",
        )

        # Swap the current allocator
        torch.mlu.memory.change_current_allocator(allocator)

        alloc_lib = ctypes.CDLL(dummy_allocator)
        init_fn = ctypes.cast(getattr(alloc_lib, "init"), ctypes.c_void_p).value
        empty_fn = ctypes.cast(getattr(alloc_lib, "empty_cache"), ctypes.c_void_p).value
        memory_fractin_fn = ctypes.cast(
            getattr(alloc_lib, "memory_fragmentation_memory_fraction"), ctypes.c_void_p
        ).value
        base_alloc_fn = ctypes.cast(
            getattr(alloc_lib, "base_alloc"), ctypes.c_void_p
        ).value
        record_stream_fn = ctypes.cast(
            getattr(alloc_lib, "record_stream"), ctypes.c_void_p
        ).value
        alloc_to_pool_fn = ctypes.cast(
            getattr(alloc_lib, "begin_alloc_to_pool"), ctypes.c_void_p
        ).value
        end_alloc_to_pool_fn = ctypes.cast(
            getattr(alloc_lib, "end_alloc_to_pool"), ctypes.c_void_p
        ).value
        release_pool_fn = ctypes.cast(
            getattr(alloc_lib, "release_alloc_pool"), ctypes.c_void_p
        ).value

        allocator.allocator().set_init_fn(init_fn)
        allocator.allocator().set_reset_fn(empty_fn)
        allocator.allocator().set_memory_fraction_fn(memory_fractin_fn)
        allocator.allocator().set_base_alloc_fn(base_alloc_fn)
        allocator.allocator().set_record_stream_fn(record_stream_fn)
        allocator.allocator().set_begin_allocate_to_pool(alloc_to_pool_fn)
        allocator.allocator().set_end_allocate_to_pool_fn(end_alloc_to_pool_fn)
        allocator.allocator().set_release_pool(release_pool_fn)

        called_dummy_alloc = ctypes.c_int.in_dll(alloc_lib, "called_dummy_alloc")
        is_init = ctypes.c_bool.in_dll(alloc_lib, "is_init")
        enable_emtpy_cache = ctypes.c_bool.in_dll(alloc_lib, "enable_empty_cache")
        memory_fraction = ctypes.c_int.in_dll(alloc_lib, "memory_fraction")
        alloc_size = ctypes.c_int.in_dll(alloc_lib, "alloc_size")
        enable_record_stream = ctypes.c_bool.in_dll(alloc_lib, "enable_record_stream")
        enable_alloc_pool = ctypes.c_bool.in_dll(alloc_lib, "enable_alloc_pool")
        end_alloc_pool = ctypes.c_bool.in_dll(alloc_lib, "end_alloc_pool")
        release_pool = ctypes.c_bool.in_dll(alloc_lib, "release_pool")

        # no allocations happened yet, so called_dummy_alloc should be 0
        self.assertEqual(called_dummy_alloc.value, 0)

        out = torch.randn(1, device="mlu")
        out.storage()._share_mlu_()
        self.assertEqual(is_init.value, True)

        # test set memory fraction
        torch.mlu.memory.set_per_process_memory_fraction(1.0)
        self.assertEqual(memory_fraction.value, 111)

        # test set base alloc
        self.assertEqual(alloc_size.value, 222)

        # test set record stream
        stream = torch.mlu.current_stream()
        out.record_stream(stream)
        self.assertEqual(enable_record_stream.value, True)

        # test set begin_allocate_to_pool
        import torch_mlu

        s = torch.mlu.Stream()
        pool = torch.mlu.graph_pool_handle()
        with torch.mlu.stream(s):
            torch_mlu._MLUC._mlu_beginAllocateCurrentStreamToPool(0, pool)
            self.assertEqual(enable_alloc_pool.value, True)
            # test set end_allocate_to_pool
            torch_mlu._MLUC._mlu_endAllocateCurrentStreamToPool(0, pool)
            self.assertEqual(end_alloc_pool.value, True)
            # test set set_release_pool
            torch_mlu._MLUC._mlu_releasePool(0, pool)
            self.assertEqual(release_pool.value, True)

        # called_dummy_alloc should be 123 if dummy_alloc was used to allocate
        # out tensor
        self.assertEqual(called_dummy_alloc.value, 123)

        # test exception
        msg = "Can't swap an already initialized allocator"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.mlu.memory.change_current_allocator(allocator)

        # test empty cache
        torch.mlu.memory.empty_cache()
        self.assertEqual(enable_emtpy_cache.value, True)


if __name__ == "__main__":
    unittest.main()
