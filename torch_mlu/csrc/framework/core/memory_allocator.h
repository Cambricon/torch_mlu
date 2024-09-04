/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/pytorch/pytorch/graphs/contributors Redistribution and use in
source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

// #include <iostream>
#include <memory>
#include "c10/util/Exception.h"
#include "utils/common.h"
#include "framework/core/MLUStream.h"
#include "framework/hooks/MLUHooks.h"

namespace torch_mlu {

/**
 * Note [HostMemoryAllocator]
 * ~~~~~~~~~~~~~~~~
 * A host caching allocator is to hold MLU host page-locked memory.
 * Which is designed for re-uses freed pinned (page-locked) memory,
 * and avoid too many time-used api call. Like cnrtHostMalloc, cnrtFreeHost.
 *
 * Also Caching allocator tries to avoid allocating and freeing memory for each
 * use for performance reasons. Resources only be freed by explicitly clearing
 * the cache or at the teardown of process.
 * https://discuss.pytorch.org/t/why-dont-explicit-free-cpu-resource-in-cachinghostallocator/189714
 *
 * Also can get more details in [CUDAHostAllocator design]
 * https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/cuda/CachingHostAllocator.cpp#L116
 *
 * Note1: To ensure correct behavior, MLUCachingHostAllocator_recordEvent must
 * be called anytime a pointer from this allocator is used. Example:
 *   {
 *     at::DataPtr ptr = getMLUCachingHostAllocator()->allocate(size);
 *     // do something
 *     MLUCachingHostAllocator_recordEvent(ptr.get(), ptr.get_context(),
 * stream);
 *   }
 *
 * Note2: when you add new public function in this class, you may
 * need add a lock guard protection.
 *
 * Note3: that this allocator does not split larger allocations into smaller
 * blocks, unlike the caching device allocator.
 *
 */

TORCH_MLU_API bool MLUCachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    torch_mlu::MLUStream stream);

TORCH_MLU_API void MLUCachingHostAllocator_emptyCache();

// To get MLUCachingHostAllocator
TORCH_MLU_API at::Allocator* getMLUCachingHostAllocator();

// Not using now, but aligned with pytorch gpu host allocator.
inline at::DataPtr HostAlloc(size_t size) {
  return getMLUCachingHostAllocator()->allocate(size);
}

} // namespace torch_mlu
