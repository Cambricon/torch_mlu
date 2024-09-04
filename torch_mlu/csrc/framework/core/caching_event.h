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

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <deque>
#include <memory>
#include "ATen/Utils.h"
#include "framework/core/device.h"
#include "framework/core/mlu_guard.h"
#include "framework/core/MLUStream.h"
#include "framework/core/MLUEvent.h"

namespace torch_mlu {

class CachingMLUEvent {
 private:
  CachingMLUEvent() {}
  std::deque<std::shared_ptr<MLUEvent>> event_pool[MLU_DEVICE_NUM_MAX];
  std::mutex event_mutex[MLU_DEVICE_NUM_MAX];
  static CachingMLUEvent instance;
  AT_DISALLOW_COPY_AND_ASSIGN(CachingMLUEvent);

 public:
  ~CachingMLUEvent() {
    clean_event();
  }

  // alloc a event from event pool.
  std::shared_ptr<MLUEvent> alloc_event(c10::DeviceIndex device_id);

  // give back event to event pool.
  void give_back_event(std::shared_ptr<MLUEvent> event);

  // Singleton
  static CachingMLUEvent& get_instance();

  // clear event in event pool.
  void clean_event();
};

#define MLUEventPool_Manager CachingMLUEvent::get_instance()

} // namespace torch_mlu
