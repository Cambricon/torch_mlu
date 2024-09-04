#include "aten/operators/cnnl/cnnl_kernel.h"

namespace torch_mlu {
namespace ops {

void cnnl_record_stream(at::Tensor& self, c10::Stream s) {
  struct c10::StreamData3 data = s.pack3();
  torch_mlu::MLUCachingAllocator::recordStream(
      self.storage().data_ptr(),
      torch_mlu::MLUStream::unpack3(
          data.stream_id, data.device_index, data.device_type));
}

} // namespace ops
} // namespace torch_mlu
