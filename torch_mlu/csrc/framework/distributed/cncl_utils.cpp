#include "cncl_utils.h"

// Provides additional detail into CNCL error codes based on when these are
// thrown in the CNCL codebase.
std::string getCnclErrorDetailStr(
    cnclResult_t error,
    c10::optional<std::string> process_group_failure_reason /* = c10::nullopt */
) {
  // Prioritize failure reason provided by PG CNCL first, as it can abort
  // communicators when it encounters collective timeouts, etc.
  if (process_group_failure_reason != c10::nullopt) {
    return *process_group_failure_reason;
  }
  return std::string(cnclGetErrorStr(error));
}

namespace torch_mlu {
namespace cncl::detail {

// Avoid send/recv when message size is zero
template <typename T>
inline bool _cncl_should_send_recv(T value) {
  return value != 0;
}

cnclDataType_t to_cncl_data_type(c10::ScalarType type) {
  switch (type) {
    case at::kFloat:
      return cnclDataType_t::cnclFloat;
    case at::kHalf:
      return cnclDataType_t::cnclHalf;
    case at::kDouble:
      return cnclDataType_t::cnclFloat;
    case at::kLong:
      return cnclDataType_t::cnclInt64;
    case at::kInt:
      return cnclDataType_t::cnclInt32;
    case at::kChar:
      return cnclDataType_t::cnclInt8;
    case at::kByte:
      return cnclDataType_t::cnclUint8;
    case at::kBool:
      return cnclDataType_t::cnclUint8;
    case at::kBFloat16:
      return cnclDataType_t::cnclBfloat16;
    default:
      TORCH_CHECK(false, "Unconvertible CNCL type ", type);
  }
}

cnclDataType_t to_cncl_data_type(const at::Tensor& t) {
  if (!t.device().is_privateuseone()) {
    TORCH_CHECK(
        false,
        "CNCL only supports MLU tensors, but got a tensor on ",
        t.device());
  }
  return to_cncl_data_type(t.scalar_type());
}

void all2all(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream) {
  CNCL_CHECK(cnclGroupStart());

  for (const auto r : c10::irange(outputTensors.size())) {
    at::Tensor& input = inputTensors[r];
    at::Tensor& output = outputTensors[r];

    auto input_impl = torch_mlu::getMluTensorImpl(input);
    auto input_ptr = input_impl->mlu_data_ptr();
    auto output_impl = torch_mlu::getMluTensorImpl(output);
    auto output_ptr = output_impl->mlu_data_ptr();

    if (_cncl_should_send_recv(input.numel())) {
      CNCL_CHECK(cnclSend(
          input_ptr,
          input.numel(),
          to_cncl_data_type(input),
          r,
          comm,
          stream.stream()));
    }
    if (_cncl_should_send_recv(output.numel())) {
      CNCL_CHECK(cnclRecv(
          output_ptr,
          output.numel(),
          to_cncl_data_type(output),
          r,
          comm,
          stream.stream()));
    }
  }
  CNCL_CHECK(cnclGroupEnd());
}

void all2all_single_unequal_split(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType _type,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream) {
  auto type = to_cncl_data_type(_type);

  int numranks;
  CNCL_CHECK(cnclGetCommCount(&numranks, comm));
  CNCL_CHECK(cnclGroupStart());
  for (const auto r : c10::irange(numranks)) {
    // Avoid send/recv when message size is zero
    if (sendcounts[r] != 0) {
      CNCL_CHECK(cnclSend(
          ((char*)sendbuff) + senddispls[r] * size,
          sendcounts[r],
          type,
          r,
          comm,
          stream.stream()));
    }
    if (recvcounts[r] != 0) {
      CNCL_CHECK(cnclRecv(
          ((char*)recvbuff) + recvdispls[r] * size,
          recvcounts[r],
          type,
          r,
          comm,
          stream.stream()));
    }
  }
  CNCL_CHECK(cnclGroupEnd());
}

void gather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream,
    int32_t root) {
  int numranks, cur_rank;
  CNCL_CHECK(cnclGetCommCount(&numranks, comm));
  CNCL_CHECK(cnclGetCommRank(&cur_rank, comm));

  size_t count = inputs.numel();
  auto type = to_cncl_data_type(inputs);
  auto input_ptr = torch_mlu::getMluTensorImpl(inputs)->mlu_data_ptr();

  CNCL_CHECK(cnclGroupStart());

  if (cur_rank == root) {
    for (const auto r : c10::irange(numranks)) {
      if (r != root) {
        auto output_ptr =
            torch_mlu::getMluTensorImpl(outputs[r])->mlu_data_ptr();
        CNCL_CHECK(cnclRecv(output_ptr, count, type, r, comm, stream.stream()));
      } else {
        // on its own rank, simply copy from the input
        outputs[r].copy_(inputs);
      }
    }
  } else {
    CNCL_CHECK(cnclSend(input_ptr, count, type, root, comm, stream.stream()));
  }
  CNCL_CHECK(cnclGroupEnd());
}

void scatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream,
    int32_t root) {
  int numranks, cur_rank;
  CNCL_CHECK(cnclGetCommCount(&numranks, comm));
  CNCL_CHECK(cnclGetCommRank(&cur_rank, comm));

  CNCL_CHECK(cnclGroupStart());
  if (cur_rank == root) {
    for (const auto r : c10::irange(numranks)) {
      if (r != root) {
        size_t send_count = inputs[r].numel();
        auto send_type = to_cncl_data_type(inputs[r]);
        auto send_ptr = torch_mlu::getMluTensorImpl(inputs[r])->mlu_data_ptr();
        auto* sendbuff = reinterpret_cast<char*>(send_ptr);
        CNCL_CHECK(cnclSend(sendbuff, send_count, send_type, r, comm, stream));
      } else {
        // on its own rank, simply copy it to the output
        outputs.copy_(inputs[r]);
      }
    }
  } else {
    size_t recv_count = outputs.numel();
    auto recv_type = to_cncl_data_type(outputs);

    auto recv_ptr = torch_mlu::getMluTensorImpl(outputs)->mlu_data_ptr();
    auto* recvbuff = reinterpret_cast<char*>(recv_ptr);
    CNCL_CHECK(cnclRecv(recvbuff, recv_count, recv_type, root, comm, stream));
  }
  CNCL_CHECK(cnclGroupEnd());
}
} // namespace cncl::detail

std::mutex* getFreeMutex() {
  static std::mutex mlu_free_mutex;
  return &mlu_free_mutex;
}

std::unordered_map<DeviceIndex, std::unordered_map<std::string, MLUStream>>
    cncl_streams;
std::mutex mutex;

std::unordered_map<std::string, MLUStream> getCnclStream(
    const DeviceIndex& device_index = -1) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cncl_streams.find(device_index);
  if (it != cncl_streams.end()) {
    return it->second;
  } else {
    return std::unordered_map<std::string, MLUStream>();
  }
}

void updateCnclStream(const cnclCliqueId* cncl_id, MLUStream cncl_stream) {
  std::lock_guard<std::mutex> lock(mutex);
  auto device_index = cncl_stream.device_index();
  std::string clique_id_data(cncl_id->data, CNCL_CLIQUE_ID_BYTES_SIZE);
  cncl_streams[device_index].emplace(clique_id_data, std::move(cncl_stream));
}

void clearCnclStream(const cnclCliqueId* cncl_id) {
  std::lock_guard<std::mutex> lock(mutex);
  std::string clique_id_data(cncl_id->data, CNCL_CLIQUE_ID_BYTES_SIZE);
  for (auto it = cncl_streams.begin(); it != cncl_streams.end();) {
    auto& clique_map = it->second;
    auto clique_it = clique_map.find(clique_id_data);
    if (clique_it != clique_map.end()) {
      clique_map.erase(clique_it);
    }
    if (clique_map.empty()) {
      it = cncl_streams.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace torch_mlu
