#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

#include "python/combined_traceback.h"

namespace py = pybind11;

namespace torch_mlu {
std::vector<py::object> py_symbolize(
    std::vector<torch::CapturedTraceback*>& to_symbolize) {
  // we dedup repeated to_symbolize objects to prevent
  // creating a bunch of duplicated frame objects
  std::unordered_map<torch::CapturedTraceback*, uint64_t> cached_frames;
  std::vector<torch::CapturedTraceback*> unique_frames;
  for (const auto& sc : to_symbolize) {
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      cached_frames.insert({sc, unique_frames.size()});
      unique_frames.push_back(sc);
    }
  }
  auto s = symbolize(unique_frames);

  py::str line_s = "line";
  py::str name_s = "name";
  py::str filename_s = "filename";
  std::vector<py::dict> all_frames;
  for (const auto& f : s.all_frames) {
    py::dict d;
    d[name_s] = f.funcname;
    d[filename_s] = f.filename;
    d[line_s] = f.lineno;
    all_frames.emplace_back(std::move(d));
  }

  std::vector<py::object> py_unique_frames;
  for (const auto& t : s.tracebacks) {
    py::list l;
    for (const auto& e : t) {
      l.append(all_frames.at(e));
    }
    py_unique_frames.push_back(std::move(l));
  }

  std::vector<py::object> result;
  result.reserve(to_symbolize.size());
  for (const auto& sc : to_symbolize) {
    result.push_back(py_unique_frames.at(cached_frames.at(sc)));
  }
  return result;
}
} // namespace torch_mlu
