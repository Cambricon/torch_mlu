#pragma once

namespace torch_mlu {

inline std::string getCvarString(
    const std::vector<std::string>& env,
    const char* def) {
  const char* ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (int i = env.size() - 1; i >= 0; i--) {
    const char* val = std::getenv(env[i].c_str());
    if (val == nullptr)
      continue;

    ret = val;
  }

  return ret;
}

inline int getCvarInt(const std::vector<std::string>& env, int def) {
  int ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (int i = env.size() - 1; i >= 0; i--) {
    char* val = std::getenv(env[i].c_str());
    if (val == nullptr)
      continue;

    try {
      ret = std::stoi(val);
    } catch (std::exception&) {
      TORCH_CHECK(false, "Invalid value for environment variable: " + env[i]);
    }
  }

  return ret;
}

inline bool getCvarBool(const std::vector<std::string>& env, bool def) {
  bool ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (int i = env.size() - 1; i >= 0; i--) {
    char* val_ = std::getenv(env[i].c_str());
    if (val_ == nullptr)
      continue;

    std::string val = std::string(val_);
    for (auto& x : val) {
      x = std::tolower(x);
    }

    if (val == "y" || val == "yes" || val == "1" || val == "t" ||
        val == "true") {
      ret = true;
    } else if (
        val == "n" || val == "no" || val == "0" || val == "f" ||
        val == "false") {
      ret = false;
    } else {
      TORCH_CHECK(false, "Invalid value for environment variable: " + env[i]);
      return ret;
    }
  }

  return ret;
}

} // namespace torch_mlu
