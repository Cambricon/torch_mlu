#pragma once

#include <cstdint>

namespace KINETO_NAMESPACE {

// cnpapi's timestamps are platform specific. This function convert the raw
// cnpapi timestamp to time since unix epoch. So that on different platform,
// correction can work correctly.
uint64_t unixEpochTimestamp(uint64_t ts);

} // namespace KINETO_NAMESPACE
