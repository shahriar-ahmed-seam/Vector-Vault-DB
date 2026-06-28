#ifndef VECTORVAULT_VERSION_HPP
#define VECTORVAULT_VERSION_HPP

#include <string>

namespace vectorvault {

constexpr int kVersionMajor = 0;
constexpr int kVersionMinor = 1;
constexpr int kVersionPatch = 0;

// Returns the engine version as a "MAJOR.MINOR.PATCH" string.
std::string version_string();

}  // namespace vectorvault

#endif  // VECTORVAULT_VERSION_HPP
