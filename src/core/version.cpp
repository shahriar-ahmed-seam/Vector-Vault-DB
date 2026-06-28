#include "vectorvault/version.hpp"

#include <string>

namespace vectorvault {

std::string version_string() {
    return std::to_string(kVersionMajor) + "." +
           std::to_string(kVersionMinor) + "." +
           std::to_string(kVersionPatch);
}

}  // namespace vectorvault
