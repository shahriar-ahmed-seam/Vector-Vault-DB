// Toolchain smoke test for the C++ core.
//
// Confirms that Catch2 is wired as the test runner, that RapidCheck is wired
// for property-based testing and integrates with Catch2, and that the
// vectorvault_core library links and its symbols run.

#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>

#include "vectorvault/version.hpp"

#include <algorithm>
#include <string>

TEST_CASE("core library links and version is well formed", "[smoke]") {
    const std::string v = vectorvault::version_string();
    REQUIRE_FALSE(v.empty());
    // "MAJOR.MINOR.PATCH" -> at least two dot separators.
    REQUIRE(std::count(v.begin(), v.end(), '.') == 2);
}

// Drives a RapidCheck property from Catch2 via rc::check (returns false if any
// case fails), proving the property-based harness executes and links against
// the core library.
TEST_CASE("rapidcheck harness runs against core", "[smoke][property]") {
    const bool ok = rc::check("version_string is deterministic", [] {
        RC_ASSERT(vectorvault::version_string() == vectorvault::version_string());
    });
    REQUIRE(ok);
}
