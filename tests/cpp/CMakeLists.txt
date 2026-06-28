# C++ test harness: Catch2 (unit/PBT runner) + RapidCheck (property generators).
# Both are fetched at configure time so contributors need no system packages.
include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
    GIT_SHALLOW    TRUE)

FetchContent_Declare(
    rapidcheck
    GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
    GIT_TAG        ff6af6fc683159deb51c543b065eba14dfcf329b
    GIT_SHALLOW    FALSE)

# RapidCheck ships an optional bundled Catch (v1) integration whose target
# collides with modern Catch2 v3. Keep it disabled and drive RapidCheck
# properties from Catch2 test cases via rc::check.
set(RC_ENABLE_CATCH OFF CACHE BOOL "Disable RapidCheck bundled Catch integration" FORCE)

FetchContent_MakeAvailable(Catch2 rapidcheck)

add_executable(vectorvault_cpp_tests
    smoke_test.cpp
    memory_allocator_test.cpp
    distance_test.cpp
    engine_lifecycle_test.cpp
    record_store_test.cpp
    index_query_test.cpp
    persistence_test.cpp
    end_to_end_test.cpp
    recall_dispatch_test.cpp)

target_link_libraries(vectorvault_cpp_tests
    PRIVATE
        vectorvault_core
        Catch2::Catch2WithMain
        rapidcheck)

# Register with CTest. Catch2 provides discovery, but a single add_test keeps
# the smoke harness dependency-light.
include(CTest)
add_test(NAME vectorvault_cpp_tests COMMAND vectorvault_cpp_tests)
