cmake_minimum_required(VERSION 3.20)

project(VectorVaultDB
    VERSION 0.1.0
    DESCRIPTION "High-performance vector database with a C++ core and Python bindings"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Build options. The Python module is built by default (this is how
# scikit-build-core drives the build). C++ tests are opt-in so that wheel
# builds do not need to fetch the test dependencies.
option(VECTORVAULT_BUILD_PYTHON "Build the pybind11 Python extension" ON)
option(VECTORVAULT_BUILD_TESTS  "Build the C++ test harness (Catch2 + RapidCheck)" OFF)

# ---------------------------------------------------------------------------
# Core engine library (C++17)
# ---------------------------------------------------------------------------
add_library(vectorvault_core STATIC
    src/core/version.cpp
    src/core/error.cpp
    src/core/memory_allocator.cpp
    src/core/distance.cpp
    src/core/collection.cpp
    src/core/hnsw_index.cpp
    src/core/ivf_index.cpp
    src/core/crc64.cpp
    src/core/persistence.cpp
    src/core/mmap_region.cpp
    src/core/engine.cpp)

target_include_directories(vectorvault_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

set_target_properties(vectorvault_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Python extension (pybind11)
# ---------------------------------------------------------------------------
if(VECTORVAULT_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_vectorvault src/python/module.cpp)
    target_link_libraries(_vectorvault PRIVATE vectorvault_core)

    # On MinGW/MSYS2 the extension would otherwise depend on libstdc++,
    # libgcc, and libwinpthread DLLs from the toolchain's bin directory.
    # Link them statically so the .pyd is self-contained and importable by a
    # stock CPython interpreter without the toolchain on PATH.
    if(MINGW)
        target_link_options(_vectorvault PRIVATE
            -static-libgcc -static-libstdc++
            -Wl,-Bstatic,--whole-archive -lwinpthread
            -Wl,--no-whole-archive)
    endif()

    # Install the extension into the vectorvault package (scikit-build-core
    # collects this together with the pure-Python package files).
    install(TARGETS _vectorvault DESTINATION vectorvault)
endif()

# ---------------------------------------------------------------------------
# C++ test harness
# ---------------------------------------------------------------------------
if(VECTORVAULT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
