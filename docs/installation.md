# Installation

## From PyPI

```bash
pip install vectorvault
```

Wheels are published for CPython 3.9–3.12 on Linux, macOS, and Windows. NumPy is
pulled in automatically.

## From source

Building from source requires a C++17 compiler (GCC, Clang, or MSVC) and CMake
3.20+. The build is driven by scikit-build-core and pybind11, which pip resolves
from `pyproject.toml`.

```bash
git clone https://github.com/shahriar-ahmed-seam/Vector-Vault-DB.git
cd Vector-Vault-DB
pip install .              # or: pip install -e ".[test]" for a dev install
```

Verify the install:

```python
import vectorvault as vv
print(vv.__version__)
```

## Building the C++ core only

The engine can be built and tested without Python. This produces the static
library `vectorvault_core` and the C++ test binary.

```bash
cmake -S . -B build -DVECTORVAULT_BUILD_TESTS=ON -DVECTORVAULT_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

See the [C++ API guide](cpp-api.md) for embedding the core in another project.

## Running the test suites

```bash
# Python (pytest + Hypothesis)
pip install -e ".[test]"
pytest

# C++ (Catch2 + RapidCheck, fetched at configure time)
cmake -S . -B build -DVECTORVAULT_BUILD_TESTS=ON -DVECTORVAULT_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```
