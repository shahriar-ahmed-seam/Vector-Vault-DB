# C++ API Guide

The engine is a standalone C++17 library, `vectorvault_core`. The Python module
is a thin pybind11 wrapper over it. This guide covers embedding the core
directly in a C++ project.

## Building and linking

There is no install/export target yet; consume the engine as a subdirectory.

```cmake
# CMakeLists.txt
add_subdirectory(third_party/Vector-Vault-DB)   # path to the repo
target_link_libraries(my_app PRIVATE vectorvault_core)
```

Headers live under `include/vectorvault/` and are available on the
`vectorvault_core` target's include path. The library targets C++17.

## Error handling

The core does not throw across its API. Operations return `Status` (success or
error) or `Result<T>` (a value or an error). Both carry an `ErrorCategory` and a
human-readable cause.

```cpp
#include "vectorvault/engine.hpp"
using namespace vectorvault;

Engine engine;

Result<CollectionHandle> created =
    engine.create_collection("docs", 128, DistanceMetric::Cosine);
if (created.is_error()) {
    std::cerr << created.status().message() << "\n";
    return 1;
}
CollectionHandle docs = created.value();
```

`CollectionHandle` is a non-owning reference to a collection owned by the engine
registry; it is valid until the collection is deleted.

## Vectors

Vectors are passed as `vectorvault::span<const float>` (a `std::span` alias on
C++20, with a built-in fallback on C++17). Length must equal the collection
dimensionality.

```cpp
std::vector<float> v(128, 0.0f);
Status s = docs->insert("a", span<const float>(v.data(), v.size()), std::nullopt);
```

## Core operations

### Engine

```cpp
Result<CollectionHandle> create_collection(std::string_view name,
                                           std::uint32_t dimensionality,
                                           DistanceMetric metric);
Result<CollectionHandle> get_collection(std::string_view name);
std::vector<CollectionInfo> list_collections() const;
Status delete_collection(std::string_view name);

Status save_collection(std::string_view name, const std::filesystem::path& path);
Result<CollectionHandle> load_collection(const std::filesystem::path& path);
```

### Collection

```cpp
Status insert(const RecordId& id, span<const float> vec,
              std::optional<Metadata> meta = std::nullopt);
Status insert_batch(span<const RecordInput> records);          // 1..10000

Result<RecordView> get(const RecordId& id) const;
Status remove(const RecordId& id);
std::uint64_t count() const;

Status build_index(IndexType type, const IndexParams& params);
Result<std::vector<Neighbor>> query(span<const float> q, std::uint32_t k,
                                    const QueryParams& params) const;
IndexType index_type() const;
```

`RecordId` is `std::string`. `Metadata` is `std::map<std::string,
std::variant<std::string, std::int64_t, double, bool>>`. `Neighbor` is
`{ RecordId id; float distance; }`.

### Parameters

```cpp
struct IndexParams {
    std::uint32_t m = 16;               // HNSW: max neighbors per node (>= 2)
    std::uint32_t ef_construction = 200; // HNSW: build breadth (>= 1)
    std::uint32_t nlist = 0;             // IVF: partitions (1..count; 0 = auto)
    std::uint64_t seed = /* fixed */;    // deterministic construction
};

struct QueryParams {
    std::uint32_t ef_search = 50;        // HNSW: search breadth (>= 1)
    std::uint32_t nprobe = 1;            // IVF: cells probed (1..nlist)
};
```

## End-to-end example

```cpp
#include "vectorvault/engine.hpp"
#include <iostream>

using namespace vectorvault;

int main() {
    Engine engine;
    auto coll = engine.create_collection("docs", 4, DistanceMetric::Euclidean).value();

    std::vector<float> a{1, 0, 0, 0};
    std::vector<float> b{0, 1, 0, 0};
    coll->insert("a", span<const float>(a.data(), a.size()), std::nullopt);
    coll->insert("b", span<const float>(b.data(), b.size()), std::nullopt);

    coll->build_index(IndexType::HNSW, IndexParams{});

    std::vector<float> q{0.9f, 0.1f, 0, 0};
    auto results = coll->query(span<const float>(q.data(), q.size()), 2, QueryParams{});
    for (const Neighbor& n : results.value()) {
        std::cout << n.id << " " << n.distance << "\n";
    }

    engine.save_collection("docs", "docs.vv");
}
```

## Subcomponents

The engine is composed of independently usable pieces, each with its own header:

- `distance.hpp` — `DistanceCalculator` (SIMD/scalar distance kernels).
- `hnsw_index.hpp`, `ivf_index.hpp`, `index.hpp` — the ANN indexes.
- `memory_allocator.hpp` — the 64-byte aligned arena allocator.
- `persistence.hpp`, `snapshot_format.hpp`, `mmap_region.hpp`, `crc64.hpp` — the
  on-disk format and memory-mapped loading.
- `error.hpp`, `types.hpp` — `Status`/`Result`, enums, and shared models.

## Thread safety

Each collection is guarded by a readers-writer lock. Reads (`get`, `query`) take
a shared lock; mutations (`insert`, `remove`, `build_index`, `save`) take an
exclusive lock. The allocator is internally synchronized.
