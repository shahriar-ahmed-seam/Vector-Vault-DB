# Vector-Vault-DB

A high-performance vector database written from scratch. The engine is C++17;
the public interface is a Python extension built with pybind11. There is no
dependency on an existing vector store — the index, distance kernels, allocator,
and on-disk format are all implemented directly.

## Features

- **Approximate nearest-neighbor search** over float32 vectors using either an
  HNSW graph or an IVF (inverted-file) index.
- **SIMD-accelerated distance kernels** (AVX-512) for Euclidean, cosine, and dot
  product, with a scalar fallback selected once at startup via CPU detection.
- **Custom arena allocator** that hands out 64-byte aligned blocks for vector
  storage and tracks per-collection usage.
- **Memory-mapped persistence**: a versioned, CRC-checked binary snapshot format.
  Saves are atomic (temp file + fsync + rename); loads map the vector region
  instead of copying it onto the heap.
- **Python bindings** with NumPy support and an exception hierarchy that mirrors
  the engine's error categories.

## Documentation

Full documentation lives in [`docs/`](docs/index.md):

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Concepts](docs/concepts.md) — metrics, indexes, tuning, persistence
- [Python API reference](docs/python-api.md)
- [C++ API guide](docs/cpp-api.md)

## Architecture

```
Python (vectorvault)            pybind11 extension, NumPy interop, exceptions
        │
        ▼
Engine                          collection registry, shared allocator + persistence
        │
        ▼
Collection                      record store, validation, query orchestration
   ├── Index            HNSW / IVF — graph + inverted-file ANN
   ├── DistanceCalculator   AVX-512 kernels with scalar fallback
   ├── MemoryAllocator      64-byte aligned arena, per-collection accounting
   └── PersistenceManager   atomic save, mmap-backed load, CRC validation
```

A collection is guarded by a single readers-writer lock: reads (get, query) take
a shared lock, mutations (insert, delete, build, save) take an exclusive lock, so
index membership stays consistent with the record store.

## Requirements

- A C++17 compiler (GCC, Clang, or MSVC)
- CMake >= 3.20
- Python >= 3.9 with NumPy (for the bindings)
- pybind11 (resolved by the build backend)

Catch2, RapidCheck, and Hypothesis are fetched automatically for the test builds.

## Install

```bash
pip install -e ".[test]"
```

This builds the native extension through scikit-build-core and installs the
`vectorvault` package in editable mode.

## Quickstart

```python
import vectorvault as vv

engine = vv.Engine()
coll = engine.create_collection("documents", dim=128, metric="cosine")

coll.insert("doc-1", embedding_a, metadata={"title": "intro"})
coll.insert("doc-2", embedding_b)

coll.build_index("hnsw", m=16, ef_construction=200)

results = coll.query(query_vector, k=10, ef_search=64)
for record_id, distance in results:
    print(record_id, distance)

engine.save("documents", "documents.vv")
restored = vv.Engine().load("documents.vv")
```

Metrics: `"euclidean"` (`"l2"`), `"cosine"`, `"dot"` (`"dot_product"`).
Index types: `"hnsw"`, `"ivf"`.

Errors surface as typed exceptions under `vectorvault.VectorVaultError`; several
also derive from a builtin (for example `NotFoundError` is a `KeyError` and
`SnapshotNotFoundError` is a `FileNotFoundError`).

## On-disk format

A snapshot is a single file: a fixed header (magic, version, dimensionality,
metric, index type, region offsets, CRC-64 of the content), the collection name,
a record directory sorted by id with optional metadata, a 64-byte aligned vector
region, and the serialized index. Records, metadata keys, and index nodes are
written in a canonical order, so saving a collection, loading it, and saving
again produces a byte-identical file. Loads validate existence, magic, version,
and checksum before mapping the vector region.

## Testing

C++ (Catch2 + RapidCheck):

```bash
cmake -S . -B build -DVECTORVAULT_BUILD_TESTS=ON -DVECTORVAULT_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

Python (pytest + Hypothesis):

```bash
pytest
```

Correctness-critical behavior — distance accuracy against a reference, allocator
accounting, query ordering, index membership, and the snapshot round-trip — is
covered by property-based tests. A recall benchmark builds an index over 10,000
vectors and asserts mean recall@10 against an exact baseline.

## Layout

```
include/vectorvault/   Public C++ headers
src/core/              Core engine implementation
src/python/            pybind11 module + vectorvault package
tests/cpp/             C++ tests (Catch2 + RapidCheck)
tests/python/          Python tests (pytest + Hypothesis)
```

## Notes and limitations

- The HNSW index selects neighbours with the diversity heuristic from Malkov &
  Yashunin (SELECT-NEIGHBORS-HEURISTIC), applied both when linking a new node and
  when pruning an existing node whose adjacency list overflows. On uniform random
  data this measures mean recall@10 of ~0.96 at dim 32 and ~0.85 at dim 64
  (m=16, ef_construction=200, ef_search=50); recall still tapers on harder,
  higher-dimensional distributions.
- The snapshot reader assumes a little-endian host (it reads float components
  directly from the mapping).
- AVX-512 is detected at runtime; on hosts without it the scalar kernels are
  used, which are also the accuracy reference.

## License

MIT
