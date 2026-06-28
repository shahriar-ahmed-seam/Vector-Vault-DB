# Vector-Vault-DB Documentation

Vector-Vault-DB is a vector database with a C++17 core and a Python interface.
It stores high-dimensional float32 vectors, indexes them for approximate
nearest-neighbor search (HNSW or IVF), and persists collections to a
memory-mapped binary format.

## Contents

- [Installation](installation.md) — install from PyPI or source; build the C++ core.
- [Quickstart](quickstart.md) — create a collection, insert vectors, query, persist.
- [Concepts](concepts.md) — collections, distance metrics, indexes, tuning, persistence.
- [Python API reference](python-api.md) — every class, method, and exception.
- [C++ API guide](cpp-api.md) — embedding the core engine in a C++ project.

## At a glance

```python
import vectorvault as vv

engine = vv.Engine()
docs = engine.create_collection("docs", dim=384, metric="cosine")

docs.insert("a", embedding_a, metadata={"title": "intro"})
docs.insert("b", embedding_b)
docs.build_index("hnsw")

for record_id, distance in docs.query(query_vector, k=5):
    print(record_id, distance)

engine.save("docs", "docs.vv")
```
