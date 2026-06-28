# Python API Reference

Everything is exported from the top-level `vectorvault` package.

```python
import vectorvault as vv
```

- [`Engine`](#engine)
- [`Collection`](#collection)
- [`CollectionInfo`](#collectioninfo)
- [Enums](#enums)
- [Exceptions](#exceptions)
- [Module attributes](#module-attributes)

---

## Engine

The entry point. Owns a registry of collections and the shared allocator,
distance calculator, and persistence layer.

```python
engine = vv.Engine()
```

### `create_collection(name, dim, metric) -> Collection`

Create an empty collection.

| Parameter | Type | Notes |
|---|---|---|
| `name` | `str` | 1–255 characters, unique within the engine. |
| `dim` | `int` | Dimensionality, `[1, 65536]`. |
| `metric` | `str` | `"euclidean"`/`"l2"`, `"cosine"`, or `"dot"`/`"dot_product"`. |

Raises `NameConflictError`, `InvalidNameError`, `InvalidDimensionalityError`, or
`InvalidMetricError`.

```python
coll = engine.create_collection("docs", dim=128, metric="cosine")
```

### `get_collection(name) -> Collection`

Return a handle to an existing collection, or raise `NotFoundError`.

### `list_collections() -> list[CollectionInfo]`

Return one [`CollectionInfo`](#collectioninfo) per collection (empty list if
none).

### `delete_collection(name) -> None`

Remove a collection and release its memory. Raises `NotFoundError` if it does not
exist.

### `save(name, path) -> None`

Write a collection to a snapshot file. The write is atomic: a failure leaves any
existing file at `path` unchanged. Raises `NotFoundError` (unknown collection) or
`WriteFailureError`.

```python
engine.save("docs", "docs.vv")
```

### `load(path) -> Collection`

Load a snapshot into this engine and return the reconstructed collection. Raises
`SnapshotNotFoundError` (missing or unparseable), `UnsupportedVersionError`,
`CorruptionError`, or `NameConflictError` (a collection with the stored name
already exists in this engine).

```python
coll = vv.Engine().load("docs.vv")
```

---

## Collection

Obtained from `create_collection`, `get_collection`, or `load`. Not constructed
directly.

### `insert(id, vector, metadata=None) -> None`

Insert or replace a record. `vector` is a numeric sequence (list, tuple, or
NumPy array) whose length equals the collection dimensionality. Re-inserting an
existing id overwrites it without changing the record count.

| Raises | When |
|---|---|
| `InvalidIdentifierError` | `id` is empty. |
| `DimensionMismatchError` | `len(vector) != dim`. |
| `InvalidValueError` | a component is NaN or infinite. |
| `TypeError` | `vector` is not a sequence, or contains a non-numeric element (the message names the first bad index). |

```python
coll.insert("a", [0.1, 0.2, 0.3], metadata={"lang": "en", "score": 0.9})
```

### `insert_batch(records) -> None`

Insert 1–10,000 records atomically: if any record is invalid, nothing is stored.
Each record is a tuple mirroring `insert`:

- `(id, vector)`, or
- `(id, vector, metadata)` where metadata is a dict or `None`.

```python
coll.insert_batch([
    ("a", [0.1, 0.2, 0.3]),
    ("b", numpy_vec, {"lang": "fr"}),
])
```

Raises the same per-record errors as `insert`, plus `InvalidArgumentError` if the
batch is empty or larger than 10,000.

### `get(id) -> tuple[list[float], dict | None]`

Return the stored vector and metadata for `id`. Metadata is `None` when none was
stored. Raises `NotFoundError` if the id does not exist.

```python
vector, metadata = coll.get("a")
```

### `delete(id) -> None`

Remove a record; it is excluded from subsequent queries. Raises `NotFoundError`
if the id does not exist.

### `count() -> int`

Number of records currently stored.

### `build_index(type, **params) -> None`

Build (or rebuild) an index over the current records. `type` is `"hnsw"`,
`"ivf"`, or `"none"`.

| Index | Parameters (keyword) | Defaults |
|---|---|---|
| HNSW | `m`, `ef_construction` | `m=16`, `ef_construction=200` |
| IVF | `nlist` | `nlist=0` (auto ≈ √n) |
| both | `seed` | fixed deterministic seed |

Out-of-range parameters raise `InvalidParameterError` and leave any existing
index unchanged.

```python
coll.build_index("hnsw", m=32, ef_construction=400)
coll.build_index("ivf", nlist=256)
```

### `query(vector, k, **params) -> list[tuple[str, float]]`

Return up to `k` nearest records as `(id, distance)` pairs ordered by ascending
distance, ties broken by ascending id. If `k` exceeds the record count, all
records are returned. An empty collection returns `[]`.

| Index | Parameters (keyword) | Defaults |
|---|---|---|
| HNSW | `ef_search` | `50` |
| IVF | `nprobe` | `1` |

| Raises | When |
|---|---|
| `InvalidArgumentError` | `k` is not an integer ≥ 1. |
| `DimensionMismatchError` | query length != dim. |
| `InvalidValueError` | query has a non-finite component. |

```python
results = coll.query(q, k=10, ef_search=128)
```

See [Concepts → Distance metrics](concepts.md#distance-metrics) for how ordering
behaves per metric (note the dot-product caveat).

### `index_type() -> IndexType`

The active index type (`IndexType.NONE`, `IndexType.HNSW`, or `IndexType.IVF`).

---

## CollectionInfo

Returned by `Engine.list_collections()`. Read-only attributes:

| Attribute | Type |
|---|---|
| `name` | `str` |
| `dimensionality` | `int` |
| `metric` | `DistanceMetric` |
| `record_count` | `int` |

---

## Enums

### `DistanceMetric`

`Euclidean`, `Cosine`, `DotProduct`.

### `IndexType`

`NONE`, `HNSW`, `IVF`.

String arguments are accepted everywhere a metric or index type is needed; the
enums appear on returned objects (`CollectionInfo.metric`, `index_type()`).

---

## Exceptions

All inherit from `VectorVaultError`. Several also inherit from a Python builtin
so you can catch either.

| Exception | Also a | Raised when |
|---|---|---|
| `NameConflictError` | | duplicate collection name |
| `InvalidNameError` | `ValueError` | name empty or > 255 chars |
| `InvalidDimensionalityError` | | dimensionality outside `[1, 65536]` |
| `InvalidMetricError` | | unknown metric string |
| `DeletionFailedError` | | a collection deletion failed mid-operation |
| `NotFoundError` | `KeyError` | missing collection or record |
| `DimensionMismatchError` | | vector length != dimensionality |
| `InvalidValueError` | | non-finite vector component |
| `InvalidIdentifierError` | | empty record id |
| `InvalidParameterError` | | index parameter out of range |
| `InvalidArgumentError` | | invalid `k` or batch size |
| `UndefinedDistanceError` | | cosine on a zero-magnitude vector |
| `AllocationFailureError` | `MemoryError` | allocator exhausted |
| `WriteFailureError` | `OSError` | snapshot write failed |
| `SnapshotNotFoundError` | `FileNotFoundError` | snapshot missing or unparseable |
| `UnsupportedVersionError` | | snapshot format version unsupported |
| `CorruptionError` | | snapshot failed its integrity check |

Passing a non-sequence vector, or a sequence with a non-numeric element, raises a
plain `TypeError`.

---

## Module attributes

- `vectorvault.__version__` — the engine version string.
- `vectorvault.version()` — same value, as a function.
