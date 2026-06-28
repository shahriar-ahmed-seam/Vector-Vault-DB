# Quickstart

This walks through the full lifecycle: create a collection, add vectors, build
an index, query, and persist to disk.

## 1. Create an engine and a collection

An `Engine` owns one or more named collections. A collection has a fixed
dimensionality and distance metric.

```python
import vectorvault as vv

engine = vv.Engine()
coll = engine.create_collection("articles", dim=128, metric="cosine")
```

Metrics are `"euclidean"`, `"cosine"`, or `"dot"`. Dimensionality must be in
`[1, 65536]`; the collection name must be 1–255 characters and unique.

## 2. Insert vectors

A vector is any numeric sequence (list, tuple, or NumPy array) whose length
equals the collection dimensionality. Metadata is an optional dict of
str/int/float/bool values.

```python
coll.insert("art-1", [0.1, 0.2, ...], metadata={"title": "Hello", "year": 2024})
coll.insert("art-2", numpy_embedding)   # NumPy arrays work directly
```

Re-inserting an existing id replaces the stored vector and metadata. To add many
records efficiently, use a batch (1–10,000 records, applied atomically):

```python
coll.insert_batch([
    ("art-3", vec3),
    ("art-4", vec4, {"title": "Fourth"}),
])
```

## 3. Build an index

Without an index, queries run an exact brute-force scan. For larger collections,
build an approximate index:

```python
coll.build_index("hnsw")                       # defaults: m=16, ef_construction=200
# or
coll.build_index("ivf", nlist=256)
```

The index updates incrementally as you insert or delete records afterward.

## 4. Query

`query` returns up to `k` results as `(id, distance)` pairs, ordered nearest
first.

```python
results = coll.query(query_vector, k=10)
for record_id, distance in results:
    print(record_id, distance)

# Tune accuracy vs. speed at query time:
coll.query(query_vector, k=10, ef_search=128)   # HNSW
coll.query(query_vector, k=10, nprobe=16)        # IVF
```

## 5. Inspect and manage

```python
coll.count()                 # number of stored records
coll.get("art-1")            # -> ([floats...], {metadata} or None)
coll.delete("art-2")         # remove a record
engine.list_collections()    # -> [CollectionInfo, ...]
```

## 6. Persist and restore

Saving writes a single self-contained snapshot. Loading reconstructs the
collection (vectors, metadata, and index) into a fresh engine.

```python
engine.save("articles", "articles.vv")

restored_engine = vv.Engine()
restored = restored_engine.load("articles.vv")
assert restored.count() == coll.count()
```

## Handling errors

Every engine error is a subclass of `vectorvault.VectorVaultError`; several also
subclass a builtin so existing `except` clauses keep working.

```python
try:
    coll.insert("art-1", [1.0, 2.0])   # wrong dimensionality
except vv.DimensionMismatchError as e:
    print(e)

try:
    engine.load("missing.vv")
except FileNotFoundError:               # SnapshotNotFoundError subclasses this
    ...
```

See the [Python API reference](python-api.md) for the complete surface.
