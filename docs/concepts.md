# Concepts

## Collections

A collection is a named container for vectors that share a fixed
**dimensionality** and a single **distance metric**. Both are set at creation
and immutable thereafter. Each record is a unique string id, a float32 vector,
and optional metadata.

Constraints enforced at the boundary:

- Collection name: 1–255 characters, unique within an engine.
- Dimensionality: an integer in `[1, 65536]`.
- Record id: non-empty string.
- Vector: length equals the collection dimensionality; all components finite
  (no NaN or infinity).
- Metadata: a map of string keys to str / int / float / bool values.

## Distance metrics

| Metric | String | Definition | Ordering |
|---|---|---|---|
| Euclidean (L2) | `"euclidean"`, `"l2"` | √Σ(aᵢ−bᵢ)² | nearest = smallest |
| Cosine | `"cosine"` | 1 − (a·b)/(‖a‖‖b‖), in [0, 2] | nearest = smallest |
| Dot product | `"dot"`, `"dot_product"` | Σ aᵢbᵢ | see note below |

Query results are always ordered by **ascending metric value**, with ties
broken by ascending id for determinism.

- For **Euclidean** and **Cosine**, a smaller value means more similar, so
  ascending order returns the nearest neighbors first — the expected behavior.
- For **Dot product**, the raw dot product is treated as the distance and sorted
  ascending, so the largest dot products come *last*. If you want
  maximum-inner-product semantics (largest dot first), use cosine on normalized
  vectors, or reverse the result list yourself.

Cosine distance is undefined for a zero-magnitude vector; computing it raises
`UndefinedDistanceError`.

## Indexes

An index makes nearest-neighbor search sublinear. With no index, queries fall
back to an exact brute-force scan, which is correct but O(n) per query.

### HNSW (Hierarchical Navigable Small World)

A layered proximity graph. Search descends through sparse upper layers to a good
entry point, then explores the base layer. Neighbor selection uses a diversity
heuristic so the graph stays navigable.

Build parameters:

- `m` (default 16): max neighbors per node on upper layers (base layer keeps
  `2*m`). Higher `m` improves recall and memory use. Must be ≥ 2.
- `ef_construction` (default 200): build-time search breadth. Higher values build
  a better graph more slowly. Must be ≥ 1.

Query parameter:

- `ef_search` (default 50): search breadth. Higher values trade speed for recall.
  Must be ≥ 1.

### IVF (Inverted File)

Partitions vectors into `nlist` cells via k-means. A query scans only the
`nprobe` cells nearest to it.

Build parameter:

- `nlist` (default 0 = auto, ≈ √n): number of partitions. Must be in
  `[1, record_count]`.

Query parameter:

- `nprobe` (default 1): partitions scanned per query. Higher values trade speed
  for recall. Must be in `[1, nlist]`.

### Choosing an index

- **HNSW** generally gives higher recall at a given speed and is a good default.
- **IVF** has lower memory overhead and a smaller build cost; recall is tuned
  primarily through `nprobe`.

Deletions are handled by tombstoning: a removed record is excluded from results
immediately and the surrounding structure is preserved.

## Persistence

`engine.save(name, path)` writes a single binary snapshot containing the
collection metadata, all records (with metadata), and the index. The format is:

- A fixed header: magic, format version, dimensionality, metric, index type,
  region offsets, and a CRC-64 of the content.
- The collection name.
- A record directory sorted by id, each with optional metadata.
- A 64-byte aligned vector region.
- The serialized index (if one was built).

Records, metadata keys, and index nodes are written in a canonical order, so
saving a collection, loading it, and saving again produces a **byte-identical**
file.

`engine.load(path)` validates the file in order — existence and size, magic and
header, format version, then content checksum — before mapping the vector region
into memory. Vector data is read through the mapping rather than copied onto the
heap. Failures raise `SnapshotNotFoundError`, `UnsupportedVersionError`, or
`CorruptionError`.

## Performance notes

- Distance kernels use AVX-512 when the CPU supports it and fall back to scalar
  code otherwise; both accumulate in double precision for accuracy.
- Vector storage is 64-byte aligned to suit SIMD access.
- A collection is guarded by a readers-writer lock: concurrent reads are allowed;
  mutations are serialized.
