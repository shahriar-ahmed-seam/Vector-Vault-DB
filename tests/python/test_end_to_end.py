"""End-to-end pipeline smoke test for the Python binding (spec: vector-vault-db,
task 13.1).

This drives the full Vector-Vault-DB pipeline through the *public* ``vectorvault``
Python API, confirming there are no orphaned components between the layers:

    Python binding -> Engine -> Collection -> Index + Distance + Allocator
                                                          -> Persistence (save)
    Persistence (mmap load) -> Collection -> query

The complete lifecycle exercised per metric/index combination is:

    create -> insert (many singles + one re-insert/overwrite) -> build_index
           -> query -> save -> load -> query-again

and the loaded collection's query results are asserted to match the pre-save
results exactly (same ids, same distances), proving the snapshot round-trip and
the mmap-backed read path reconstruct an equivalent, queryable collection.

The binding exposes single ``insert`` (Req 9.1 lists single-record insertion);
true batch insertion is exercised at the C++ layer in
``tests/cpp/end_to_end_test.cpp``. Here "many singles" populates the collection.
"""

import os

import pytest

import vectorvault


# A small, deterministic synthetic dataset: clustered around a few centers so an
# ANN index has structure to find and nearest-neighbor results are stable.
DIM = 16
CENTERS = [
    [float((c * 3 + i) % 7) for i in range(DIM)]
    for c in range(8)
]


def _make_vector(record_index):
    """A deterministic vector derived from a center plus a small per-record
    offset, so distinct records have distinct-but-related vectors."""
    center = CENTERS[record_index % len(CENTERS)]
    offset = (record_index % 5) * 0.01
    return [v + offset for v in center]


def _build_query_vector(seed):
    center = CENTERS[seed % len(CENTERS)]
    return [v + 0.001 for v in center]


def _populate(collection, count):
    """Insert ``count`` records as individual single inserts, then overwrite one
    id in place (Req 2.3) to exercise the update path. Returns the set of ids."""
    ids = []
    for i in range(count):
        rid = f"rec-{i:04d}"
        meta = {"idx": i, "tag": f"t{i % 4}", "active": (i % 2 == 0)}
        collection.insert(rid, _make_vector(i), meta)
        ids.append(rid)

    # Overwrite an existing id with a new vector + metadata; count must not grow.
    before = collection.count()
    collection.insert(ids[0], _make_vector(0), {"idx": 0, "tag": "overwritten"})
    assert collection.count() == before
    return ids


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "dot"])
@pytest.mark.parametrize("index_type", ["hnsw", "ivf"])
def test_full_pipeline_create_insert_build_query_save_load(tmp_path, metric, index_type):
    count = 120
    k = 10

    engine = vectorvault.Engine()

    # create ---------------------------------------------------------------
    name = f"e2e_{metric}_{index_type}"
    coll = engine.create_collection(name, DIM, metric)
    assert coll.count() == 0

    # insert (many singles + an in-place overwrite) ------------------------
    _populate(coll, count)
    assert coll.count() == count

    # build_index ----------------------------------------------------------
    if index_type == "hnsw":
        coll.build_index("hnsw", m=16, ef_construction=200)
    else:
        coll.build_index("ivf", nlist=8)

    # query ----------------------------------------------------------------
    query_vec = _build_query_vector(3)
    pre_save = coll.query(query_vec, k)

    # Result shape: ordered (id, float) pairs, non-increasing distance order.
    assert len(pre_save) == k
    for rid, dist in pre_save:
        assert isinstance(rid, str)
        assert isinstance(dist, float)
    distances = [d for _, d in pre_save]
    assert distances == sorted(distances), "results must be ascending by distance"

    # save -----------------------------------------------------------------
    snapshot_path = os.path.join(str(tmp_path), f"{name}.vv")
    engine.save(name, snapshot_path)
    assert os.path.exists(snapshot_path)

    # load (into a fresh engine, via the mmap-backed load path) ------------
    reload_engine = vectorvault.Engine()
    loaded = reload_engine.load(snapshot_path)

    # The reconstructed collection matches the original's shape.
    assert loaded.count() == count

    # query-again: loaded results must match the pre-save results exactly ---
    post_load = loaded.query(query_vec, k)
    assert post_load == pre_save, "loaded query results must match pre-save results"

    # A retrieved record's vector + metadata survive the round-trip.
    top_id = pre_save[0][0]
    vec, meta = loaded.get(top_id)
    assert len(vec) == DIM
    assert meta is not None


def test_pipeline_without_index_uses_bruteforce(tmp_path):
    """The full save/load/query lifecycle also works with no index built (the
    query path falls back to an exact brute-force scan)."""
    engine = vectorvault.Engine()
    coll = engine.create_collection("e2e_noindex", DIM, "euclidean")
    ids = _populate(coll, 40)

    query_vec = _build_query_vector(1)
    pre_save = coll.query(query_vec, 5)
    assert len(pre_save) == 5

    path = os.path.join(str(tmp_path), "noindex.vv")
    engine.save("e2e_noindex", path)

    reload_engine = vectorvault.Engine()
    loaded = reload_engine.load(path)
    assert loaded.count() == len(ids)
    assert loaded.query(query_vec, 5) == pre_save


def test_k_exceeds_count_returns_all_records(tmp_path):
    """When k exceeds the record count the query returns every record (Req 6.6),
    and this still round-trips through save/load."""
    engine = vectorvault.Engine()
    coll = engine.create_collection("e2e_small", DIM, "euclidean")
    _populate(coll, 6)
    coll.build_index("hnsw")

    result = coll.query(_make_vector(0), 100)
    assert len(result) == 6

    path = os.path.join(str(tmp_path), "small.vv")
    engine.save("e2e_small", path)
    loaded = vectorvault.Engine().load(path)
    assert len(loaded.query(_make_vector(0), 100)) == 6
