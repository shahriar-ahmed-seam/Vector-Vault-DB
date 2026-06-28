"""Example-based unit tests for the Vector-Vault-DB Python_Binding surface.

Covers the binding-surface acceptance criteria that are example/edge-case
shaped rather than universal properties:

* Req 9.4 - passing a non-sequence where a vector is expected raises TypeError.
* Req 9.1 - the binding exposes every required operation on ``Engine`` and
  ``Collection`` (create collection, insert, delete, build index, query, save,
  load).
"""

import pytest

import vectorvault as vv


# ---------------------------------------------------------------------------
# Req 9.1 - required operations are exposed
# ---------------------------------------------------------------------------

# Operations Req 9.1 requires the binding to expose: create a Collection,
# insert a Vector_Record, delete a Vector_Record, build an Index, run a kNN
# query, save a Snapshot, and load a Snapshot. The public surface spreads these
# across Engine and Collection.
ENGINE_OPERATIONS = [
    "create_collection",
    "get_collection",
    "list_collections",
    "delete_collection",
    "save",
    "load",
]

COLLECTION_OPERATIONS = [
    "insert",
    "get",
    "delete",
    "build_index",
    "query",
    "count",
    "index_type",
]


@pytest.mark.parametrize("operation", ENGINE_OPERATIONS)
def test_engine_exposes_required_operation(operation):
    assert hasattr(vv.Engine, operation)
    assert callable(getattr(vv.Engine, operation))


@pytest.mark.parametrize("operation", COLLECTION_OPERATIONS)
def test_collection_exposes_required_operation(operation):
    assert hasattr(vv.Collection, operation)
    assert callable(getattr(vv.Collection, operation))


def test_required_operations_run_end_to_end(tmp_path):
    # Smoke the full required-operation set: create -> insert -> build_index ->
    # query -> save -> load -> delete, proving each exposed operation is wired.
    engine = vv.Engine()
    collection = engine.create_collection("surface", 3, "euclidean")

    collection.insert("a", [1.0, 0.0, 0.0])
    collection.insert("b", [0.0, 1.0, 0.0])
    assert collection.count() == 2

    collection.build_index("hnsw")
    results = collection.query([1.0, 0.0, 0.0], 1)
    assert results[0][0] == "a"

    snapshot = tmp_path / "surface.vv"
    engine.save("surface", str(snapshot))
    assert snapshot.exists()

    # Load into a fresh engine so the reconstructed collection name does not
    # collide with the one already registered above.
    other_engine = vv.Engine()
    loaded = other_engine.load(str(snapshot))
    assert loaded.count() == 2

    collection.delete("a")
    assert collection.count() == 1


# ---------------------------------------------------------------------------
# Req 9.4 - non-sequence vector argument raises TypeError
# ---------------------------------------------------------------------------

NON_SEQUENCE_VECTORS = [
    pytest.param(5, id="int"),
    pytest.param(3.14, id="float"),
    pytest.param(None, id="none"),
    pytest.param(True, id="bool"),
    pytest.param({"a": 1}, id="dict"),
    pytest.param(object(), id="object"),
]


@pytest.fixture()
def collection():
    engine = vv.Engine()
    return engine.create_collection("nonseq", 3, "euclidean")


@pytest.mark.parametrize("bad_vector", NON_SEQUENCE_VECTORS)
def test_insert_with_non_sequence_vector_raises_typeerror(collection, bad_vector):
    with pytest.raises(TypeError):
        collection.insert("rec", bad_vector)


@pytest.mark.parametrize("bad_vector", NON_SEQUENCE_VECTORS)
def test_query_with_non_sequence_vector_raises_typeerror(collection, bad_vector):
    with pytest.raises(TypeError):
        collection.query(bad_vector, 1)
