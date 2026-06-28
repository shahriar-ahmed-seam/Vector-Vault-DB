"""Tests for the Vector-Vault-DB ``Collection.insert_batch`` binding.

``insert_batch`` mirrors ``insert``: it accepts an iterable of records, each a
2-tuple ``(id, vector)`` or a 3-tuple ``(id, vector, metadata)``. Vectors go
through the same float32 conversion as ``insert`` (lists, tuples, and NumPy
arrays of int/float all work), metadata is converted the same way, and the
whole batch is applied atomically: if any record is invalid, nothing is
stored and the matching exception from the existing hierarchy is raised.
"""

import math

import numpy as np
import pytest

import vectorvault as vv


@pytest.fixture()
def collection():
    engine = vv.Engine()
    return engine.create_collection("batch", 3, "euclidean")


def _vectors_equal(stored, expected):
    assert len(stored) == len(expected)
    for a, b in zip(stored, expected):
        assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Happy path: pairs and triples are fully stored and retrievable
# ---------------------------------------------------------------------------

def test_batch_of_pairs_and_triples_is_fully_stored(collection):
    records = [
        ("a", [1.0, 0.0, 0.0]),
        ("b", [0.0, 1.0, 0.0], {"label": "second", "rank": 2}),
        ("c", (0.0, 0.0, 1.0), None),
    ]
    collection.insert_batch(records)

    assert collection.count() == 3

    vec_a, meta_a = collection.get("a")
    _vectors_equal(vec_a, [1.0, 0.0, 0.0])
    assert meta_a is None

    vec_b, meta_b = collection.get("b")
    _vectors_equal(vec_b, [0.0, 1.0, 0.0])
    assert meta_b == {"label": "second", "rank": 2}

    vec_c, meta_c = collection.get("c")
    _vectors_equal(vec_c, [0.0, 0.0, 1.0])
    assert meta_c is None


def test_batch_accepts_integer_elements(collection):
    collection.insert_batch([("i", [1, 2, 3])])
    vec, _ = collection.get("i")
    _vectors_equal(vec, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# NumPy-array vectors work in a batch
# ---------------------------------------------------------------------------

def test_batch_with_numpy_vectors(collection):
    records = [
        ("n0", np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        ("n1", np.array([4, 5, 6], dtype=np.int64), {"src": "numpy"}),
    ]
    collection.insert_batch(records)

    assert collection.count() == 2
    vec0, _ = collection.get("n0")
    _vectors_equal(vec0, [1.0, 2.0, 3.0])
    vec1, meta1 = collection.get("n1")
    _vectors_equal(vec1, [4.0, 5.0, 6.0])
    assert meta1 == {"src": "numpy"}


# ---------------------------------------------------------------------------
# Atomicity: an invalid record stores nothing
# ---------------------------------------------------------------------------

def test_wrong_dimension_in_batch_is_atomic(collection):
    records = [
        ("ok", [1.0, 2.0, 3.0]),
        ("bad", [1.0, 2.0]),  # wrong dimensionality
    ]
    with pytest.raises(vv.DimensionMismatchError):
        collection.insert_batch(records)
    # Nothing was stored.
    assert collection.count() == 0
    with pytest.raises(KeyError):
        collection.get("ok")


def test_non_finite_in_batch_is_atomic(collection):
    records = [
        ("ok", [1.0, 2.0, 3.0]),
        ("bad", [1.0, math.inf, 3.0]),  # non-finite component
    ]
    with pytest.raises(vv.InvalidValueError):
        collection.insert_batch(records)
    assert collection.count() == 0


def test_empty_id_in_batch_is_atomic(collection):
    records = [
        ("ok", [1.0, 2.0, 3.0]),
        ("", [4.0, 5.0, 6.0]),  # empty identifier
    ]
    with pytest.raises(vv.InvalidIdentifierError):
        collection.insert_batch(records)
    assert collection.count() == 0


# ---------------------------------------------------------------------------
# Batch-size bounds
# ---------------------------------------------------------------------------

def test_empty_batch_raises_invalid_argument(collection):
    with pytest.raises(vv.InvalidArgumentError):
        collection.insert_batch([])
    assert collection.count() == 0


def test_oversized_batch_raises_invalid_argument(collection):
    records = [(str(i), [float(i), 0.0, 0.0]) for i in range(10001)]
    with pytest.raises(vv.InvalidArgumentError):
        collection.insert_batch(records)
    assert collection.count() == 0


# ---------------------------------------------------------------------------
# Malformed record shapes raise TypeError before any core work
# ---------------------------------------------------------------------------

def test_non_sequence_record_raises_typeerror(collection):
    with pytest.raises(TypeError):
        collection.insert_batch([("only_id",)])
    assert collection.count() == 0


def test_non_numeric_vector_element_raises_typeerror(collection):
    with pytest.raises(TypeError):
        collection.insert_batch([("x", [1.0, "two", 3.0])])
    assert collection.count() == 0
