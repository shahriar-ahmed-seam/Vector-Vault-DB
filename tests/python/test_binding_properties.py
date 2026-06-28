"""Property-based tests for the Vector-Vault-DB Python_Binding.

These tests exercise the binding-layer correctness properties (23-26) from
design.md using Hypothesis. Each property is implemented by exactly one
property-based test and runs a minimum of 100 examples.

The binding is imported as the public ``vectorvault`` package, which wraps the
compiled ``_vectorvault`` extension.
"""

import math
import struct

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis import HealthCheck

import vectorvault as vv


# ---------------------------------------------------------------------------
# Shared helpers / strategies
# ---------------------------------------------------------------------------

# Finite Python floats/ints the binding accepts as vector components. The range
# is kept well within float32's representable magnitude so the only lossy step
# is the float64 -> float32 narrowing performed by the binding (Req 9.2).
_numeric_component = st.one_of(
    st.integers(min_value=-100_000, max_value=100_000),
    st.floats(
        min_value=-1.0e6,
        max_value=1.0e6,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)


def _f32(value):
    """Round-trip a Python number through IEEE-754 float32, as the binding does."""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _new_collection(engine, name, dim, metric="euclidean"):
    return engine.create_collection(name, dim, metric)


# ===========================================================================
# Property 23
# ===========================================================================
# Feature: vector-vault-db, Property 23: Python sequence conversion round-trips through float32
# Validates: Requirements 9.2
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    data=st.data(),
    dim=st.integers(min_value=1, max_value=16),
    container=st.sampled_from(["list", "tuple", "ndarray"]),
)
def test_property_23_sequence_round_trips_through_float32(data, dim, container):
    values = data.draw(
        st.lists(_numeric_component, min_size=dim, max_size=dim),
        label="vector components",
    )

    if container == "list":
        vector = list(values)
    elif container == "tuple":
        vector = tuple(values)
    else:  # ndarray (float64 input exercises the narrowing path)
        vector = np.array(values, dtype=np.float64)

    engine = vv.Engine()
    collection = _new_collection(engine, "p23", dim)
    collection.insert("rec", vector)

    stored, meta = collection.get("rec")

    assert meta is None
    assert len(stored) == dim
    # The binding stores float32; getting back yields the float32 image of the
    # input, so each component must equal the float32 representation exactly.
    expected = [_f32(v) for v in values]
    assert stored == expected


# ===========================================================================
# Property 24
# ===========================================================================
# Feature: vector-vault-db, Property 24: Non-numeric vector elements raise TypeError identifying the first bad index
# Validates: Requirements 9.3
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_property_24_non_numeric_element_raises_typeerror_with_first_index(data):
    # Build a sequence of numeric components, then inject a non-numeric element
    # at a known first-bad index. Numeric values are placed before that index;
    # anything (numeric or not) may follow it.
    prefix_len = data.draw(st.integers(min_value=0, max_value=6), label="prefix length")
    prefix = data.draw(
        st.lists(_numeric_component, min_size=prefix_len, max_size=prefix_len),
        label="numeric prefix",
    )
    bad = data.draw(
        st.one_of(
            st.text(min_size=1, max_size=4),
            st.none(),
            st.binary(min_size=1, max_size=4),
            st.lists(st.integers(), min_size=1, max_size=2),
            st.builds(object),
            st.complex_numbers(allow_nan=False, allow_infinity=False).filter(
                lambda c: c.imag != 0
            ),
        ),
        label="first non-numeric element",
    )
    suffix = data.draw(
        st.lists(_numeric_component, min_size=0, max_size=4), label="numeric suffix"
    )

    vector = list(prefix) + [bad] + list(suffix)
    first_bad_index = prefix_len

    engine = vv.Engine()
    collection = _new_collection(engine, "p24", max(1, len(vector)))

    with pytest.raises(TypeError) as exc_info:
        collection.insert("rec", vector)

    message = str(exc_info.value)
    # The message must identify the zero-based index of the first bad element.
    assert str(first_bad_index) in message


# ===========================================================================
# Property 25
# ===========================================================================
# Feature: vector-vault-db, Property 25: Core error categories map to corresponding Python exceptions
# Validates: Requirements 9.5
#
# Each reachable Core_Engine error category is triggered through the binding and
# the specific mapped Python exception type plus a category-naming message are
# asserted. Hypothesis randomizes the inputs that drive each category (names,
# dimensions, identifiers, k values, paths) so the mapping is exercised across
# the input space rather than a single fixed example.
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    name=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=20,
    ),
    dim=st.integers(min_value=1, max_value=32),
    rec_id=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=10,
    ),
    bad_dim=st.integers(min_value=66000, max_value=200000),
    big_k=st.integers(min_value=-5, max_value=0),
    missing=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=10,
    ),
)
def test_property_25_error_categories_map_to_python_exceptions(
    name, dim, rec_id, bad_dim, big_k, missing
):
    engine = vv.Engine()
    collection = engine.create_collection(name, dim, "euclidean")
    collection.insert(rec_id, [0.0] * dim)

    def assert_category(exc_type, category, fn, also=None):
        with pytest.raises(exc_type) as info:
            fn()
        msg = str(info.value)
        # Message states the error category (Req 9.5).
        assert category in msg
        if also is not None:
            assert isinstance(info.value, also)

    # NameConflict -> NameConflictError (duplicate create)
    assert_category(
        vv.NameConflictError,
        "NameConflict",
        lambda: engine.create_collection(name, dim, "euclidean"),
    )
    # InvalidMetric -> InvalidMetricError (unknown metric string)
    assert_category(
        vv.InvalidMetricError,
        "InvalidMetric",
        lambda: engine.create_collection(name + "_m", dim, "not-a-metric"),
    )
    # InvalidDimensionality -> InvalidDimensionalityError (dim out of range)
    assert_category(
        vv.InvalidDimensionalityError,
        "InvalidDimensionality",
        lambda: engine.create_collection(name + "_d", bad_dim, "euclidean"),
    )
    # InvalidName -> InvalidNameError (empty name); also a ValueError
    assert_category(
        vv.InvalidNameError,
        "InvalidName",
        lambda: engine.create_collection("", dim, "euclidean"),
        also=ValueError,
    )
    # NotFound (missing collection) -> NotFoundError; also a KeyError
    assert_category(
        vv.NotFoundError,
        "NotFound",
        lambda: engine.get_collection(name + "_absent"),
        also=KeyError,
    )
    # NotFound (missing record) -> NotFoundError; also a KeyError
    assert_category(
        vv.NotFoundError,
        "NotFound",
        lambda: collection.get(missing + "_absent_record"),
        also=KeyError,
    )
    # DimensionMismatch -> DimensionMismatchError (wrong-length insert)
    assert_category(
        vv.DimensionMismatchError,
        "DimensionMismatch",
        lambda: collection.insert(rec_id + "_x", [0.0] * (dim + 1)),
    )
    # DimensionMismatch on query as well
    assert_category(
        vv.DimensionMismatchError,
        "DimensionMismatch",
        lambda: collection.query([0.0] * (dim + 1), 1),
    )
    # InvalidValue -> InvalidValueError (NaN component)
    assert_category(
        vv.InvalidValueError,
        "InvalidValue",
        lambda: collection.insert(rec_id + "_nan", [math.nan] + [0.0] * (dim - 1)),
    )
    # InvalidValue on query (inf component)
    assert_category(
        vv.InvalidValueError,
        "InvalidValue",
        lambda: collection.query([math.inf] + [0.0] * (dim - 1), 1),
    )
    # InvalidIdentifier -> InvalidIdentifierError (empty id)
    assert_category(
        vv.InvalidIdentifierError,
        "InvalidIdentifier",
        lambda: collection.insert("", [0.0] * dim),
    )
    # InvalidArgument -> InvalidArgumentError (k < 1)
    assert_category(
        vv.InvalidArgumentError,
        "InvalidArgument",
        lambda: collection.query([0.0] * dim, big_k),
    )
    # SnapshotNotFound -> SnapshotNotFoundError (missing path); also FileNotFoundError
    assert_category(
        vv.SnapshotNotFoundError,
        "SnapshotNotFound",
        lambda: engine.load("Z:/vectorvault/definitely/missing/" + name + ".vv"),
        also=FileNotFoundError,
    )
    # InvalidParameter -> InvalidParameterError (HNSW m below minimum)
    assert_category(
        vv.InvalidParameterError,
        "InvalidParameter",
        lambda: collection.build_index("hnsw", m=1),
    )


# ===========================================================================
# Property 26
# ===========================================================================
# Feature: vector-vault-db, Property 26: Query results surface as ordered (id, float) pairs through the binding
# Validates: Requirements 9.6
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    data=st.data(),
    dim=st.integers(min_value=1, max_value=8),
    n_records=st.integers(min_value=1, max_value=25),
    k=st.integers(min_value=1, max_value=30),
    build=st.booleans(),
)
def test_property_26_query_results_are_ordered_id_float_pairs(
    data, dim, n_records, k, build
):
    engine = vv.Engine()
    collection = _new_collection(engine, "p26", dim)

    for i in range(n_records):
        components = data.draw(
            st.lists(_numeric_component, min_size=dim, max_size=dim),
            label=f"record {i}",
        )
        collection.insert(f"id-{i}", components)

    if build:
        collection.build_index("hnsw")

    query_vec = data.draw(
        st.lists(_numeric_component, min_size=dim, max_size=dim), label="query vector"
    )
    results = collection.query(query_vec, k)

    # Result is a Python sequence of (str id, float distance) pairs.
    assert isinstance(results, list)
    assert len(results) <= k
    assert len(results) <= n_records

    prev_distance = None
    for entry in results:
        assert isinstance(entry, tuple)
        assert len(entry) == 2
        identifier, distance = entry
        assert isinstance(identifier, str)
        assert isinstance(distance, float)
        # Ordered by ascending (non-decreasing) distance.
        if prev_distance is not None:
            assert distance >= prev_distance
        prev_distance = distance
