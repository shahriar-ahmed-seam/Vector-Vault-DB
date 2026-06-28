// Collection record-store tests: insert, insert_batch, get, remove, and count.
// Components are generated as float32 from a scaled integer range, so a
// round-trip preserves the exact generated value (no narrowing).

#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>

#include "vectorvault/engine.hpp"
#include "vectorvault/error.hpp"
#include "vectorvault/types.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace vectorvault;

namespace {

// Small dimensionality keeps each case cheap while still exercising the
// multi-component copy path. Any value in [1, 32] is a valid dimensionality.
std::uint32_t gen_small_dim() {
    return *rc::gen::inRange<std::uint32_t>(1, 33);  // [1, 32]
}

// One of the three supported metrics. The record-store operations are
// metric-agnostic; varying it guards against accidental coupling.
DistanceMetric gen_metric() {
    return *rc::gen::element(DistanceMetric::Euclidean,
                             DistanceMetric::Cosine,
                             DistanceMetric::DotProduct);
}

// A non-empty record identifier: 1 to 12 printable ASCII characters.
std::string gen_id() {
    const std::size_t len = *rc::gen::inRange<std::size_t>(1, 13);
    return *rc::gen::container<std::string>(
        len, rc::gen::inRange<char>(static_cast<char>(33),
                                    static_cast<char>(127)));  // '!'..'~'
}

// A finite float32 component from a scaled integer range. Generating directly as
// float32 means the stored block holds the same bits, so the round-trip is exact.
const auto kComponentGen = rc::gen::map(
    rc::gen::inRange<int>(-1000000, 1000001),
    [](int v) { return static_cast<float>(v) / 1000.0f; });

// A vector of `dim` finite float32 components.
std::vector<float> gen_vector(std::uint32_t dim) {
    return *rc::gen::container<std::vector<float>>(
        static_cast<std::size_t>(dim), kComponentGen);
}

// A single metadata value: a string, int64, finite double, or bool. The double
// is drawn from a scaled integer range so it round-trips exactly (no NaN).
MetadataValue gen_metadata_value() {
    const int which = *rc::gen::inRange<int>(0, 4);
    switch (which) {
        case 0: {
            const std::size_t len = *rc::gen::inRange<std::size_t>(0, 8);
            return MetadataValue{*rc::gen::container<std::string>(
                len, rc::gen::inRange<char>(static_cast<char>(33),
                                            static_cast<char>(127)))};
        }
        case 1:
            return MetadataValue{*rc::gen::arbitrary<std::int64_t>()};
        case 2:
            return MetadataValue{
                static_cast<double>(*rc::gen::inRange<int>(-1000000, 1000001)) /
                1000.0};
        default:
            return MetadataValue{*rc::gen::arbitrary<bool>()};
    }
}

// A metadata map of 0 to 4 entries with canonical (sorted) string keys.
Metadata gen_metadata() {
    const std::size_t entries = *rc::gen::inRange<std::size_t>(0, 5);
    Metadata meta;
    for (std::size_t i = 0; i < entries; ++i) {
        const std::size_t klen = *rc::gen::inRange<std::size_t>(1, 8);
        std::string key = *rc::gen::container<std::string>(
            klen, rc::gen::inRange<char>(static_cast<char>(33),
                                         static_cast<char>(127)));
        meta[std::move(key)] = gen_metadata_value();
    }
    return meta;
}

// Optional metadata: either a (possibly empty) map or an explicit absence.
std::optional<Metadata> gen_optional_metadata() {
    if (*rc::gen::arbitrary<bool>()) {
        return gen_metadata();
    }
    return std::nullopt;
}

// Creates a fresh single-collection Engine for one property case and returns
// the handle.
CollectionHandle make_collection(Engine& engine, std::uint32_t dim,
                                 DistanceMetric metric) {
    auto result = engine.create_collection("records", dim, metric);
    RC_ASSERT(result.is_ok());
    return result.value();
}

}  // namespace

// Feature: vector-vault-db, Property 5: Insert/get round-trip preserves vector and metadata
// Validates: Requirements 2.1, 3.1, 3.2
TEST_CASE("Property 5: insert/get round-trip preserves vector and metadata",
          "[collection][records][property]") {
    const bool ok = rc::check(
        "after inserting a valid record, get returns the same vector and the "
        "same metadata (present or explicitly absent)",
        [] {
            const std::uint32_t  dim    = gen_small_dim();
            const DistanceMetric metric = gen_metric();
            const std::string    id     = gen_id();
            const std::vector<float>      vec  = gen_vector(dim);
            const std::optional<Metadata> meta = gen_optional_metadata();

            Engine engine;
            CollectionHandle coll = make_collection(engine, dim, metric);

            RC_ASSERT(coll->insert(id, vec, meta).is_ok());
            RC_ASSERT(coll->count() == 1);

            auto got = coll->get(id);
            RC_ASSERT(got.is_ok());

            // The vector equals the float32 representation of the input.
            RC_ASSERT(got.value().vector == vec);

            // Metadata matches, including an explicit absence when none supplied.
            RC_ASSERT(got.value().meta.has_value() == meta.has_value());
            RC_ASSERT(got.value().meta == meta);
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 6: Dimension-mismatched insertion is rejected without side effects
// Validates: Requirements 2.2
TEST_CASE("Property 6: dimension-mismatched insertion is rejected without side effects",
          "[collection][records][property]") {
    const bool ok = rc::check(
        "inserting a vector whose length differs from the dimensionality "
        "returns DimensionMismatch and leaves the count and contents unchanged",
        [] {
            const std::uint32_t  dim    = gen_small_dim();
            const DistanceMetric metric = gen_metric();

            Engine engine;
            CollectionHandle coll = make_collection(engine, dim, metric);

            // A baseline valid record gives observable state to verify the
            // rejected insert leaves untouched.
            const std::string        base_id  = "baseline";
            const std::vector<float> base_vec = gen_vector(dim);
            RC_ASSERT(coll->insert(base_id, base_vec, std::nullopt).is_ok());
            RC_ASSERT(coll->count() == 1);

            // A vector whose component count differs from the dimensionality.
            const std::size_t wrong_size = *rc::gen::suchThat(
                rc::gen::inRange<std::size_t>(0, static_cast<std::size_t>(dim) * 2 + 4),
                [dim](std::size_t s) { return s != dim; });
            const std::vector<float> wrong_vec = *rc::gen::container<std::vector<float>>(
                wrong_size, kComponentGen);

            const auto status = coll->insert("candidate", wrong_vec, std::nullopt);
            RC_ASSERT(status.is_error());
            RC_ASSERT(status.category() == ErrorCategory::DimensionMismatch);

            // No side effects: count unchanged, baseline intact, candidate absent.
            RC_ASSERT(coll->count() == 1);
            auto base = coll->get(base_id);
            RC_ASSERT(base.is_ok());
            RC_ASSERT(base.value().vector == base_vec);
            RC_ASSERT(coll->get("candidate").is_error());
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 7: Re-inserting an existing id overwrites without changing count
// Validates: Requirements 2.3
TEST_CASE("Property 7: re-inserting an existing id overwrites without changing count",
          "[collection][records][property]") {
    const bool ok = rc::check(
        "inserting a record whose id already exists replaces the stored vector "
        "and metadata and leaves the record count unchanged",
        [] {
            const std::uint32_t  dim    = gen_small_dim();
            const DistanceMetric metric = gen_metric();
            const std::string    id     = gen_id();

            Engine engine;
            CollectionHandle coll = make_collection(engine, dim, metric);

            const std::vector<float>      vec1  = gen_vector(dim);
            const std::optional<Metadata> meta1 = gen_optional_metadata();
            RC_ASSERT(coll->insert(id, vec1, meta1).is_ok());
            RC_ASSERT(coll->count() == 1);

            // Re-insert the same id with fresh contents.
            const std::vector<float>      vec2  = gen_vector(dim);
            const std::optional<Metadata> meta2 = gen_optional_metadata();
            RC_ASSERT(coll->insert(id, vec2, meta2).is_ok());

            // The count is unchanged ...
            RC_ASSERT(coll->count() == 1);

            // ... and the stored vector and metadata are the new values.
            auto got = coll->get(id);
            RC_ASSERT(got.is_ok());
            RC_ASSERT(got.value().vector == vec2);
            RC_ASSERT(got.value().meta == meta2);
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 8: Non-finite components are rejected without side effects
// Validates: Requirements 2.4
TEST_CASE("Property 8: non-finite components are rejected without side effects",
          "[collection][records][property]") {
    const float kInf = std::numeric_limits<float>::infinity();
    const float kNan = std::numeric_limits<float>::quiet_NaN();

    const bool ok = rc::check(
        "inserting a vector with at least one non-finite component returns "
        "InvalidValue and leaves the count and contents unchanged",
        [&] {
            const std::uint32_t  dim    = gen_small_dim();
            const DistanceMetric metric = gen_metric();

            Engine engine;
            CollectionHandle coll = make_collection(engine, dim, metric);

            const std::string        base_id  = "baseline";
            const std::vector<float> base_vec = gen_vector(dim);
            RC_ASSERT(coll->insert(base_id, base_vec, std::nullopt).is_ok());
            RC_ASSERT(coll->count() == 1);

            // A valid vector with one component overwritten by a non-finite
            // value at an arbitrary position.
            std::vector<float> bad_vec = gen_vector(dim);
            const std::size_t pos =
                *rc::gen::inRange<std::size_t>(0, static_cast<std::size_t>(dim));
            const float bad_value = *rc::gen::element(kNan, kInf, -kInf);
            bad_vec[pos] = bad_value;

            const auto status = coll->insert("candidate", bad_vec, std::nullopt);
            RC_ASSERT(status.is_error());
            RC_ASSERT(status.category() == ErrorCategory::InvalidValue);

            // No side effects.
            RC_ASSERT(coll->count() == 1);
            auto base = coll->get(base_id);
            RC_ASSERT(base.is_ok());
            RC_ASSERT(base.value().vector == base_vec);
            RC_ASSERT(coll->get("candidate").is_error());
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 9: Valid batches store every record
// Validates: Requirements 2.5
TEST_CASE("Property 9: valid batches store every record",
          "[collection][records][property]") {
    const bool ok = rc::check(
        "a batch of valid records stores every record so each is retrievable "
        "by its id",
        [] {
            const std::uint32_t  dim    = gen_small_dim();
            const DistanceMetric metric = gen_metric();

            Engine engine;
            CollectionHandle coll = make_collection(engine, dim, metric);

            const std::size_t batch_size = *rc::gen::inRange<std::size_t>(1, 33);
            std::vector<RecordInput> batch;
            batch.reserve(batch_size);

            // Expected store after the batch: last-writer-wins on duplicate ids,
            // mirroring repeated single inserts.
            std::map<RecordId, RecordInput> expected;
            for (std::size_t i = 0; i < batch_size; ++i) {
                RecordInput rec;
                rec.id     = gen_id();
                rec.vector = gen_vector(dim);
                rec.meta   = gen_optional_metadata();
                expected[rec.id] = rec;
                batch.push_back(std::move(rec));
            }

            RC_ASSERT(coll->insert_batch(batch).is_ok());

            // Every distinct id is stored exactly once.
            RC_ASSERT(coll->count() == expected.size());

            // Each record is retrievable with the last-written vector/metadata.
            for (const auto& [id, rec] : expected) {
                auto got = coll->get(id);
                RC_ASSERT(got.is_ok());
                RC_ASSERT(got.value().vector == rec.vector);
                RC_ASSERT(got.value().meta == rec.meta);
            }
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 10: Invalid batches are rejected atomically
// Validates: Requirements 2.6
// One record in an otherwise-valid batch is corrupted (empty id, dimension
// mismatch, or non-finite component); being the only invalid record it is the
// first, so the returned error category must match its kind and nothing is stored.
TEST_CASE("Property 10: invalid batches are rejected atomically",
          "[collection][records][property]") {
    const float kInf = std::numeric_limits<float>::infinity();

    const bool ok = rc::check(
        "a batch containing one invalid record stores nothing and returns the "
        "first invalid record's error category",
        [&] {
            // Dimensionality >= 2 so a dimension-mismatch corruption (dim - 1
            // components) still leaves a non-empty vector to fill.
            const std::uint32_t  dim    = *rc::gen::inRange<std::uint32_t>(2, 33);
            const DistanceMetric metric = gen_metric();

            Engine engine;
            CollectionHandle coll = make_collection(engine, dim, metric);

            const std::size_t batch_size = *rc::gen::inRange<std::size_t>(1, 33);
            std::vector<RecordInput> batch;
            batch.reserve(batch_size);
            for (std::size_t i = 0; i < batch_size; ++i) {
                RecordInput rec;
                rec.id     = "id-" + std::to_string(i);  // unique, non-empty
                rec.vector = gen_vector(dim);
                rec.meta   = std::nullopt;
                batch.push_back(std::move(rec));
            }

            // Corrupt exactly one record with a randomly chosen failure kind.
            const std::size_t bad_index =
                *rc::gen::inRange<std::size_t>(0, batch_size);
            const int kind = *rc::gen::inRange<int>(0, 3);
            ErrorCategory expected_category = ErrorCategory::InvalidIdentifier;
            switch (kind) {
                case 0:  // empty identifier
                    batch[bad_index].id = "";
                    expected_category = ErrorCategory::InvalidIdentifier;
                    break;
                case 1:  // dimension mismatch: drop one component
                    batch[bad_index].vector.pop_back();
                    expected_category = ErrorCategory::DimensionMismatch;
                    break;
                default:  // non-finite component
                    batch[bad_index].vector[0] = kInf;
                    expected_category = ErrorCategory::InvalidValue;
                    break;
            }

            const auto status = coll->insert_batch(batch);
            RC_ASSERT(status.is_error());
            RC_ASSERT(status.category() == expected_category);

            // Atomic rejection: nothing stored.
            RC_ASSERT(coll->count() == 0);
            for (const RecordInput& rec : batch) {
                if (!rec.id.empty()) {
                    RC_ASSERT(coll->get(rec.id).is_error());
                }
            }
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 11: Empty identifiers are rejected
// Validates: Requirements 2.8
TEST_CASE("Property 11: empty identifiers are rejected",
          "[collection][records][property]") {
    const bool ok = rc::check(
        "inserting a record with an empty identifier returns InvalidIdentifier "
        "and leaves the collection unchanged",
        [] {
            const std::uint32_t  dim    = gen_small_dim();
            const DistanceMetric metric = gen_metric();
            const std::vector<float>      vec  = gen_vector(dim);
            const std::optional<Metadata> meta = gen_optional_metadata();

            Engine engine;
            CollectionHandle coll = make_collection(engine, dim, metric);
            RC_ASSERT(coll->count() == 0);

            const auto status = coll->insert("", vec, meta);
            RC_ASSERT(status.is_error());
            RC_ASSERT(status.category() == ErrorCategory::InvalidIdentifier);

            // The collection is unchanged: still empty and the empty id absent.
            RC_ASSERT(coll->count() == 0);
            RC_ASSERT(coll->get("").is_error());
        });
    REQUIRE(ok);
}

// Unit tests: record edge cases.

// Inserting into a Collection that does not exist is a not-found condition,
// enforced at the Engine: resolving a missing collection returns NotFound, so a
// caller never obtains a handle to insert into.
TEST_CASE("inserting into a nonexistent collection is rejected with NotFound",
          "[collection][records][unit]") {
    Engine engine;

    auto missing = engine.get_collection("does-not-exist");
    REQUIRE(missing.is_error());
    REQUIRE(missing.category() == ErrorCategory::NotFound);

    // A real collection is still reachable, confirming the lookup itself works.
    REQUIRE(engine.create_collection("present", 8, DistanceMetric::Euclidean).is_ok());
    REQUIRE(engine.get_collection("present").is_ok());
    REQUIRE(engine.get_collection("absent").is_error());
}

// Getting a record by an id that does not exist returns NotFound and leaves the
// collection unchanged.
TEST_CASE("get of a nonexistent id returns NotFound and leaves the collection unchanged",
          "[collection][records][unit]") {
    Engine engine;
    auto created = engine.create_collection("c", 4, DistanceMetric::Euclidean);
    REQUIRE(created.is_ok());
    CollectionHandle coll = created.value();

    const std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    REQUIRE(coll->insert("present", vec, std::nullopt).is_ok());
    REQUIRE(coll->count() == 1);

    auto got = coll->get("missing");
    REQUIRE(got.is_error());
    REQUIRE(got.category() == ErrorCategory::NotFound);

    // The collection is unchanged: count intact and the existing record intact.
    REQUIRE(coll->count() == 1);
    auto present = coll->get("present");
    REQUIRE(present.is_ok());
    REQUIRE(present.value().vector == vec);
}

// Deleting a record by an id that does not exist returns NotFound and leaves the
// record count and contents unchanged.
TEST_CASE("delete of a nonexistent id returns NotFound and leaves the collection unchanged",
          "[collection][records][unit]") {
    Engine engine;
    auto created = engine.create_collection("c", 4, DistanceMetric::Euclidean);
    REQUIRE(created.is_ok());
    CollectionHandle coll = created.value();

    const std::vector<float> vec = {5.0f, 6.0f, 7.0f, 8.0f};
    REQUIRE(coll->insert("present", vec, std::nullopt).is_ok());
    REQUIRE(coll->count() == 1);

    auto removed = coll->remove("missing");
    REQUIRE(removed.is_error());
    REQUIRE(removed.category() == ErrorCategory::NotFound);

    // The count and contents are unchanged.
    REQUIRE(coll->count() == 1);
    auto present = coll->get("present");
    REQUIRE(present.is_ok());
    REQUIRE(present.value().vector == vec);
}

// A freshly created collection that contains zero records reports a count of 0.
TEST_CASE("an empty collection reports a count of 0",
          "[collection][records][unit]") {
    Engine engine;
    auto created = engine.create_collection("empty", 16, DistanceMetric::Cosine);
    REQUIRE(created.is_ok());
    REQUIRE(created.value()->count() == 0);
}
