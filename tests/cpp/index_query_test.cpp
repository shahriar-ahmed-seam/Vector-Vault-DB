// Index and query tests: build_index parameter validation, searchability of
// inserted records, deletion exclusion, and brute-force query ordering/distance
// accuracy. Small integer-valued components keep float32 reductions exact so an
// in-test reference reproduces the engine's distances deterministically.

#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>

#include "vectorvault/distance.hpp"
#include "vectorvault/engine.hpp"
#include "vectorvault/error.hpp"
#include "vectorvault/index.hpp"
#include "vectorvault/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

using namespace vectorvault;

namespace {

// One of the three supported metrics.
DistanceMetric gen_metric() {
    return *rc::gen::element(DistanceMetric::Euclidean,
                             DistanceMetric::Cosine,
                             DistanceMetric::DotProduct);
}

// A small integer-valued float32 component in [-10, 10]; integer components
// keep the float32 reductions exact for the small dimensionalities used here.
const auto kComponentGen = rc::gen::map(
    rc::gen::inRange<int>(-10, 11), [](int v) { return static_cast<float>(v); });

// A vector of `dim` integer-valued float32 components.
std::vector<float> gen_vector(std::uint32_t dim) {
    return *rc::gen::container<std::vector<float>>(
        static_cast<std::size_t>(dim), kComponentGen);
}

// Sum of element-wise products in double (exact for the small integer inputs).
// Used to detect zero-norm vectors for the cosine metric.
double dot_double(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

// For cosine, force a non-zero L2 norm so the metric is defined. A zero vector
// becomes a unit vector along the first axis.
void ensure_nonzero_norm(std::vector<float>& v) {
    if (dot_double(v, v) == 0.0) {
        v[0] = 1.0f;
    }
}

// Independent scalar reference distance built from the documented float32 scalar
// kernels, separate from the query path's DistanceCalculator and so a genuine
// oracle. Cosine is combined in float32 just as the engine does, so the
// 1 - cos cancellation is shared rather than attributed to the engine.
float reference_distance(DistanceMetric metric,
                         const std::vector<float>& q,
                         const std::vector<float>& v) {
    const std::size_t n = q.size();
    switch (metric) {
        case DistanceMetric::Euclidean:
            return std::sqrt(kernels::scalar_l2_squared(q.data(), v.data(), n));
        case DistanceMetric::DotProduct:
            return kernels::scalar_dot(q.data(), v.data(), n);
        case DistanceMetric::Cosine: {
            const float na = kernels::scalar_dot(q.data(), q.data(), n);
            const float nb = kernels::scalar_dot(v.data(), v.data(), n);
            const float d  = kernels::scalar_dot(q.data(), v.data(), n);
            const float denom = std::sqrt(na) * std::sqrt(nb);
            float cd = 1.0f - d / denom;
            if (cd < 0.0f) cd = 0.0f;
            if (cd > 2.0f) cd = 2.0f;
            return cd;
        }
    }
    return 0.0f;  // unreachable
}

// Distance tolerance: 1e-4 relative error, falling back to 1e-4 absolute error
// when the reference magnitude is below 1e-4.
bool within_tolerance(double actual, double reference) {
    constexpr double kTol = 1e-4;
    const double diff   = std::fabs(actual - reference);
    const double ref_ab = std::fabs(reference);
    if (ref_ab < kTol) {
        return diff <= kTol;
    }
    return (diff / ref_ab) <= kTol;
}

// Stable, unique record identifiers r0, r1, ... Unique ids give the
// ascending-id tie-break a total order shared by the engine and the reference.
std::string make_id(std::size_t i) { return "r" + std::to_string(i); }

}  // namespace

// Feature: vector-vault-db, Property 17: Index parameter validation accepts in-range and rejects out-of-range values
// Validates: Requirements 5.4, 5.5, 5.6
// Builds a baseline HNSW index, then exercises a valid HNSW build, a valid IVF
// build, or an out-of-range build; rejections must leave the baseline intact.
TEST_CASE("Property 17: index parameter validation accepts in-range and rejects out-of-range values",
          "[index][property]") {
    const bool ok = rc::check(
        "in-range index parameters are accepted; out-of-range values are "
        "rejected with InvalidParameter and leave any existing index unchanged",
        [] {
            const std::uint32_t dim   = *rc::gen::inRange<std::uint32_t>(2, 9);
            const std::size_t   count = *rc::gen::inRange<std::size_t>(2, 13);

            Engine engine;
            auto created = engine.create_collection("idx", dim,
                                                    DistanceMetric::Euclidean);
            RC_ASSERT(created.is_ok());
            CollectionHandle coll = created.value();

            // Populate the collection so IVF's nlist range [1, count] is
            // non-trivial and a query has records to return.
            std::vector<std::vector<float>> vectors;
            vectors.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                std::vector<float> v = gen_vector(dim);
                RC_ASSERT(coll->insert(make_id(i), v, std::nullopt).is_ok());
                vectors.push_back(std::move(v));
            }
            RC_ASSERT(coll->count() == count);

            // Build a known-good baseline HNSW index. This is the "existing
            // index" that a later rejection must leave unchanged.
            IndexParams baseline;
            baseline.m               = 16;
            baseline.ef_construction = 200;
            RC_ASSERT(coll->build_index(IndexType::HNSW, baseline).is_ok());
            RC_ASSERT(coll->index_type() == IndexType::HNSW);

            const int scenario = *rc::gen::inRange<int>(0, 5);
            switch (scenario) {
                case 0: {
                    // Valid HNSW: m in [2, 64], ef_construction >= 1.
                    IndexParams p;
                    p.m               = *rc::gen::inRange<std::uint32_t>(2, 65);
                    p.ef_construction = *rc::gen::inRange<std::uint32_t>(1, 501);
                    RC_ASSERT(coll->build_index(IndexType::HNSW, p).is_ok());
                    RC_ASSERT(coll->index_type() == IndexType::HNSW);
                    break;
                }
                case 1: {
                    // Valid IVF: nlist in [1, count] (0 selects the auto
                    // default, also valid).
                    IndexParams p;
                    p.nlist = *rc::gen::inRange<std::uint32_t>(
                        0, static_cast<std::uint32_t>(count) + 1);
                    RC_ASSERT(coll->build_index(IndexType::IVF, p).is_ok());
                    RC_ASSERT(coll->index_type() == IndexType::IVF);
                    break;
                }
                case 2: {
                    // Out-of-range HNSW m (< 2).
                    IndexParams p;
                    p.m               = *rc::gen::element<std::uint32_t>(0u, 1u);
                    p.ef_construction = 200;
                    const auto status = coll->build_index(IndexType::HNSW, p);
                    RC_ASSERT(status.is_error());
                    RC_ASSERT(status.category() == ErrorCategory::InvalidParameter);
                    RC_ASSERT(coll->index_type() == IndexType::HNSW);
                    RC_ASSERT(coll->query(vectors[0], 1, QueryParams{}).is_ok());
                    break;
                }
                case 3: {
                    // Out-of-range HNSW ef_construction (< 1).
                    IndexParams p;
                    p.m               = 16;
                    p.ef_construction = 0;
                    const auto status = coll->build_index(IndexType::HNSW, p);
                    RC_ASSERT(status.is_error());
                    RC_ASSERT(status.category() == ErrorCategory::InvalidParameter);
                    RC_ASSERT(coll->index_type() == IndexType::HNSW);
                    RC_ASSERT(coll->query(vectors[0], 1, QueryParams{}).is_ok());
                    break;
                }
                default: {
                    // Out-of-range IVF nlist (> record count).
                    IndexParams p;
                    p.nlist = static_cast<std::uint32_t>(count) +
                              *rc::gen::inRange<std::uint32_t>(1, 100);
                    const auto status = coll->build_index(IndexType::IVF, p);
                    RC_ASSERT(status.is_error());
                    RC_ASSERT(status.category() == ErrorCategory::InvalidParameter);
                    RC_ASSERT(coll->index_type() == IndexType::HNSW);
                    RC_ASSERT(coll->query(vectors[0], 1, QueryParams{}).is_ok());
                    break;
                }
            }
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 18: Newly inserted records are searchable
// Validates: Requirements 5.1, 5.2
// A query with a record's own vector (Euclidean self-distance 0) under a
// covering k returns that record's id; generous ef_search/nprobe keep it
// deterministic across seeds.
TEST_CASE("Property 18: newly inserted records are searchable",
          "[index][property]") {
    const bool ok = rc::check(
        "after inserting records into an indexed collection, a query with a "
        "record's own vector and sufficient k returns that record's id",
        [] {
            const std::uint32_t dim   = *rc::gen::inRange<std::uint32_t>(2, 9);
            const std::size_t   n     = *rc::gen::inRange<std::size_t>(1, 13);
            const bool          hnsw  = *rc::gen::arbitrary<bool>();

            Engine engine;
            auto created = engine.create_collection("search", dim,
                                                    DistanceMetric::Euclidean);
            RC_ASSERT(created.is_ok());
            CollectionHandle coll = created.value();

            // Build the index over the still-empty collection.
            IndexParams build;
            build.m               = 16;
            build.ef_construction = 200;
            build.nlist           = 0;  // IVF auto-default
            const IndexType type = hnsw ? IndexType::HNSW : IndexType::IVF;
            RC_ASSERT(coll->build_index(type, build).is_ok());
            RC_ASSERT(coll->index_type() == type);

            // Insert records; each is incrementally added to the index.
            std::vector<std::vector<float>> vectors;
            vectors.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                std::vector<float> v = gen_vector(dim);
                RC_ASSERT(coll->insert(make_id(i), v, std::nullopt).is_ok());
                vectors.push_back(std::move(v));
            }
            RC_ASSERT(coll->count() == n);

            // Generous query breadth so the index path explores widely; a
            // covering k guarantees every live record is returnable.
            QueryParams qp;
            qp.ef_search = static_cast<std::uint32_t>(2 * n + 64);
            qp.nprobe    = static_cast<std::uint32_t>(n + 64);

            const std::size_t target = *rc::gen::inRange<std::size_t>(0, n);
            auto result = coll->query(vectors[target],
                                      static_cast<std::uint32_t>(n), qp);
            RC_ASSERT(result.is_ok());

            const auto& neighbors = result.value();
            const bool found = std::any_of(
                neighbors.begin(), neighbors.end(),
                [&](const Neighbor& nb) { return nb.id == make_id(target); });
            RC_ASSERT(found);
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 12: Deletion removes the record and excludes it from queries
// Validates: Requirements 3.4, 3.5, 5.3
// For indexed and non-indexed collections, a deleted record must be absent
// everywhere: count drops by one, get returns NotFound, and a covering query
// (using the deleted record's own vector) never returns the deleted id.
TEST_CASE("Property 12: deletion removes the record and excludes it from queries",
          "[index][records][property]") {
    const bool ok = rc::check(
        "deleting a record decrements the count, makes get return NotFound, and "
        "excludes the record from later queries (indexed or not)",
        [] {
            const std::uint32_t dim   = *rc::gen::inRange<std::uint32_t>(2, 9);
            const std::size_t   n     = *rc::gen::inRange<std::size_t>(2, 13);

            Engine engine;
            auto created = engine.create_collection("del", dim,
                                                    DistanceMetric::Euclidean);
            RC_ASSERT(created.is_ok());
            CollectionHandle coll = created.value();

            std::vector<std::vector<float>> vectors;
            vectors.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                std::vector<float> v = gen_vector(dim);
                RC_ASSERT(coll->insert(make_id(i), v, std::nullopt).is_ok());
                vectors.push_back(std::move(v));
            }
            RC_ASSERT(coll->count() == n);

            // Optionally build an index so deletion must update the index too.
            const int mode = *rc::gen::inRange<int>(0, 3);  // 0 none, 1 HNSW, 2 IVF
            if (mode != 0) {
                IndexParams build;
                build.m               = 16;
                build.ef_construction = 200;
                build.nlist           = 0;
                const IndexType type =
                    (mode == 1) ? IndexType::HNSW : IndexType::IVF;
                RC_ASSERT(coll->build_index(type, build).is_ok());
            }

            const std::size_t target = *rc::gen::inRange<std::size_t>(0, n);
            const std::string target_id = make_id(target);

            RC_ASSERT(coll->remove(target_id).is_ok());

            // Count decremented by exactly one.
            RC_ASSERT(coll->count() == n - 1);

            // The deleted record is no longer retrievable.
            auto got = coll->get(target_id);
            RC_ASSERT(got.is_error());
            RC_ASSERT(got.category() == ErrorCategory::NotFound);

            // Querying with the deleted record's own vector makes the exclusion
            // meaningful: it would otherwise be the nearest match.
            QueryParams qp;
            qp.ef_search = static_cast<std::uint32_t>(2 * n + 64);
            qp.nprobe    = static_cast<std::uint32_t>(n + 64);
            auto result = coll->query(vectors[target],
                                      static_cast<std::uint32_t>(n), qp);
            RC_ASSERT(result.is_ok());

            const auto& neighbors = result.value();
            RC_ASSERT(neighbors.size() == n - 1);
            const bool present = std::any_of(
                neighbors.begin(), neighbors.end(),
                [&](const Neighbor& nb) { return nb.id == target_id; });
            RC_ASSERT(!present);
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 19: Query results are correctly ordered and distance-accurate
// Validates: Requirements 6.1, 6.2, 6.3, 6.6
// A query returns min(k, count) results ordered by ascending distance with
// ascending-id tie-break, distances matching an independent reference within
// tolerance, and the id sequence equal to the true top-k.
TEST_CASE("Property 19: query results are correctly ordered and distance-accurate",
          "[query][property]") {
    const bool ok = rc::check(
        "brute-force query results are ordered by ascending distance with "
        "ascending-id tie-break, report accurate distances, and equal the true "
        "top-k",
        [] {
            const DistanceMetric metric = gen_metric();
            const std::uint32_t  dim    = *rc::gen::inRange<std::uint32_t>(1, 17);
            const std::size_t    count  = *rc::gen::inRange<std::size_t>(1, 33);
            const std::uint32_t  k =
                *rc::gen::inRange<std::uint32_t>(1, static_cast<std::uint32_t>(count) + 6);

            Engine engine;
            auto created = engine.create_collection("q", dim, metric);
            RC_ASSERT(created.is_ok());
            CollectionHandle coll = created.value();  // no index -> brute force

            // For cosine, force non-zero norms so every distance is defined.
            std::map<std::string, std::vector<float>> by_id;
            for (std::size_t i = 0; i < count; ++i) {
                std::vector<float> v = gen_vector(dim);
                if (metric == DistanceMetric::Cosine) {
                    ensure_nonzero_norm(v);
                }
                const std::string id = make_id(i);
                RC_ASSERT(coll->insert(id, v, std::nullopt).is_ok());
                by_id.emplace(id, std::move(v));
            }
            std::vector<float> q = gen_vector(dim);
            if (metric == DistanceMetric::Cosine) {
                ensure_nonzero_norm(q);
            }

            auto result = coll->query(q, k, QueryParams{});
            RC_ASSERT(result.is_ok());
            const auto& neighbors = result.value();

            // At most min(k, count) results; exactly that many here since the
            // collection is non-empty.
            const std::size_t expected =
                std::min<std::size_t>(k, count);
            RC_ASSERT(neighbors.size() == expected);

            // Independent reference: distance per record, sorted by ascending
            // float32 distance with ascending-id tie-break.
            std::vector<std::pair<float, std::string>> reference;
            reference.reserve(count);
            for (const auto& [id, v] : by_id) {
                reference.emplace_back(reference_distance(metric, q, v), id);
            }
            std::sort(reference.begin(), reference.end(),
                      [](const auto& a, const auto& b) {
                          if (a.first != b.first) return a.first < b.first;
                          return a.second < b.second;
                      });

            // The returned id sequence equals the true top-k, and each reported
            // distance matches the scalar reference within tolerance.
            for (std::size_t i = 0; i < neighbors.size(); ++i) {
                RC_ASSERT(neighbors[i].id == reference[i].second);
                const double ref_d = reference_distance(
                    metric, q, by_id.at(neighbors[i].id));
                RC_ASSERT(within_tolerance(neighbors[i].distance, ref_d));
            }

            // The engine output is itself ordered by ascending distance with
            // ties broken by ascending id.
            for (std::size_t i = 1; i < neighbors.size(); ++i) {
                const Neighbor& prev = neighbors[i - 1];
                const Neighbor& cur  = neighbors[i];
                if (prev.distance == cur.distance) {
                    RC_ASSERT(prev.id < cur.id);
                } else {
                    RC_ASSERT(prev.distance < cur.distance);
                }
            }
        });
    REQUIRE(ok);
}

// Unit tests: query edge cases.

// A query whose vector length differs from the dimensionality is rejected with
// DimensionMismatch.
TEST_CASE("query with a dimension-mismatched vector returns DimensionMismatch",
          "[query][unit]") {
    Engine engine;
    auto created = engine.create_collection("c", 4, DistanceMetric::Euclidean);
    REQUIRE(created.is_ok());
    CollectionHandle coll = created.value();

    const std::vector<float> stored = {1.0f, 2.0f, 3.0f, 4.0f};
    REQUIRE(coll->insert("a", stored, std::nullopt).is_ok());

    // Too short and too long both mismatch (k >= 1 so the k check passes first).
    const std::vector<float> too_short = {1.0f, 2.0f, 3.0f};
    const std::vector<float> too_long  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto short_result = coll->query(too_short, 1, QueryParams{});
    REQUIRE(short_result.is_error());
    REQUIRE(short_result.category() == ErrorCategory::DimensionMismatch);

    auto long_result = coll->query(too_long, 1, QueryParams{});
    REQUIRE(long_result.is_error());
    REQUIRE(long_result.category() == ErrorCategory::DimensionMismatch);
}

// A query vector containing a non-finite component is rejected with InvalidValue.
TEST_CASE("query with a non-finite component returns InvalidValue",
          "[query][unit]") {
    Engine engine;
    auto created = engine.create_collection("c", 3, DistanceMetric::Euclidean);
    REQUIRE(created.is_ok());
    CollectionHandle coll = created.value();

    const std::vector<float> stored = {1.0f, 2.0f, 3.0f};
    REQUIRE(coll->insert("a", stored, std::nullopt).is_ok());

    const float kInf = std::numeric_limits<float>::infinity();
    const float kNan = std::numeric_limits<float>::quiet_NaN();

    const std::vector<float> nan_query = {kNan, 0.0f, 0.0f};
    auto nan_result = coll->query(nan_query, 1, QueryParams{});
    REQUIRE(nan_result.is_error());
    REQUIRE(nan_result.category() == ErrorCategory::InvalidValue);

    const std::vector<float> inf_query = {0.0f, kInf, 0.0f};
    auto inf_result = coll->query(inf_query, 1, QueryParams{});
    REQUIRE(inf_result.is_error());
    REQUIRE(inf_result.category() == ErrorCategory::InvalidValue);
}

// A query against a collection containing zero records returns an empty result set.
TEST_CASE("query against an empty collection returns an empty result set",
          "[query][unit]") {
    Engine engine;
    auto created = engine.create_collection("c", 5, DistanceMetric::Cosine);
    REQUIRE(created.is_ok());
    CollectionHandle coll = created.value();
    REQUIRE(coll->count() == 0);

    const std::vector<float> q = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto result = coll->query(q, 3, QueryParams{});
    REQUIRE(result.is_ok());
    REQUIRE(result.value().empty());
}

// A query with k = 0 is rejected with InvalidArgument, and the k check precedes
// the dimensionality and finite-value checks.
TEST_CASE("query with k = 0 returns InvalidArgument",
          "[query][unit]") {
    Engine engine;
    auto created = engine.create_collection("c", 3, DistanceMetric::Euclidean);
    REQUIRE(created.is_ok());
    CollectionHandle coll = created.value();

    const std::vector<float> stored = {1.0f, 2.0f, 3.0f};
    REQUIRE(coll->insert("a", stored, std::nullopt).is_ok());

    const std::vector<float> q = {1.0f, 2.0f, 3.0f};
    auto result = coll->query(q, 0, QueryParams{});
    REQUIRE(result.is_error());
    REQUIRE(result.category() == ErrorCategory::InvalidArgument);
}
