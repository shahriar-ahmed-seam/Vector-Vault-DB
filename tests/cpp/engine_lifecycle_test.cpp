// Engine / Collection lifecycle tests: create, get, list, and delete, including
// the test-only deletion fault-injection hook.

#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>

#include "vectorvault/engine.hpp"
#include "vectorvault/error.hpp"
#include "vectorvault/types.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <vector>

using namespace vectorvault;

namespace {

// A valid collection name: 1 to 255 printable ASCII characters, which keeps
// shrunk counterexamples readable.
std::string gen_valid_name() {
    const std::size_t len = *rc::gen::inRange<std::size_t>(1, 256);  // [1, 255]
    return *rc::gen::container<std::string>(
        len, rc::gen::inRange<char>(static_cast<char>(33),
                                    static_cast<char>(127)));  // '!'..'~'
}

// A valid dimensionality in [1, 65536] inclusive.
std::uint32_t gen_valid_dim() {
    return *rc::gen::inRange<std::uint32_t>(1, 65537);  // [1, 65536]
}

// One of the three supported metrics.
DistanceMetric gen_valid_metric() {
    return *rc::gen::element(DistanceMetric::Euclidean,
                             DistanceMetric::Cosine,
                             DistanceMetric::DotProduct);
}

}  // namespace

// Feature: vector-vault-db, Property 1: Collection creation reflects inputs
// Validates: Requirements 1.1
TEST_CASE("Property 1: collection creation reflects inputs",
          "[engine][lifecycle][property]") {
    const bool ok = rc::check(
        "a created collection has the supplied name, dimensionality, and "
        "metric, and an initial record count of 0",
        [] {
            const std::string    name   = gen_valid_name();
            const std::uint32_t  dim    = gen_valid_dim();
            const DistanceMetric metric = gen_valid_metric();

            Engine engine;
            auto result = engine.create_collection(name, dim, metric);
            RC_ASSERT(result.is_ok());

            const CollectionHandle handle = result.value();
            RC_ASSERT(handle.valid());

            RC_ASSERT(handle->config().name == name);
            RC_ASSERT(handle->config().dimensionality == dim);
            RC_ASSERT(handle->config().metric == metric);
            RC_ASSERT(handle->count() == 0);

            const auto infos = engine.list_collections();
            RC_ASSERT(infos.size() == 1);
            RC_ASSERT(infos.front().name == name);
            RC_ASSERT(infos.front().dimensionality == dim);
            RC_ASSERT(infos.front().metric == metric);
            RC_ASSERT(infos.front().record_count == 0);
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 2: Duplicate collection names are rejected without side effects
// Validates: Requirements 1.2
TEST_CASE("Property 2: duplicate collection names are rejected without side effects",
          "[engine][lifecycle][property]") {
    const bool ok = rc::check(
        "a second create with an existing name returns NameConflict and leaves "
        "the registry unchanged",
        [] {
            const std::string    name   = gen_valid_name();
            const std::uint32_t  dim    = gen_valid_dim();
            const DistanceMetric metric = gen_valid_metric();

            Engine engine;
            auto first = engine.create_collection(name, dim, metric);
            RC_ASSERT(first.is_ok());

            // Snapshot the registry before the duplicate attempt.
            const auto before = engine.list_collections();
            RC_ASSERT(before.size() == 1);

            // A second create with the same name may differ in dimensionality
            // and metric; it must still be rejected purely on the name clash.
            const std::uint32_t  dim2    = gen_valid_dim();
            const DistanceMetric metric2 = gen_valid_metric();
            auto second = engine.create_collection(name, dim2, metric2);
            RC_ASSERT(second.is_error());
            RC_ASSERT(second.category() == ErrorCategory::NameConflict);

            // No collection was created and nothing changed: the listing is
            // identical and the original config is intact.
            const auto after = engine.list_collections();
            RC_ASSERT(after.size() == before.size());
            RC_ASSERT(after.front().name == name);
            RC_ASSERT(after.front().dimensionality == dim);
            RC_ASSERT(after.front().metric == metric);
            RC_ASSERT(after.front().record_count == 0);

            auto got = engine.get_collection(name);
            RC_ASSERT(got.is_ok());
            RC_ASSERT(got.value()->config().dimensionality == dim);
            RC_ASSERT(got.value()->config().metric == metric);
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 3: Invalid collection configuration is rejected
// Validates: Requirements 1.3, 1.4
// Each iteration exercises one invalid axis: a dimensionality outside [1, 65536]
// or a name of length 0 or > 255. In both cases no collection is created.
TEST_CASE("Property 3: invalid collection configuration is rejected",
          "[engine][lifecycle][property]") {
    const bool ok = rc::check(
        "out-of-range dimensionality yields InvalidDimensionality and "
        "out-of-range name yields InvalidName; no collection is created",
        [] {
            Engine engine;

            const bool test_dimensionality = *rc::gen::arbitrary<bool>();

            if (test_dimensionality) {
                // Out-of-range dimensionality (0 or > 65536) with a valid name.
                const std::string name = gen_valid_name();
                const bool use_zero    = *rc::gen::arbitrary<bool>();
                const std::uint32_t dim =
                    use_zero ? 0u
                             : *rc::gen::inRange<std::uint32_t>(
                                   65537u, std::numeric_limits<std::uint32_t>::max());

                auto result = engine.create_collection(
                    name, dim, DistanceMetric::Euclidean);
                RC_ASSERT(result.is_error());
                RC_ASSERT(result.category() ==
                          ErrorCategory::InvalidDimensionality);
            } else {
                // Out-of-range name (empty, or length > 255) with a valid dim.
                const bool use_empty = *rc::gen::arbitrary<bool>();
                std::string name;
                if (!use_empty) {
                    const std::size_t len =
                        *rc::gen::inRange<std::size_t>(256, 1025);  // > 255
                    name = *rc::gen::container<std::string>(
                        len, rc::gen::inRange<char>(static_cast<char>(33),
                                                    static_cast<char>(127)));
                }

                auto result = engine.create_collection(
                    name, gen_valid_dim(), DistanceMetric::Euclidean);
                RC_ASSERT(result.is_error());
                RC_ASSERT(result.category() == ErrorCategory::InvalidName);
            }

            // In every rejection case no collection is created.
            RC_ASSERT(engine.list_collections().empty());
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 4: Collection listing matches created collections
// Validates: Requirements 1.8
TEST_CASE("Property 4: collection listing matches created collections",
          "[engine][lifecycle][property]") {
    const bool ok = rc::check(
        "listing returns exactly one matching info entry per successfully "
        "created collection",
        [] {
            Engine engine;

            // Expected collections keyed by name. Duplicate generated names
            // collide on creation and are deduplicated by the map, so the model
            // mirrors the engine's registry.
            std::map<std::string, CollectionInfo> expected;

            const std::size_t requested = *rc::gen::inRange<std::size_t>(0, 16);
            for (std::size_t i = 0; i < requested; ++i) {
                const std::string    name   = gen_valid_name();
                const std::uint32_t  dim    = gen_valid_dim();
                const DistanceMetric metric = gen_valid_metric();

                auto result = engine.create_collection(name, dim, metric);
                if (result.is_ok()) {
                    RC_ASSERT(expected.find(name) == expected.end());
                    expected[name] = CollectionInfo{name, dim, metric, 0};
                } else {
                    // The only valid rejection is a duplicate name.
                    RC_ASSERT(result.category() == ErrorCategory::NameConflict);
                    RC_ASSERT(expected.find(name) != expected.end());
                }
            }

            const auto infos = engine.list_collections();

            // Exactly one entry per created collection, matching its config.
            RC_ASSERT(infos.size() == expected.size());

            for (const auto& info : infos) {
                const auto it = expected.find(info.name);
                RC_ASSERT(it != expected.end());
                RC_ASSERT(info.dimensionality == it->second.dimensionality);
                RC_ASSERT(info.metric == it->second.metric);
                RC_ASSERT(info.record_count == 0);
            }
        });
    REQUIRE(ok);
}

// Unit tests: lifecycle edge cases.

// A distance metric outside the three supported values is rejected with
// InvalidMetric and no collection is created.
TEST_CASE("create_collection rejects an invalid distance metric",
          "[engine][lifecycle][unit]") {
    Engine engine;

    const auto invalid_metric = static_cast<DistanceMetric>(99);
    auto result = engine.create_collection("valid-name", 128, invalid_metric);

    REQUIRE(result.is_error());
    REQUIRE(result.category() == ErrorCategory::InvalidMetric);
    REQUIRE(engine.list_collections().empty());
}

// When no collection exists, the listing is empty.
TEST_CASE("list_collections returns an empty list when no collection exists",
          "[engine][lifecycle][unit]") {
    Engine engine;
    REQUIRE(engine.list_collections().empty());
}

// When deletion fails before completion, the collection is left retrievable in
// its pre-deletion state and a DeletionFailed error is returned. The test-only
// fault-injection hook deterministically forces the failure path.
TEST_CASE("failed deletion leaves the collection retrievable",
          "[engine][lifecycle][unit]") {
    Engine engine;

    auto created =
        engine.create_collection("keep-me", 64, DistanceMetric::Euclidean);
    REQUIRE(created.is_ok());

    // Force every deletion to fail before completion.
    engine.set_delete_fault_injector(
        [](std::string_view) { return true; });

    auto deleted = engine.delete_collection("keep-me");
    REQUIRE(deleted.is_error());
    REQUIRE(deleted.category() == ErrorCategory::DeletionFailed);

    // The collection is still retrievable in its pre-deletion state.
    auto got = engine.get_collection("keep-me");
    REQUIRE(got.is_ok());
    REQUIRE(got.value()->config().name == "keep-me");
    REQUIRE(got.value()->config().dimensionality == 64);
    REQUIRE(got.value()->config().metric == DistanceMetric::Euclidean);
    REQUIRE(got.value()->count() == 0);

    const auto infos = engine.list_collections();
    REQUIRE(infos.size() == 1);
    REQUIRE(infos.front().name == "keep-me");

    // Clearing the injector lets a subsequent deletion succeed.
    engine.set_delete_fault_injector(nullptr);
    REQUIRE(engine.delete_collection("keep-me").is_ok());
    REQUIRE(engine.list_collections().empty());
}
