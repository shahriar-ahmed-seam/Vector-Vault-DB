// End-to-end pipeline smoke test: drives the engine stack through create ->
// insert_batch + insert -> build_index -> query -> save -> load -> query, and
// asserts the reloaded query result matches the pre-save result. Small
// integer-valued components keep float32 reductions exact and the comparison
// deterministic.

#include <catch2/catch_test_macros.hpp>

#include "vectorvault/engine.hpp"
#include "vectorvault/error.hpp"
#include "vectorvault/index.hpp"
#include "vectorvault/types.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

using namespace vectorvault;

namespace {

namespace fs = std::filesystem;

// A unique per-process temp directory, removed when the test case ends.
struct TempDir {
    fs::path dir;

    TempDir() {
        static std::atomic<std::uint64_t> counter{0};
        const auto stamp = static_cast<unsigned long long>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        const auto n = counter.fetch_add(1, std::memory_order_relaxed);
        dir = fs::temp_directory_path() /
              ("vvault_e2e_" + std::to_string(stamp) + "_" +
               std::to_string(static_cast<unsigned long long>(n)));
        std::error_code ec;
        fs::create_directories(dir, ec);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    fs::path file(const char* stem) const { return dir / (std::string(stem) + ".vv"); }
};

constexpr std::uint32_t kDim = 12;

// A deterministic integer-valued vector for record `i` (kept small so float32
// distance reductions are exact).
std::vector<float> make_vector(std::size_t i) {
    std::vector<float> v(kDim);
    for (std::uint32_t c = 0; c < kDim; ++c) {
        v[c] = static_cast<float>((i * 7 + c * 3) % 17) - 8.0f;
    }
    return v;
}

std::string make_id(std::size_t i) { return "rec-" + std::to_string(i); }

}  // namespace

// Feature: vector-vault-db, end-to-end pipeline (task 13.1)
// Validates: Requirements 9.1
TEST_CASE("end-to-end: create -> batch+single insert -> build -> query -> save -> load -> query",
          "[e2e][integration]") {
    // Run once per index type so both HNSW and IVF traversal are wired end to
    // end through the same lifecycle.
    for (const IndexType index_type : {IndexType::HNSW, IndexType::IVF}) {
        TempDir tmp;
        Engine engine;

        const std::string name =
            std::string("e2e_") + (index_type == IndexType::HNSW ? "hnsw" : "ivf");
        auto created =
            engine.create_collection(name, kDim, DistanceMetric::Euclidean);
        REQUIRE(created.is_ok());
        CollectionHandle coll = created.value();
        REQUIRE(coll->count() == 0);

        constexpr std::size_t kBatch = 80;
        std::vector<RecordInput> batch;
        batch.reserve(kBatch);
        for (std::size_t i = 0; i < kBatch; ++i) {
            RecordInput in;
            in.id     = make_id(i);
            in.vector = make_vector(i);
            Metadata meta;
            meta.emplace("idx", MetadataValue{static_cast<std::int64_t>(i)});
            in.meta = meta;
            batch.push_back(std::move(in));
        }
        REQUIRE(coll->insert_batch(batch).is_ok());
        REQUIRE(coll->count() == kBatch);

        constexpr std::size_t kSingles = 20;
        for (std::size_t i = kBatch; i < kBatch + kSingles; ++i) {
            const std::vector<float> v = make_vector(i);
            REQUIRE(coll->insert(make_id(i), v, std::nullopt).is_ok());
        }
        const std::uint64_t total = kBatch + kSingles;
        REQUIRE(coll->count() == total);

        const std::uint64_t before_overwrite = coll->count();
        const std::vector<float> v0 = make_vector(0);
        REQUIRE(coll->insert(make_id(0), v0, std::nullopt).is_ok());
        REQUIRE(coll->count() == before_overwrite);  // overwrite, not append

        IndexParams params;
        params.nlist = 0;  // IVF auto-default; ignored by HNSW
        REQUIRE(coll->build_index(index_type, params).is_ok());
        REQUIRE(coll->index_type() == index_type);

        const std::vector<float> q = make_vector(3);
        const std::uint32_t k = 10;
        QueryParams qp;
        auto pre = coll->query(q, k, qp);
        REQUIRE(pre.is_ok());
        const std::vector<Neighbor>& pre_neighbors = pre.value();
        REQUIRE(pre_neighbors.size() == k);
        // Results are ordered by non-decreasing distance.
        for (std::size_t i = 1; i < pre_neighbors.size(); ++i) {
            REQUIRE(pre_neighbors[i - 1].distance <= pre_neighbors[i].distance);
        }

        const fs::path path = tmp.file("snapshot");
        REQUIRE(engine.save_collection(name, path).is_ok());
        REQUIRE(fs::exists(path));

        // Load into a fresh engine (mmap-backed read path).
        Engine reload;
        auto loaded = reload.load_collection(path);
        REQUIRE(loaded.is_ok());
        CollectionHandle reloaded = loaded.value();
        REQUIRE(reloaded->config().dimensionality == kDim);
        REQUIRE(reloaded->count() == total);
        REQUIRE(reloaded->index_type() == index_type);

        // Loaded query results match the pre-save results exactly.
        auto post = reloaded->query(q, k, qp);
        REQUIRE(post.is_ok());
        const std::vector<Neighbor>& post_neighbors = post.value();
        REQUIRE(post_neighbors.size() == pre_neighbors.size());
        for (std::size_t i = 0; i < pre_neighbors.size(); ++i) {
            REQUIRE(post_neighbors[i].id == pre_neighbors[i].id);
            REQUIRE(post_neighbors[i].distance == pre_neighbors[i].distance);
        }

        // A stored record (with batch-inserted metadata) survives the round trip.
        auto got = reloaded->get(make_id(5));
        REQUIRE(got.is_ok());
        REQUIRE(got.value().vector == make_vector(5));
        REQUIRE(got.value().meta.has_value());
    }
}
