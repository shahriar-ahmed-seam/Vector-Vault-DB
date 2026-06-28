// Recall benchmark and persistence/dispatch integration tests: a deterministic
// recall measurement plus full-pipeline round-trip and dispatch-correctness
// checks, tagged [integration] so ctest runs them.

#include <catch2/catch_test_macros.hpp>

#include "vectorvault/distance.hpp"
#include "vectorvault/engine.hpp"
#include "vectorvault/error.hpp"
#include "vectorvault/index.hpp"
#include "vectorvault/types.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
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
              ("vvault_int_" + std::to_string(stamp) + "_" +
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

// Zero-padded id so lexical and numeric order agree, keeping distance-tie
// ordering predictable.
std::string make_id(std::size_t i) {
    std::string s = std::to_string(i);
    if (s.size() < 6) {
        s = std::string(6 - s.size(), '0') + s;
    }
    return "rec-" + s;
}

}  // namespace

// Feature: vector-vault-db, recall benchmark (task 13.2)
// Validates: Requirements 5.7
// Fixed-seed measurement: build a >=10000-record Euclidean collection with the
// default HNSW parameters and query with default QueryParams, comparing each
// approximate top-10 against the exact brute-force top-10 (mean recall@10 >= 0.90).
TEST_CASE("recall benchmark: HNSW mean recall@10 >= 0.90 on >=10000 records",
          "[recall][integration]") {
    constexpr std::uint32_t kDim        = 32;     // modest embedding dimensionality
    constexpr std::size_t   kRecords    = 10000;  // >= 10000
    constexpr std::size_t   kQueries    = 1000;   // >= 1000
    constexpr std::uint32_t kK          = 10;
    constexpr std::uint64_t kDataSeed   = 0xC0FFEEULL;
    constexpr std::uint64_t kQuerySeed  = 0x1234567ULL;

    // Components drawn uniformly from [-1, 1] with a fixed seed so the recall
    // figure is reproducible across runs and hosts.
    std::mt19937_64 data_rng(kDataSeed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> vectors;
    vectors.reserve(kRecords);
    std::vector<RecordInput> batch;
    batch.reserve(kRecords);
    for (std::size_t i = 0; i < kRecords; ++i) {
        std::vector<float> v(kDim);
        for (std::uint32_t c = 0; c < kDim; ++c) {
            v[c] = dist(data_rng);
        }
        RecordInput in;
        in.id     = make_id(i);
        in.vector = v;
        batch.push_back(std::move(in));
        vectors.push_back(std::move(v));
    }

    Engine engine;
    auto created = engine.create_collection("recall_bench", kDim,
                                            DistanceMetric::Euclidean);
    REQUIRE(created.is_ok());
    CollectionHandle coll = created.value();

    // insert_batch accepts 1..10000 records, so the corpus is one batch.
    REQUIRE(coll->insert_batch(batch).is_ok());
    REQUIRE(coll->count() == kRecords);

    // Build the HNSW index with the documented default parameters
    // (m=16, ef_construction=200).
    IndexParams params;  // m=16, ef_construction=200 by default
    REQUIRE(params.m == 16);
    REQUIRE(params.ef_construction == 200);
    REQUIRE(coll->build_index(IndexType::HNSW, params).is_ok());
    REQUIRE(coll->index_type() == IndexType::HNSW);

    // Exact brute-force top-k baseline: the true nearest kK ids for a query,
    // scanned with the scalar L2 reference kernel (independent of the index).
    const auto exact_topk = [&](const std::vector<float>& q) {
        std::vector<std::pair<float, std::size_t>> scored(kRecords);
        for (std::size_t i = 0; i < kRecords; ++i) {
            const float d2 = kernels::scalar_l2_squared(q.data(), vectors[i].data(), kDim);
            scored[i] = {d2, i};
        }
        std::partial_sort(
            scored.begin(), scored.begin() + kK, scored.end(),
            [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;  // tie-break by ascending index/id
            });
        std::unordered_set<std::string> ids;
        ids.reserve(kK * 2);
        for (std::uint32_t r = 0; r < kK; ++r) {
            ids.insert(make_id(scored[r].second));
        }
        return ids;
    };

    std::mt19937_64 query_rng(kQuerySeed);
    const QueryParams qp;  // ef_search=50 by default
    REQUIRE(qp.ef_search == 50);

    double recall_sum = 0.0;
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        std::vector<float> q(kDim);
        for (std::uint32_t c = 0; c < kDim; ++c) {
            q[c] = dist(query_rng);
        }

        auto result = coll->query(q, kK, qp);
        REQUIRE(result.is_ok());
        const std::vector<Neighbor>& neighbors = result.value();
        REQUIRE(neighbors.size() == kK);

        const std::unordered_set<std::string> truth = exact_topk(q);
        std::size_t hits = 0;
        for (const Neighbor& n : neighbors) {
            if (truth.count(n.id) != 0) {
                ++hits;
            }
        }
        recall_sum += static_cast<double>(hits) / static_cast<double>(kK);
    }

    const double mean_recall = recall_sum / static_cast<double>(kQueries);

    // Report the measured figure so it is visible in the test log.
    std::cout << "[recall benchmark] HNSW(m=16, ef_construction=200, ef_search=50) "
              << "mean recall@10 over " << kQueries << " queries on " << kRecords
              << " records (dim " << kDim << ") = " << mean_recall << std::endl;

    REQUIRE(mean_recall >= 0.90);
}

// Feature: vector-vault-db, end-to-end persistence equivalence (task 13.3)
// Validates: Requirements 8.2, 8.4
// Save a populated, indexed collection and load it into a fresh engine
// (mmap-backed), asserting equivalence of config, count, index type, per-record
// vector+metadata, and pre-save vs post-load query results, for HNSW and IVF.
TEST_CASE("end-to-end persistence: save/load reconstructs an equivalent indexed collection",
          "[persistence][integration]") {
    constexpr std::uint32_t kDim     = 16;
    constexpr std::size_t   kRecords = 300;

    for (const IndexType index_type : {IndexType::HNSW, IndexType::IVF}) {
        TempDir tmp;
        Engine engine;

        const std::string name =
            std::string("persist_") + (index_type == IndexType::HNSW ? "hnsw" : "ivf");
        auto created =
            engine.create_collection(name, kDim, DistanceMetric::Euclidean);
        REQUIRE(created.is_ok());
        CollectionHandle coll = created.value();

        // Attach metadata to some records and leave others absent so both the
        // metadata-preservation and explicit "no metadata" cases round-trip.
        std::mt19937_64 rng(0xABCDEFULL + static_cast<std::uint64_t>(index_type));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<std::vector<float>> vectors(kRecords);
        for (std::size_t i = 0; i < kRecords; ++i) {
            std::vector<float> v(kDim);
            for (std::uint32_t c = 0; c < kDim; ++c) {
                v[c] = dist(rng);
            }
            std::optional<Metadata> meta;
            if (i % 3 == 0) {
                Metadata m;
                m.emplace("idx", MetadataValue{static_cast<std::int64_t>(i)});
                m.emplace("tag", MetadataValue{std::string("v") + std::to_string(i)});
                m.emplace("flag", MetadataValue{(i % 2) == 0});
                meta = std::move(m);
            }
            REQUIRE(coll->insert(make_id(i), v, meta).is_ok());
            vectors[i] = std::move(v);
        }
        REQUIRE(coll->count() == kRecords);

        IndexParams params;
        params.nlist = 0;  // IVF auto-default; ignored by HNSW
        REQUIRE(coll->build_index(index_type, params).is_ok());

        // Capture pre-save query results for a handful of probes.
        const std::uint32_t kK = 10;
        const QueryParams qp;
        std::vector<std::vector<Neighbor>> pre_results;
        for (std::size_t qi = 0; qi < 5; ++qi) {
            auto r = coll->query(vectors[qi * 7], kK, qp);
            REQUIRE(r.is_ok());
            pre_results.push_back(r.value());
        }

        // save -> load into a fresh engine.
        const fs::path path = tmp.file("snapshot");
        REQUIRE(engine.save_collection(name, path).is_ok());
        REQUIRE(fs::exists(path));

        Engine reload;
        auto loaded = reload.load_collection(path);
        REQUIRE(loaded.is_ok());
        CollectionHandle reloaded = loaded.value();

        // Config / count / index_type equivalence.
        REQUIRE(reloaded->config().name == name);
        REQUIRE(reloaded->config().dimensionality == kDim);
        REQUIRE(reloaded->config().metric == DistanceMetric::Euclidean);
        REQUIRE(reloaded->count() == kRecords);
        REQUIRE(reloaded->index_type() == index_type);

        // Every record's vector and metadata preserved (read through the mmap).
        for (std::size_t i = 0; i < kRecords; ++i) {
            auto got = reloaded->get(make_id(i));
            REQUIRE(got.is_ok());
            REQUIRE(got.value().vector == vectors[i]);
            if (i % 3 == 0) {
                REQUIRE(got.value().meta.has_value());
                const Metadata& m = got.value().meta.value();
                REQUIRE(m.at("idx") == MetadataValue{static_cast<std::int64_t>(i)});
                REQUIRE(m.at("tag") ==
                        MetadataValue{std::string("v") + std::to_string(i)});
                REQUIRE(m.at("flag") == MetadataValue{(i % 2) == 0});
            } else {
                REQUIRE_FALSE(got.value().meta.has_value());
            }
        }

        // Queries return equivalent results pre-save vs post-load.
        for (std::size_t qi = 0; qi < pre_results.size(); ++qi) {
            auto r = reloaded->query(vectors[qi * 7], kK, qp);
            REQUIRE(r.is_ok());
            const std::vector<Neighbor>& post = r.value();
            const std::vector<Neighbor>& pre  = pre_results[qi];
            REQUIRE(post.size() == pre.size());
            for (std::size_t j = 0; j < pre.size(); ++j) {
                REQUIRE(post[j].id == pre[j].id);
                REQUIRE(post[j].distance == pre[j].distance);
            }
        }
    }
}

// Feature: vector-vault-db, dispatch correctness on the full pipeline (task 13.3)
// Validates: Requirements 4.6, 4.7
// The active path (SIMD or scalar fallback) is asserted equal to the independent
// scalar reference kernels over many random vectors for every metric, with a
// stable uses_simd().
TEST_CASE("dispatch correctness: active path matches the scalar reference kernels",
          "[dispatch][integration]") {
    const DistanceCalculator calc = DistanceCalculator::create();

    // uses_simd() is stable across repeated create() calls.
    const bool simd = calc.uses_simd();
    for (int i = 0; i < 4; ++i) {
        REQUIRE(DistanceCalculator::create().uses_simd() == simd);
    }
    std::cout << "[dispatch] active distance path: "
              << (simd ? "AVX-512 SIMD" : "scalar fallback") << std::endl;

    // Reference distance built only from the exposed scalar reference kernels,
    // independent of the calculator.
    const auto reference = [](DistanceMetric metric, const std::vector<float>& a,
                              const std::vector<float>& b) -> float {
        const std::size_t n = a.size();
        switch (metric) {
            case DistanceMetric::Euclidean:
                return std::sqrt(kernels::scalar_l2_squared(a.data(), b.data(), n));
            case DistanceMetric::DotProduct:
                return kernels::scalar_dot(a.data(), b.data(), n);
            case DistanceMetric::Cosine: {
                const float dot   = kernels::scalar_dot(a.data(), b.data(), n);
                const float na    = std::sqrt(kernels::scalar_dot(a.data(), a.data(), n));
                const float nb    = std::sqrt(kernels::scalar_dot(b.data(), b.data(), n));
                float cd = 1.0f - dot / (na * nb);
                if (cd < 0.0f) cd = 0.0f;
                if (cd > 2.0f) cd = 2.0f;
                return cd;
            }
        }
        return 0.0f;
    };

    // Tolerance: 1e-4 relative with a 1e-4 absolute fallback. Cosine uses a pure
    // absolute tolerance because its 1 - dot/(norms) cancellation amplifies
    // float32-vs-double differences into a large relative error near zero.
    const auto within_tol = [](DistanceMetric metric, float actual, float ref) {
        constexpr float kTol = 1e-4f;
        const float diff = std::fabs(actual - ref);
        if (metric == DistanceMetric::Cosine) {
            return diff <= kTol;  // bounded [0,2]: absolute tolerance
        }
        const float mag = std::fabs(ref);
        return (mag < kTol) ? (diff <= kTol) : (diff / mag <= kTol);
    };

    // Deterministic random vectors with non-negative components and modest
    // dimensionality, so reductions are cancellation-free and float32/double
    // agree; the 1e-4 comparison then isolates dispatch correctness from
    // accumulation differences (which Property 13 covers against a double oracle).
    std::mt19937_64 rng(0xD15DA7C4ULL);
    std::uniform_int_distribution<int> dim_dist(1, 128);
    std::uniform_real_distribution<float> comp(0.0f, 1.0f);

    constexpr int kTrials = 3000;
    for (int t = 0; t < kTrials; ++t) {
        const std::size_t n = static_cast<std::size_t>(dim_dist(rng));
        std::vector<float> a(n), b(n);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = comp(rng);
            b[i] = comp(rng);
        }

        for (const DistanceMetric metric : {DistanceMetric::Euclidean,
                                            DistanceMetric::DotProduct,
                                            DistanceMetric::Cosine}) {
            if (metric == DistanceMetric::Cosine) {
                // Guarantee non-zero norms so cosine is defined.
                if (kernels::scalar_dot(a.data(), a.data(), n) == 0.0f) a[0] = 1.0f;
                if (kernels::scalar_dot(b.data(), b.data(), n) == 0.0f) b[0] = 1.0f;
            }
            const auto result = calc.distance(metric, a, b);
            REQUIRE(result.is_ok());
            REQUIRE(within_tol(metric, result.value(), reference(metric, a, b)));
            if (metric == DistanceMetric::Cosine) {
                REQUIRE(result.value() >= 0.0f);
                REQUIRE(result.value() <= 2.0f);
            }
        }
    }
}
