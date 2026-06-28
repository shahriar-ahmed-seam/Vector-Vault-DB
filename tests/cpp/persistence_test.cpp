// Persistence tests: the binary snapshot save/load path through the Engine
// registry, covering round-trip equivalence, byte-identical re-save, content
// corruption detection, and save/load error edge cases. Components and metadata
// doubles are finite and stored as raw bits, so round-trips are bit-exact.

#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>

#include "vectorvault/engine.hpp"
#include "vectorvault/error.hpp"
#include "vectorvault/index.hpp"
#include "vectorvault/snapshot_format.hpp"
#include "vectorvault/types.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace vectorvault;

namespace {

namespace fs = std::filesystem;

// A unique per-process temp subdirectory, removed recursively when destroyed.
struct TempDir {
    fs::path dir;

    TempDir() {
        static std::atomic<std::uint64_t> counter{0};
        const auto pid =
            static_cast<unsigned long long>(
                std::chrono::steady_clock::now().time_since_epoch().count());
        const auto n = counter.fetch_add(1, std::memory_order_relaxed);
        dir = fs::temp_directory_path() /
              ("vvault_persist_" + std::to_string(pid) + "_" +
               std::to_string(static_cast<unsigned long long>(n)));
        std::error_code ec;
        fs::create_directories(dir, ec);
    }

    ~TempDir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    // A unique file path inside this directory (unique across calls so a held
    // memory mapping from a prior iteration cannot collide with a new save).
    fs::path file(const char* stem) const {
        static std::atomic<std::uint64_t> counter{0};
        const auto n = counter.fetch_add(1, std::memory_order_relaxed);
        return dir / (std::string(stem) + "_" +
                      std::to_string(static_cast<unsigned long long>(n)) + ".vv");
    }
};

// Reads an entire file into a byte buffer. Returns an empty buffer if the file
// cannot be opened.
std::vector<unsigned char> read_all(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(in),
                                      std::istreambuf_iterator<char>());
}

// Overwrites a file with the given bytes (binary, truncating).
void write_all(const fs::path& path, const std::vector<unsigned char>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
}

// A randomly generated collection: its configuration, its records (preserving
// the optional-metadata distinction), and which index, if any, to build.
struct GenSpec {
    struct Rec {
        RecordId                id;
        std::vector<float>      vector;
        std::optional<Metadata> meta;
    };

    std::string      name;
    std::uint32_t    dim = 1;
    DistanceMetric   metric = DistanceMetric::Euclidean;
    int              index_mode = 0;  // 0 = none, 1 = HNSW, 2 = IVF
    std::vector<Rec> records;

    IndexType expected_index_type() const {
        switch (index_mode) {
            case 1:  return IndexType::HNSW;
            case 2:  return IndexType::IVF;
            default: return IndexType::None;
        }
    }
};

// Stable, unique record identifiers r0, r1, ... (non-empty).
std::string make_id(std::size_t i) { return "r" + std::to_string(i); }

// A finite float32 component in [-1000, 1000] with 0.001 granularity. The
// integer source keeps values finite, varied, and exactly representable across
// the float32 store/load round-trip.
float gen_component() {
    return static_cast<float>(*rc::gen::inRange<int>(-1000000, 1000001)) /
           1000.0f;
}

std::vector<float> gen_vector(std::uint32_t dim) {
    std::vector<float> v;
    v.reserve(dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
        v.push_back(gen_component());
    }
    return v;
}

// A short lowercase string value for metadata.
std::string gen_meta_string() {
    const std::size_t len = *rc::gen::inRange<std::size_t>(0, 12);
    return *rc::gen::container<std::string>(
        len, rc::gen::inRange<char>('a', static_cast<char>('z' + 1)));
}

// Optional metadata. Roughly half the records carry no metadata (an explicit
// absence); the rest carry a map drawn from a subset of the four supported
// value types. Doubles are finite so the loaded map compares equal.
std::optional<Metadata> gen_metadata() {
    if (!*rc::gen::arbitrary<bool>()) {
        return std::nullopt;
    }
    Metadata m;
    if (*rc::gen::arbitrary<bool>()) {
        m.emplace("k_str", MetadataValue{gen_meta_string()});
    }
    if (*rc::gen::arbitrary<bool>()) {
        m.emplace("k_int",
                  MetadataValue{static_cast<std::int64_t>(
                      *rc::gen::arbitrary<std::int64_t>())});
    }
    if (*rc::gen::arbitrary<bool>()) {
        const double d =
            static_cast<double>(*rc::gen::inRange<int>(-1000000, 1000001)) /
            1000.0;
        m.emplace("k_dbl", MetadataValue{d});
    }
    if (*rc::gen::arbitrary<bool>()) {
        m.emplace("k_bool", MetadataValue{*rc::gen::arbitrary<bool>()});
    }
    return m;  // may be an empty map: "present but empty", distinct from nullopt
}

// A valid collection name: 1 to 20 characters from [A-Za-z0-9_].
std::string gen_name() {
    static const std::string kAlphabet =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
    const std::size_t len = *rc::gen::inRange<std::size_t>(1, 21);
    std::string s;
    s.reserve(len);
    for (std::size_t i = 0; i < len; ++i) {
        s.push_back(kAlphabet[*rc::gen::inRange<std::size_t>(0, kAlphabet.size())]);
    }
    return s;
}

// Generates a random collection spec: small dimensionality, 0..12 records with
// unique ids, and one of {no index, HNSW, IVF}.
GenSpec gen_spec() {
    GenSpec s;
    s.name   = gen_name();
    s.dim    = *rc::gen::inRange<std::uint32_t>(1, 17);
    s.metric = *rc::gen::element(DistanceMetric::Euclidean,
                                 DistanceMetric::Cosine,
                                 DistanceMetric::DotProduct);

    const std::size_t n = *rc::gen::inRange<std::size_t>(0, 13);
    s.records.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        GenSpec::Rec rec;
        rec.id     = make_id(i);
        rec.vector = gen_vector(s.dim);
        rec.meta   = gen_metadata();
        s.records.push_back(std::move(rec));
    }

    s.index_mode = *rc::gen::inRange<int>(0, 3);  // 0 none, 1 HNSW, 2 IVF
    return s;
}

// Materializes a GenSpec into a fresh collection inside `engine`. Uses the IVF
// auto-default nlist (0) so any record count is valid, including empty.
CollectionHandle build_collection(Engine& engine, const GenSpec& s) {
    auto created = engine.create_collection(s.name, s.dim, s.metric);
    RC_ASSERT(created.is_ok());
    CollectionHandle coll = created.value();

    for (const GenSpec::Rec& rec : s.records) {
        RC_ASSERT(coll->insert(rec.id, rec.vector, rec.meta).is_ok());
    }

    if (s.index_mode != 0) {
        IndexParams params;
        params.nlist = 0;  // IVF auto-default; ignored by HNSW
        const IndexType type =
            (s.index_mode == 1) ? IndexType::HNSW : IndexType::IVF;
        RC_ASSERT(coll->build_index(type, params).is_ok());
    }

    RC_ASSERT(coll->count() == s.records.size());
    return coll;
}

// Asserts that `coll` is equivalent to the original spec: same dimensionality,
// metric, record count, active index type, and every record's vector and
// metadata (present/absent) preserved exactly.
void assert_equivalent(const GenSpec& s, Collection& coll) {
    RC_ASSERT(coll.config().dimensionality == s.dim);
    RC_ASSERT(coll.config().metric == s.metric);
    RC_ASSERT(coll.config().name == s.name);
    RC_ASSERT(coll.count() == s.records.size());
    RC_ASSERT(coll.index_type() == s.expected_index_type());

    for (const GenSpec::Rec& rec : s.records) {
        auto got = coll.get(rec.id);
        RC_ASSERT(got.is_ok());
        // float32 components round-trip bit-exactly through the snapshot.
        RC_ASSERT(got.value().vector == rec.vector);
        // The optional-metadata distinction and every typed value are preserved.
        RC_ASSERT(got.value().meta == rec.meta);
    }
}

}  // namespace

// Feature: vector-vault-db, Property 20: Save/load reconstructs an equivalent collection
// Validates: Requirements 8.1, 8.2
TEST_CASE("Property 20: save/load reconstructs an equivalent collection",
          "[persistence][property]") {
    TempDir tmp;
    const bool ok = rc::check(
        "saving a collection and loading the snapshot yields a collection whose "
        "dimensionality, metric, records, metadata, and index type all match",
        [&] {
            const GenSpec spec = gen_spec();

            Engine src;
            build_collection(src, spec);

            const fs::path path = tmp.file("p20");
            RC_ASSERT(src.save_collection(spec.name, path).is_ok());

            // Load into a fresh engine so the stored name registers cleanly.
            Engine dst;
            auto loaded = dst.load_collection(path);
            RC_ASSERT(loaded.is_ok());

            assert_equivalent(spec, *loaded.value());
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 21: Snapshot save/load/save is byte-identical
// Validates: Requirements 8.3
TEST_CASE("Property 21: snapshot save/load/save is byte-identical",
          "[persistence][property]") {
    TempDir tmp;
    const bool ok = rc::check(
        "saving to A, loading A, and saving the loaded collection to B produces "
        "byte-identical files A and B",
        [&] {
            const GenSpec spec = gen_spec();

            Engine src;
            build_collection(src, spec);

            const fs::path path_a = tmp.file("p21a");
            RC_ASSERT(src.save_collection(spec.name, path_a).is_ok());

            // Load A into a fresh engine, then re-save the loaded collection.
            Engine dst;
            auto loaded = dst.load_collection(path_a);
            RC_ASSERT(loaded.is_ok());
            const std::string loaded_name = loaded.value()->config().name;
            RC_ASSERT(loaded_name == spec.name);

            const fs::path path_b = tmp.file("p21b");
            RC_ASSERT(dst.save_collection(loaded_name, path_b).is_ok());

            const std::vector<unsigned char> bytes_a = read_all(path_a);
            const std::vector<unsigned char> bytes_b = read_all(path_b);
            RC_ASSERT(!bytes_a.empty());
            RC_ASSERT(bytes_a == bytes_b);
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 22: Content corruption is detected on load
// Validates: Requirements 8.8
// A single byte in the checksummed content region (offset >= kHeaderSize) is
// XOR-mutated, leaving the magic and header intact, so the load reaches and
// trips the content integrity check.
TEST_CASE("Property 22: content corruption is detected on load",
          "[persistence][property]") {
    TempDir tmp;
    const bool ok = rc::check(
        "mutating any single byte in the checksummed content region makes the "
        "load fail with Corruption and create no collection",
        [&] {
            const GenSpec spec = gen_spec();

            Engine src;
            build_collection(src, spec);

            const fs::path path = tmp.file("p22");
            RC_ASSERT(src.save_collection(spec.name, path).is_ok());

            std::vector<unsigned char> bytes = read_all(path);
            RC_ASSERT(bytes.size() > snapshot::kHeaderSize);

            // The checksummed region is every byte after the fixed header.
            const std::size_t content_size = bytes.size() - snapshot::kHeaderSize;
            const std::size_t offset =
                snapshot::kHeaderSize +
                *rc::gen::inRange<std::size_t>(0, content_size);
            const unsigned char flip =
                static_cast<unsigned char>(*rc::gen::inRange<int>(1, 256));
            bytes[offset] ^= flip;  // non-zero XOR guarantees the byte changes

            write_all(path, bytes);

            Engine dst;
            auto loaded = dst.load_collection(path);
            RC_ASSERT(loaded.is_error());
            RC_ASSERT(loaded.category() == ErrorCategory::Corruption);

            // No collection was created from the corrupt snapshot.
            RC_ASSERT(dst.list_collections().empty());
        });
    REQUIRE(ok);
}

// Unit / integration tests: persistence edge cases.

namespace {

// Builds a small, deterministic populated + indexed collection for the unit
// tests below, and returns a handle to it.
CollectionHandle make_fixture(Engine& engine, const std::string& name,
                              std::uint32_t dim, std::size_t n,
                              IndexType index = IndexType::HNSW) {
    auto created = engine.create_collection(name, dim, DistanceMetric::Euclidean);
    REQUIRE(created.is_ok());
    CollectionHandle coll = created.value();

    for (std::size_t i = 0; i < n; ++i) {
        std::vector<float> v(dim);
        for (std::uint32_t c = 0; c < dim; ++c) {
            v[c] = static_cast<float>((i * 31 + c) % 97) - 48.0f;
        }
        Metadata meta;
        meta.emplace("idx", MetadataValue{static_cast<std::int64_t>(i)});
        meta.emplace("tag", MetadataValue{std::string("rec") + std::to_string(i)});
        REQUIRE(coll->insert(make_id(i), v, meta).is_ok());
    }

    if (index != IndexType::None) {
        IndexParams params;
        params.nlist = 0;
        REQUIRE(coll->build_index(index, params).is_ok());
    }
    return coll;
}

}  // namespace

// A save targeting an unwritable path aborts with WriteFailure and leaves any
// pre-existing snapshot unmodified. The second save targets a path inside a
// non-existent directory to force the temp-file write to fail before the rename.
TEST_CASE("save failure leaves a pre-existing snapshot unmodified (Req 8.5)",
          "[persistence][unit]") {
    TempDir tmp;
    Engine engine;
    make_fixture(engine, "c", 8, 5);

    const fs::path good = tmp.file("good");
    REQUIRE(engine.save_collection("c", good).is_ok());
    const std::vector<unsigned char> before = read_all(good);
    REQUIRE_FALSE(before.empty());

    // Parent directory does not exist, so the atomic save cannot create its
    // temp file there and the write fails before any rename.
    const fs::path bad = tmp.dir / "no_such_dir" / "snapshot.vv";
    const Status st = engine.save_collection("c", bad);
    REQUIRE(st.is_error());
    REQUIRE(st.category() == ErrorCategory::WriteFailure);

    // The pre-existing snapshot is untouched: same bytes, still loadable.
    const std::vector<unsigned char> after = read_all(good);
    REQUIRE(after == before);

    Engine reload;
    auto loaded = reload.load_collection(good);
    REQUIRE(loaded.is_ok());
    REQUIRE(loaded.value()->count() == 5);
}

// Loading from a path that does not exist returns SnapshotNotFound and creates
// no collection.
TEST_CASE("loading a missing path returns SnapshotNotFound (Req 8.6)",
          "[persistence][unit]") {
    TempDir tmp;
    Engine engine;

    const fs::path missing = tmp.file("does_not_exist");
    auto loaded = engine.load_collection(missing);
    REQUIRE(loaded.is_error());
    REQUIRE(loaded.category() == ErrorCategory::SnapshotNotFound);
    REQUIRE(engine.list_collections().empty());
}

// A file whose contents are not a parseable snapshot (wrong magic) returns
// SnapshotNotFound and creates no collection.
TEST_CASE("loading a garbage / wrong-magic file returns SnapshotNotFound (Req 8.6)",
          "[persistence][unit]") {
    TempDir tmp;
    Engine engine;

    // A buffer larger than the header but starting with non-matching bytes. The
    // magic check fails before any checksum/version logic.
    std::vector<unsigned char> garbage(snapshot::kHeaderSize + 128);
    for (std::size_t i = 0; i < garbage.size(); ++i) {
        garbage[i] = static_cast<unsigned char>((i * 37 + 5) & 0xFF);
    }
    garbage[0] = 'X';  // ensure the magic does not accidentally match

    const fs::path path = tmp.file("garbage");
    write_all(path, garbage);

    auto loaded = engine.load_collection(path);
    REQUIRE(loaded.is_error());
    REQUIRE(loaded.category() == ErrorCategory::SnapshotNotFound);
    REQUIRE(engine.list_collections().empty());

    // A too-small file (smaller than the header) is likewise not a snapshot.
    const fs::path tiny = tmp.file("tiny");
    write_all(tiny, std::vector<unsigned char>{'V', 'V', 'A', 'U'});
    auto tiny_loaded = engine.load_collection(tiny);
    REQUIRE(tiny_loaded.is_error());
    REQUIRE(tiny_loaded.category() == ErrorCategory::SnapshotNotFound);
}

// A snapshot whose stored format-version is not supported is rejected with
// UnsupportedVersion and creates no collection. The version field lives in the
// header (outside the content CRC), so bumping it still passes the checksum.
TEST_CASE("loading an unsupported format version returns UnsupportedVersion (Req 8.7)",
          "[persistence][unit]") {
    TempDir tmp;
    Engine engine;
    make_fixture(engine, "c", 6, 4);

    const fs::path path = tmp.file("versioned");
    REQUIRE(engine.save_collection("c", path).is_ok());

    std::vector<unsigned char> bytes = read_all(path);
    REQUIRE(bytes.size() > snapshot::kHeaderSize);

    // format_version is the little-endian u32 immediately after the 8-byte
    // magic. Overwrite it with a version this engine does not support.
    const std::size_t version_offset = sizeof(snapshot::kMagic);  // 8
    const std::uint32_t bogus_version = snapshot::kFormatVersion + 1000;
    for (int i = 0; i < 4; ++i) {
        bytes[version_offset + i] =
            static_cast<unsigned char>((bogus_version >> (8 * i)) & 0xFF);
    }
    write_all(path, bytes);

    Engine reload;
    auto loaded = reload.load_collection(path);
    REQUIRE(loaded.is_error());
    REQUIRE(loaded.category() == ErrorCategory::UnsupportedVersion);
    REQUIRE(reload.list_collections().empty());
}

// A memory-mapped load resolves vector reads through the mapping: a large
// collection is saved and loaded, and every stored vector and its metadata is
// read back exactly through the mmap-backed record store. (A strict
// "not heap-copied" assertion is not portably observable.)
TEST_CASE("memory-mapped load resolves vector reads correctly (Req 8.4)",
          "[persistence][integration]") {
    TempDir tmp;

    constexpr std::uint32_t kDim = 64;
    constexpr std::size_t   kCount = 2000;

    Engine src;
    auto created = src.create_collection("big", kDim, DistanceMetric::Euclidean);
    REQUIRE(created.is_ok());
    CollectionHandle coll = created.value();

    // Deterministic vectors so the reload can be checked exactly.
    auto component = [](std::size_t i, std::uint32_t c) {
        return static_cast<float>((i * 131 + c * 7) % 1000) / 4.0f - 100.0f;
    };
    for (std::size_t i = 0; i < kCount; ++i) {
        std::vector<float> v(kDim);
        for (std::uint32_t c = 0; c < kDim; ++c) {
            v[c] = component(i, c);
        }
        // Give roughly every other record metadata to exercise both paths.
        if (i % 2 == 0) {
            Metadata meta;
            meta.emplace("n", MetadataValue{static_cast<std::int64_t>(i)});
            REQUIRE(coll->insert(make_id(i), v, meta).is_ok());
        } else {
            REQUIRE(coll->insert(make_id(i), v, std::nullopt).is_ok());
        }
    }
    REQUIRE(coll->count() == kCount);

    const fs::path path = tmp.file("big");
    REQUIRE(src.save_collection("big", path).is_ok());

    Engine dst;
    auto loaded = dst.load_collection(path);
    REQUIRE(loaded.is_ok());
    CollectionHandle reloaded = loaded.value();
    REQUIRE(reloaded->config().dimensionality == kDim);
    REQUIRE(reloaded->count() == kCount);

    // Every vector read through the mapping matches the stored components
    // exactly, and the metadata present/absent distinction survives.
    for (std::size_t i = 0; i < kCount; ++i) {
        auto got = reloaded->get(make_id(i));
        REQUIRE(got.is_ok());
        const std::vector<float>& v = got.value().vector;
        REQUIRE(v.size() == kDim);
        for (std::uint32_t c = 0; c < kDim; ++c) {
            REQUIRE(v[c] == component(i, c));
        }
        if (i % 2 == 0) {
            REQUIRE(got.value().meta.has_value());
        } else {
            REQUIRE_FALSE(got.value().meta.has_value());
        }
    }
}
