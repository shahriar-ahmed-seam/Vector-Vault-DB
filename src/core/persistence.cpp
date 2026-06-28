#include "vectorvault/persistence.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "vectorvault/crc64.hpp"
#include "vectorvault/distance.hpp"
#include "vectorvault/hnsw_index.hpp"
#include "vectorvault/ivf_index.hpp"
#include "vectorvault/mmap_region.hpp"
#include "vectorvault/snapshot_format.hpp"

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <io.h>      // _commit
#else
#  include <unistd.h>  // fsync, close
#  include <fcntl.h>   // open
#endif

namespace vectorvault {

namespace {

using snapshot::ByteReader;
using snapshot::ByteWriter;

// Reads one record's metadata blob written by write_metadata into `out`,
// preserving the present/absent distinction. Returns false on a malformed blob
// (a truncated read or an unknown type tag; trailing bytes are detected by the
// caller). The variant alternatives and their fixed type tags mirror
// snapshot::MetaTypeTag exactly, so the decode is the inverse of the encode.
bool read_metadata(ByteReader& r, std::optional<Metadata>& out) {
    const std::uint8_t present = r.get_u8();
    if (!r.ok()) {
        return false;
    }
    if (present == 0) {
        out = std::nullopt;
        return true;
    }

    Metadata m;
    const std::uint32_t count = r.get_u32();
    if (!r.ok()) {
        return false;
    }
    for (std::uint32_t i = 0; i < count; ++i) {
        std::string key = r.get_string();
        const std::uint8_t tag = r.get_u8();
        if (!r.ok()) {
            return false;
        }
        switch (static_cast<snapshot::MetaTypeTag>(tag)) {
            case snapshot::MetaTypeTag::String: {
                std::string value = r.get_string();
                if (!r.ok()) return false;
                m.emplace(std::move(key), MetadataValue{std::move(value)});
                break;
            }
            case snapshot::MetaTypeTag::Int64: {
                const std::int64_t value = r.get_i64();
                if (!r.ok()) return false;
                m.emplace(std::move(key), MetadataValue{value});
                break;
            }
            case snapshot::MetaTypeTag::Double: {
                const double value = r.get_f64();
                if (!r.ok()) return false;
                m.emplace(std::move(key), MetadataValue{value});
                break;
            }
            case snapshot::MetaTypeTag::Bool: {
                const bool value = r.get_u8() != 0;
                if (!r.ok()) return false;
                m.emplace(std::move(key), MetadataValue{value});
                break;
            }
            default:
                return false;  // unknown type tag -> malformed
        }
    }
    out = std::move(m);
    return true;
}

// Appends one record's metadata blob to `w` using the canonical encoding:
//   present u8 (0 = no metadata, 1 = metadata present)
//   if present: entry_count u32, then for each (key, value) in ascending key
//   order: key (u32 length + bytes), type_tag u8, then the typed value.
// Metadata is a std::map, so iteration visits keys in sorted order, giving a
// deterministic layout. Absence is preserved distinctly from an empty map via
// the present flag.
void write_metadata(ByteWriter& w, const std::optional<Metadata>& meta) {
    if (!meta.has_value()) {
        w.put_u8(0);
        return;
    }
    w.put_u8(1);
    const Metadata& m = *meta;
    w.put_u32(static_cast<std::uint32_t>(m.size()));
    for (const auto& kv : m) {
        w.put_string(kv.first);
        const MetadataValue& value = kv.second;
        // The variant alternatives are, in order: string, int64, double, bool;
        // the fixed type tags must match snapshot::MetaTypeTag.
        if (const auto* s = std::get_if<std::string>(&value)) {
            w.put_u8(static_cast<std::uint8_t>(snapshot::MetaTypeTag::String));
            w.put_string(*s);
        } else if (const auto* i = std::get_if<std::int64_t>(&value)) {
            w.put_u8(static_cast<std::uint8_t>(snapshot::MetaTypeTag::Int64));
            w.put_i64(*i);
        } else if (const auto* d = std::get_if<double>(&value)) {
            w.put_u8(static_cast<std::uint8_t>(snapshot::MetaTypeTag::Double));
            w.put_f64(*d);
        } else if (const auto* b = std::get_if<bool>(&value)) {
            w.put_u8(static_cast<std::uint8_t>(snapshot::MetaTypeTag::Bool));
            w.put_u8(*b ? 1u : 0u);
        }
    }
}

// Builds a process-and-call-unique temporary file name beside the target so the
// final rename stays within one directory (a requirement for an atomic rename).
std::filesystem::path make_temp_path(const std::filesystem::path& target) {
    static std::atomic<std::uint64_t> counter{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::uint64_t n = counter.fetch_add(1, std::memory_order_relaxed);

    std::filesystem::path dir = target.parent_path();
    if (dir.empty()) {
        dir = std::filesystem::path(".");
    }
    std::string name = target.filename().string() + ".vvtmp-" +
                       std::to_string(static_cast<unsigned long long>(now)) + "-" +
                       std::to_string(static_cast<unsigned long long>(n));
    return dir / name;
}

// Writes `bytes` to a fresh file at `temp_path` and flushes it durably to disk
// (so the subsequent rename publishes fully-persisted data). Returns true on
// success; on any failure the (possibly partial) temp file is removed.
bool write_temp_durably(const std::filesystem::path& temp_path,
                        const std::vector<std::byte>& bytes) {
#if defined(_WIN32)
    FILE* fp = _wfopen(temp_path.wstring().c_str(), L"wb");
#else
    FILE* fp = std::fopen(temp_path.c_str(), "wb");
#endif
    if (fp == nullptr) {
        return false;
    }

    bool ok = true;
    if (!bytes.empty()) {
        const std::size_t written =
            std::fwrite(bytes.data(), 1, bytes.size(), fp);
        ok = (written == bytes.size());
    }

    if (ok && std::fflush(fp) != 0) {
        ok = false;
    }

    if (ok) {
        // Flush the file's data and metadata through the OS cache to stable
        // storage so the rename publishes durable content.
#if defined(_WIN32)
        if (_commit(_fileno(fp)) != 0) {
            ok = false;
        }
#else
        if (::fsync(::fileno(fp)) != 0) {
            ok = false;
        }
#endif
    }

    if (std::fclose(fp) != 0) {
        ok = false;
    }

    if (!ok) {
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
    }
    return ok;
}

// Atomically renames `temp_path` over `target`, replacing any existing file.
// Returns true on success. On failure the caller removes the temp file; the
// pre-existing target is never touched because the rename is the only step that
// modifies it.
bool atomic_rename(const std::filesystem::path& temp_path,
                   const std::filesystem::path& target) {
#if defined(_WIN32)
    return MoveFileExW(temp_path.wstring().c_str(), target.wstring().c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    // POSIX rename(2) atomically replaces the destination within a filesystem.
    if (std::rename(temp_path.c_str(), target.c_str()) != 0) {
        return false;
    }
    // Best-effort: flush the directory entry so the rename itself is durable.
    const std::filesystem::path dir =
        target.parent_path().empty() ? std::filesystem::path(".")
                                     : target.parent_path();
    const int dfd = ::open(dir.c_str(), O_RDONLY);
    if (dfd >= 0) {
        ::fsync(dfd);
        ::close(dfd);
    }
    return true;
#endif
}

}  // namespace

std::vector<std::byte> PersistenceManager::encode(
    const CollectionSnapshotData& data) {
    namespace fmt = snapshot;

    const std::uint32_t dim = data.config.dimensionality;
    const std::uint64_t record_count = data.records.size();

    // Build everything that follows the fixed header into `body`. A body offset
    // of 0 corresponds to file offset fmt::kHeaderSize, so absolute region
    // offsets are fmt::kHeaderSize + (offset within body).
    ByteWriter body;

    // Collection name.
    body.put_raw(data.config.name.data(), data.config.name.size());

    // Record directory: entries in the already-sorted (ascending id) order
    // provided by the capture. vector_index is the record's position, matching
    // its slot in the vector region.
    for (std::uint64_t i = 0; i < record_count; ++i) {
        const CollectionSnapshotData::Record& rec = data.records[i];
        body.put_string(rec.id);
        body.put_u64(i);  // vector_index

        ByteWriter meta;
        write_metadata(meta, rec.meta);
        body.put_u32(static_cast<std::uint32_t>(meta.size()));
        body.put_raw(meta.bytes().data(), meta.size());
    }

    // Zero padding so the vector region begins on a 64-byte boundary measured
    // from the start of the file.
    while (((fmt::kHeaderSize + body.size()) % fmt::kAlignment) != 0) {
        body.put_u8(0);
    }

    // Vector region: record_count * dim float32 values, in record-directory
    // order, written as little-endian IEEE-754 bits.
    const std::uint64_t vector_region_off = fmt::kHeaderSize + body.size();
    for (const CollectionSnapshotData::Record& rec : data.records) {
        for (std::uint32_t c = 0; c < dim; ++c) {
            body.put_f32(rec.vector[c]);
        }
    }
    const std::uint64_t vector_region_len =
        record_count * static_cast<std::uint64_t>(dim) * sizeof(float);

    // Index region: present iff an index exists (index_type != None), placed
    // immediately after the vector region.
    const std::uint64_t index_region_off = fmt::kHeaderSize + body.size();
    const std::uint64_t index_region_len = data.index_blob.size();
    if (!data.index_blob.empty()) {
        body.put_raw(data.index_blob.data(), data.index_blob.size());
    }

    // Content checksum covers every byte after the header.
    const std::uint64_t crc =
        crc64_ecma(body.bytes().data(), body.size());

    // Header. Field order and widths match the on-disk layout exactly and total
    // fmt::kHeaderSize bytes.
    ByteWriter header;
    header.put_raw(fmt::kMagic, sizeof(fmt::kMagic));
    header.put_u32(fmt::kFormatVersion);
    header.put_u32(0);  // flags (reserved, zero)
    header.put_u32(dim);
    header.put_u8(static_cast<std::uint8_t>(data.config.metric));
    header.put_u8(static_cast<std::uint8_t>(data.index_type));
    header.put_u8(0);  // reserved
    header.put_u8(0);  // reserved
    header.put_u64(record_count);
    header.put_u32(static_cast<std::uint32_t>(data.config.name.size()));
    header.put_u64(vector_region_off);
    header.put_u64(vector_region_len);
    header.put_u64(index_region_off);
    header.put_u64(index_region_len);
    header.put_u64(crc);

    // Concatenate header + body into the final image.
    std::vector<std::byte> image = std::move(header.bytes());
    image.insert(image.end(), body.bytes().begin(), body.bytes().end());
    return image;
}

Status PersistenceManager::save(const Collection& collection,
                                const std::filesystem::path& path) {
    // Capture a consistent view under the collection's lock and encode it
    // deterministically.
    const CollectionSnapshotData data = collection.snapshot_data();
    const std::vector<std::byte> image = encode(data);

    // Write to a temp file in the target's directory, flush durably, then
    // atomically rename over the target. Any failure before the rename leaves a
    // pre-existing snapshot at `path` unmodified.
    const std::filesystem::path temp_path = make_temp_path(path);

    if (!write_temp_durably(temp_path, image)) {
        return Status::error(
            ErrorCategory::WriteFailure,
            "failed to write snapshot temp file '" + temp_path.string() + "'");
    }

    if (!atomic_rename(temp_path, path)) {
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
        return Status::error(
            ErrorCategory::WriteFailure,
            "failed to atomically rename snapshot into '" + path.string() + "'");
    }

    return Status::ok();
}

Result<std::unique_ptr<Collection>> PersistenceManager::load(
    const std::filesystem::path& path, MemoryAllocator& allocator,
    CollectionTag tag, const DistanceCalculator& calculator) {
    namespace fmt = snapshot;
    using LoadResult = Result<std::unique_ptr<Collection>>;

    // Existence and minimum size: a missing, non-regular, too-small, or
    // unreadable file is not a parseable snapshot.
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return LoadResult::error(
            ErrorCategory::SnapshotNotFound,
            "snapshot file '" + path.string() + "' does not exist");
    }
    if (!std::filesystem::is_regular_file(path, ec) || ec) {
        return LoadResult::error(
            ErrorCategory::SnapshotNotFound,
            "snapshot path '" + path.string() + "' is not a regular file");
    }
    const std::uintmax_t file_size = std::filesystem::file_size(path, ec);
    if (ec || file_size < fmt::kHeaderSize) {
        return LoadResult::error(
            ErrorCategory::SnapshotNotFound,
            "file '" + path.string() +
                "' is too small to contain a snapshot header");
    }

    // Memory-map the file read-only so the vector region is accessed through
    // the mapping rather than copied into the heap.
    std::unique_ptr<MmapRegion> region = MmapRegion::map_file_readonly(path);
    if (!region || !region->valid() || region->size() < fmt::kHeaderSize) {
        return LoadResult::error(
            ErrorCategory::SnapshotNotFound,
            "failed to memory-map snapshot '" + path.string() + "'");
    }
    const std::byte* const base  = region->data();
    const std::size_t       total = region->size();

    // Magic + header parse.
    fmt::ByteReader hr(base, total);
    char magic[8];
    if (!hr.get_raw(magic, sizeof(magic)) ||
        std::memcmp(magic, fmt::kMagic, sizeof(fmt::kMagic)) != 0) {
        return LoadResult::error(
            ErrorCategory::SnapshotNotFound,
            "file '" + path.string() + "' is not a Vector-Vault snapshot");
    }

    fmt::Header h;
    std::memcpy(h.magic, magic, sizeof(magic));
    h.format_version    = hr.get_u32();
    h.flags             = hr.get_u32();
    h.dimensionality    = hr.get_u32();
    h.metric            = hr.get_u8();
    h.index_type        = hr.get_u8();
    h.reserved[0]       = hr.get_u8();
    h.reserved[1]       = hr.get_u8();
    h.record_count      = hr.get_u64();
    h.name_length       = hr.get_u32();
    h.vector_region_off = hr.get_u64();
    h.vector_region_len = hr.get_u64();
    h.index_region_off  = hr.get_u64();
    h.index_region_len  = hr.get_u64();
    h.content_crc64     = hr.get_u64();
    if (!hr.ok()) {
        return LoadResult::error(
            ErrorCategory::SnapshotNotFound,
            "snapshot '" + path.string() + "' has a malformed header");
    }

    // Format version.
    if (h.format_version != fmt::kFormatVersion) {
        return LoadResult::error(
            ErrorCategory::UnsupportedVersion,
            "snapshot format version " + std::to_string(h.format_version) +
                " is not supported (this engine reads version " +
                std::to_string(fmt::kFormatVersion) + ")");
    }

    // Content checksum over every byte after the header.
    const std::uint64_t actual_crc =
        crc64_ecma(base + fmt::kHeaderSize, total - fmt::kHeaderSize);
    if (actual_crc != h.content_crc64) {
        return LoadResult::error(
            ErrorCategory::Corruption,
            "snapshot '" + path.string() +
                "' failed its content integrity check");
    }

    // Every gate has passed. The checksum proves the bytes are intact, so any
    // structural inconsistency from here on indicates a malformed (corrupt)
    // snapshot rather than a wrong version or a missing file.
    auto corrupt = [&](const std::string& why) {
        return LoadResult::error(
            ErrorCategory::Corruption,
            "snapshot '" + path.string() + "' is corrupt: " + why);
    };

    // Validate the configuration enumerations and dimensionality.
    if (h.dimensionality < 1 || h.dimensionality > 65536) {
        return corrupt("invalid dimensionality " +
                       std::to_string(h.dimensionality));
    }
    if (h.metric > static_cast<std::uint8_t>(DistanceMetric::DotProduct)) {
        return corrupt("invalid metric tag " + std::to_string(h.metric));
    }
    if (h.index_type > static_cast<std::uint8_t>(IndexType::IVF)) {
        return corrupt("invalid index type tag " +
                       std::to_string(h.index_type));
    }

    const std::uint32_t dim    = h.dimensionality;
    const std::uint64_t stride = static_cast<std::uint64_t>(dim) * sizeof(float);
    const IndexType     index_type = static_cast<IndexType>(h.index_type);

    // Bound record_count by the file size (every directory entry occupies more
    // than one byte) so a bogus count cannot drive a pathological allocation.
    if (h.record_count > total) {
        return corrupt("record count exceeds file size");
    }
    // The name must fit between the header and the end of the file.
    if (h.name_length > total - fmt::kHeaderSize) {
        return corrupt("collection name length out of bounds");
    }

    // Validate the region offsets/lengths against the file size and each other.
    if (h.vector_region_off < fmt::kHeaderSize || h.vector_region_off > total ||
        h.vector_region_len > total - h.vector_region_off) {
        return corrupt("vector region out of bounds");
    }
    if (h.vector_region_len != h.record_count * stride) {
        return corrupt("vector region length inconsistent with record count");
    }
    if (h.index_region_off > total ||
        h.index_region_len > total - h.index_region_off) {
        return corrupt("index region out of bounds");
    }
    // An index type implies a non-empty region and vice versa: both index
    // encoders always emit a non-empty header, while a None index has none.
    if ((index_type == IndexType::None) != (h.index_region_len == 0)) {
        return corrupt("index type and index region presence disagree");
    }

    // Body reader positioned just after the fixed header (body offset 0).
    fmt::ByteReader br(base + fmt::kHeaderSize, total - fmt::kHeaderSize);

    // Collection name (name_length bytes, no separate length prefix).
    std::string name;
    if (h.name_length > 0) {
        name.resize(h.name_length);
        if (!br.get_raw(name.data(), h.name_length)) {
            return corrupt("truncated collection name");
        }
    }
    if (name.empty() || name.size() > 255) {
        return corrupt("collection name must be 1 to 255 characters");
    }

    // Record directory: record_count entries in ascending-id order, each
    // `id (u32 len + bytes) | vector_index u64 | meta_length u32 | meta`.
    std::map<RecordId, RecordSlot>                records;
    std::unordered_map<RecordId, const float*>    vec_by_id;
    vec_by_id.reserve(static_cast<std::size_t>(h.record_count) * 2 + 1);

    for (std::uint64_t i = 0; i < h.record_count; ++i) {
        RecordId            id           = br.get_string();
        const std::uint64_t vector_index = br.get_u64();
        const std::uint32_t meta_length  = br.get_u32();
        if (!br.ok()) {
            return corrupt("truncated record directory");
        }

        fmt::ByteReader mr = br.sub(meta_length);
        if (!br.ok()) {
            return corrupt("truncated record metadata blob");
        }
        std::optional<Metadata> meta;
        if (!read_metadata(mr, meta) || !mr.ok() || mr.remaining() != 0) {
            return corrupt("malformed record metadata");
        }

        if (id.empty()) {
            return corrupt("empty record identifier");
        }
        if (vector_index >= h.record_count) {
            return corrupt("record vector index out of range");
        }

        // Point the record's vector block into the mapped vector region rather
        // than copying the components. The region begins on a 64-byte boundary
        // and the mapping base is page-aligned, so each vector is 64-byte
        // aligned. The mapping is read-only; the components are only ever read
        // (get/query), never written through this pointer.
        const std::byte* vptr =
            base + h.vector_region_off + vector_index * stride;

        RecordSlot slot;
        slot.vector_block.data =
            const_cast<void*>(static_cast<const void*>(vptr));
        slot.vector_block.size = static_cast<std::size_t>(stride);
        slot.meta              = std::move(meta);

        auto inserted = records.emplace(id, std::move(slot));
        if (!inserted.second) {
            return corrupt("duplicate record identifier '" + id + "'");
        }
        vec_by_id.emplace(std::move(id), reinterpret_cast<const float*>(vptr));
    }

    // The directory plus its zero padding must end no later than the start of
    // the vector region.
    if (br.offset() > h.vector_region_off - fmt::kHeaderSize) {
        return corrupt("record directory overruns the vector region");
    }

    // (c) Index region: reconstruct the persisted index so its membership
    // matches the records and a re-save is byte-identical.
    std::unique_ptr<Index> index;
    if (index_type != IndexType::None) {
        const std::byte* idx_data = base + h.index_region_off;
        const std::size_t idx_len =
            static_cast<std::size_t>(h.index_region_len);

        if (index_type == IndexType::HNSW) {
            // HNSW persists its full graph; vectors come from the record store
            // (mmap-backed) via the lookup.
            auto vector_for = [&](const RecordId& rid) -> const float* {
                auto it = vec_by_id.find(rid);
                return it == vec_by_id.end() ? nullptr : it->second;
            };
            bool ok = false;
            std::unique_ptr<HnswIndex> hnsw = HnswIndex::deserialize(
                dim, static_cast<DistanceMetric>(h.metric), calculator,
                idx_data, idx_len, vector_for, ok);
            if (!ok || !hnsw) {
                return corrupt("malformed HNSW index region");
            }
            index = std::move(hnsw);
        } else {  // IndexType::IVF
            // IVF persists only its construction parameters; its partitioning
            // is a deterministic function of the live record set, so re-add
            // every record (record-directory order) to rebuild membership.
            fmt::ByteReader ir(idx_data, idx_len);
            IndexParams params;
            params.nlist = ir.get_u32();
            params.seed  = ir.get_u64();
            if (!ir.ok() || ir.remaining() != 0) {
                return corrupt("malformed IVF index region");
            }
            auto ivf = std::make_unique<IvfIndex>(
                dim, static_cast<DistanceMetric>(h.metric), calculator, params);
            for (const auto& entry : records) {
                const float* comp =
                    static_cast<const float*>(entry.second.vector_block.data);
                Status added =
                    ivf->add(entry.first, span<const float>(comp, dim));
                if (added.is_error()) {
                    return corrupt("failed to rebuild IVF index membership");
                }
            }
            index = std::move(ivf);
        }
    }

    // Assemble the Collection and install the loaded state directly. Friend
    // access keeps the vector data mapped (rather than re-inserted and
    // heap-copied) and preserves the exact reconstructed index.
    CollectionConfig config;
    config.name           = std::move(name);
    config.dimensionality = dim;
    config.metric         = static_cast<DistanceMetric>(h.metric);

    auto collection = std::make_unique<Collection>(std::move(config), allocator,
                                                   tag, calculator);
    collection->records_       = std::move(records);
    collection->record_count_  = collection->records_.size();
    collection->index_         = std::move(index);
    collection->index_type_    = index_type;
    collection->mapped_region_ = std::move(region);

    return LoadResult{std::move(collection)};
}

}  // namespace vectorvault
