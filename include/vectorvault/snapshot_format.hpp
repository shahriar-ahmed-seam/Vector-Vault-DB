#ifndef VECTORVAULT_SNAPSHOT_FORMAT_HPP
#define VECTORVAULT_SNAPSHOT_FORMAT_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "vectorvault/types.hpp"

namespace vectorvault {

// On-disk snapshot format.
//
// A Snapshot is the durable binary representation of a Collection: its
// configuration, every Vector_Record (vector + optional metadata), and the
// Index if one exists. All multi-byte integers are little-endian. The format is
// versioned and checksummed so loads can reject unsupported versions and detect
// corruption, and is laid out deterministically so a save -> load -> save cycle
// is byte-identical.
//
// File layout (in file order):
//   1. Header (fixed-size, see kHeaderSize) - magic, version, flags,
//      dimensionality, metric, index_type, record_count, name_length, the
//      offsets/lengths of the vector and index regions, and the content CRC-64.
//   2. Collection name (name_length bytes, UTF-8).
//   3. Record directory: record_count entries sorted by id ascending; each is
//      `id_length u32 | id bytes | vector_index u64 | meta_length u32 | meta`.
//   4. Zero padding to the next 64-byte boundary.
//   5. Vector region: record_count * dimensionality * 4 bytes, in
//      record-directory (sorted id) order; begins on a 64-byte boundary so a
//      memory mapping yields aligned vectors.
//   6. Index region: the serialized HNSW/IVF structure, present iff
//      index_type != 0.
//
// Determinism rules: records ascending by id; metadata keys sorted with fixed
// type tags; reserved and padding bytes zero-filled; the index region uses a
// canonical node ordering with fixed-width fields; the vector region follows
// record-directory order.
//
// Header size note: the enumerated header fields total 76 bytes, so the header
// is fixed at exactly that packed little-endian size (kHeaderSize). The CRC
// covers all bytes after the header, and the vector region is independently
// 64-byte aligned via explicit padding, so the precise header size does not
// affect alignment.
namespace snapshot {

// 8-byte magic identifying a Vector-Vault snapshot and its on-disk generation.
inline constexpr char kMagic[8] = {'V', 'V', 'A', 'U', 'L', 'T', '0', '1'};

// Current snapshot format version stored in the header.
inline constexpr std::uint32_t kFormatVersion = 1;

// Region alignment: the vector region begins on a 64-byte boundary so the
// mapping is aligned for SIMD reads, matching the allocator.
inline constexpr std::uint64_t kAlignment = 64;

// Fixed-width metadata value type tags written before each metadata value so
// the canonical encoding is self-describing and deterministic. The values are
// stable and must not change once written.
enum class MetaTypeTag : std::uint8_t {
    String = 0,
    Int64  = 1,
    Double = 2,
    Bool   = 3
};

// Packed size of the header, in bytes (sum of every header field below); the
// CRC is computed over the bytes that follow this many bytes from the start of
// the file.
//   magic 8 + format_version 4 + flags 4 + dimensionality 4 + metric 1 +
//   index_type 1 + reserved 2 + record_count 8 + name_length 4 +
//   vector_region_off 8 + vector_region_len 8 + index_region_off 8 +
//   index_region_len 8 + content_crc64 8 = 76
inline constexpr std::size_t kHeaderSize = 76;

// Byte offset of the content_crc64 field within the header. Load reads the
// stored CRC from here and recomputes the checksum over bytes [kHeaderSize..].
inline constexpr std::size_t kContentCrcOffset = 68;

// Parsed view of the fixed header. Field types and order mirror the on-disk
// layout exactly (used by save to build the header and by load to validate it).
struct Header {
    char          magic[8];
    std::uint32_t format_version = kFormatVersion;
    std::uint32_t flags          = 0;
    std::uint32_t dimensionality = 0;
    std::uint8_t  metric         = 0;
    std::uint8_t  index_type     = 0;
    std::uint8_t  reserved[2]    = {0, 0};
    std::uint64_t record_count   = 0;
    std::uint32_t name_length    = 0;
    std::uint64_t vector_region_off = 0;
    std::uint64_t vector_region_len = 0;
    std::uint64_t index_region_off  = 0;
    std::uint64_t index_region_len  = 0;
    std::uint64_t content_crc64     = 0;
};

// Little-endian, append-only byte buffer. Centralises the primitive writes so
// every part of the format (header, record directory, vectors, index region)
// shares one deterministic encoding.
class ByteWriter {
public:
    ByteWriter() = default;

    // The accumulated bytes.
    const std::vector<std::byte>& bytes() const { return buffer_; }
    std::vector<std::byte>&       bytes() { return buffer_; }

    // Number of bytes written so far (also the offset of the next write).
    std::size_t size() const { return buffer_.size(); }

    void put_u8(std::uint8_t v) { buffer_.push_back(static_cast<std::byte>(v)); }

    void put_u32(std::uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            put_u8(static_cast<std::uint8_t>(v >> (8 * i)));
        }
    }

    void put_u64(std::uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            put_u8(static_cast<std::uint8_t>(v >> (8 * i)));
        }
    }

    void put_i64(std::int64_t v) {
        put_u64(static_cast<std::uint64_t>(v));
    }

    // Writes a float32 as its raw IEEE-754 bits in little-endian order.
    void put_f32(float v) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        put_u32(bits);
    }

    // Writes a double as its raw IEEE-754 bits in little-endian order.
    void put_f64(double v) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        put_u64(bits);
    }

    // Appends raw bytes verbatim.
    void put_raw(const void* data, std::size_t len) {
        const auto* p = static_cast<const std::byte*>(data);
        buffer_.insert(buffer_.end(), p, p + len);
    }

    // Writes a length-prefixed (u32) byte string: `length | bytes`.
    void put_string(const std::string& s) {
        put_u32(static_cast<std::uint32_t>(s.size()));
        put_raw(s.data(), s.size());
    }

    // Appends zero bytes until the buffer length is a multiple of `alignment`.
    void pad_to_alignment(std::uint64_t alignment) {
        while (alignment != 0 && (buffer_.size() % alignment) != 0) {
            put_u8(0);
        }
    }

private:
    std::vector<std::byte> buffer_;
};

// Little-endian, bounds-checked sequential reader; the mirror of ByteWriter used
// by snapshot load. Every read advances a cursor and is bounds-checked: a read
// that would run past the end trips a sticky failure flag and yields a
// zero/empty value, so callers may parse optimistically and check ok() once at
// the end. Floating-point values are read back from their raw IEEE-754 bits.
class ByteReader {
public:
    ByteReader(const std::byte* data, std::size_t size)
        : data_(data), size_(size) {}

    // True while no read has overrun the buffer.
    bool ok() const { return ok_; }

    // Bytes consumed so far / still available.
    std::size_t offset() const { return pos_; }
    std::size_t remaining() const { return pos_ <= size_ ? size_ - pos_ : 0; }

    // The address of the next unread byte (valid only while ok()).
    const std::byte* cursor() const { return data_ + pos_; }

    std::uint8_t get_u8() {
        if (!ensure(1)) return 0;
        return static_cast<std::uint8_t>(data_[pos_++]);
    }

    std::uint32_t get_u32() {
        if (!ensure(4)) return 0;
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= static_cast<std::uint32_t>(
                     static_cast<std::uint8_t>(data_[pos_ + i]))
                 << (8 * i);
        }
        pos_ += 4;
        return v;
    }

    std::uint64_t get_u64() {
        if (!ensure(8)) return 0;
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<std::uint64_t>(
                     static_cast<std::uint8_t>(data_[pos_ + i]))
                 << (8 * i);
        }
        pos_ += 8;
        return v;
    }

    std::int64_t get_i64() { return static_cast<std::int64_t>(get_u64()); }

    float get_f32() {
        const std::uint32_t bits = get_u32();
        float v = 0.0f;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }

    double get_f64() {
        const std::uint64_t bits = get_u64();
        double v = 0.0;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }

    // Copies `len` raw bytes into `dst`. Returns false (and trips ok_) on
    // overrun, leaving `dst` partially untouched.
    bool get_raw(void* dst, std::size_t len) {
        if (!ensure(len)) return false;
        std::memcpy(dst, data_ + pos_, len);
        pos_ += len;
        return true;
    }

    // Reads a u32 length prefix followed by that many bytes as a string,
    // mirroring ByteWriter::put_string.
    std::string get_string() {
        const std::uint32_t len = get_u32();
        if (!ensure(len)) return std::string{};
        std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
        pos_ += len;
        return s;
    }

    // Advances the cursor by `len` bytes without reading them.
    bool skip(std::size_t len) {
        if (!ensure(len)) return false;
        pos_ += len;
        return true;
    }

    // Returns a sub-reader over the next `len` bytes and advances past them in
    // this reader. On overrun the parent fails and an empty sub-reader is
    // returned so the caller's subsequent reads also fail safely.
    ByteReader sub(std::size_t len) {
        if (!ensure(len)) return ByteReader(data_ + size_, 0);
        ByteReader child(data_ + pos_, len);
        pos_ += len;
        return child;
    }

private:
    // Returns true when `need` more bytes are available; otherwise trips the
    // sticky failure flag. The second comparison guards against pos_ + need
    // wrapping around on a maliciously large `need`.
    bool ensure(std::size_t need) {
        if (!ok_) return false;
        if (pos_ + need > size_ || pos_ + need < pos_) {
            ok_ = false;
            return false;
        }
        return true;
    }

    const std::byte* data_;
    std::size_t      size_;
    std::size_t      pos_ = 0;
    bool             ok_  = true;
};

}  // namespace snapshot

// A consistent in-memory view of a Collection captured under its readers-writer
// lock and handed to the Persistence_Manager for serialization, so encoding
// happens against a point-in-time view without reaching into Collection
// internals. Records are provided already sorted by id ascending (matching the
// on-disk record-directory ordering); the index, if any, is provided as its
// already-canonical serialized bytes.
struct CollectionSnapshotData {
    // One record in sorted-id order: identifier, float32 components, and the
    // optional metadata (absence preserved).
    struct Record {
        RecordId                id;
        std::vector<float>      vector;
        std::optional<Metadata> meta;
    };

    CollectionConfig       config;                       // name, dim, metric
    IndexType              index_type = IndexType::None;  // active index, if any
    std::vector<Record>    records;                       // sorted by id ascending
    std::vector<std::byte> index_blob;                    // canonical index bytes
};

}  // namespace vectorvault

#endif  // VECTORVAULT_SNAPSHOT_FORMAT_HPP
