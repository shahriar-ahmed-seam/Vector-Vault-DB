#ifndef VECTORVAULT_TYPES_HPP
#define VECTORVAULT_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace vectorvault {

// The numeric values are part of the on-disk snapshot format: the header stores
// `metric` and `index_type` as single bytes, so these enumerator values must
// remain stable.
enum class DistanceMetric : std::uint8_t {
    Euclidean  = 0,  // L2 distance
    Cosine     = 1,  // 1 - cosine similarity
    DotProduct = 2   // sum of element-wise products
};

enum class IndexType : std::uint8_t {
    None = 0,  // no index built over the collection
    HNSW = 1,  // Hierarchical Navigable Small World graph
    IVF  = 2   // Inverted File with coarse quantization
};

// A caller-supplied identifier that must be non-empty.
using RecordId = std::string;

// A single metadata value: a UTF-8 string, a signed 64-bit integer, a double,
// or a boolean.
using MetadataValue = std::variant<std::string, std::int64_t, double, bool>;

// Metadata is a canonical, ordered map of string keys to typed values. std::map
// keeps keys in ascending order, which makes serialization deterministic and
// underpins the byte-identical round-trip property.
using Metadata = std::map<std::string, MetadataValue>;

// A region of storage handed out by the Memory_Allocator. Defined here in the
// shared model because record storage refers to it; the allocator owns the
// lifetime of the underlying memory and guarantees vector blocks start on a
// 64-byte boundary.
struct Block {
    void*       data = nullptr;  // start address (64-byte aligned for vectors)
    std::size_t size = 0;        // size of the block in bytes
};

struct CollectionConfig {
    std::string    name;            // 1..255 characters
    std::uint32_t  dimensionality;  // 1..65536 inclusive
    DistanceMetric metric;          // Euclidean, Cosine, or DotProduct
};

// One entry per collection returned by Engine::list_collections.
struct CollectionInfo {
    std::string    name;
    std::uint32_t  dimensionality;
    DistanceMetric metric;
    std::uint64_t  record_count;
};

// The in-memory store entry for one Vector_Record. Points at the aligned
// float32 vector block owned by the allocator and carries the optional
// metadata; the optional makes metadata absence explicit.
struct RecordSlot {
    Block                   vector_block;  // 64-byte aligned float32[dimensionality]
    std::optional<Metadata> meta;          // absent when no metadata was supplied
};

// One element of a batch supplied to Collection::insert_batch: the
// caller-supplied identifier, the vector components, and optional metadata.
// `meta` being absent (std::nullopt) records an explicit "no metadata" intent,
// mirroring RecordSlot::meta so that absence survives an insert/get round-trip.
struct RecordInput {
    RecordId                id;      // non-empty
    std::vector<float>      vector;  // length must equal the dimensionality
    std::optional<Metadata> meta;    // absent when no metadata is supplied
};

// The result of Collection::get: a copy of the stored float32 vector together
// with the optional metadata; the optional makes metadata presence/absence
// explicit to the caller.
struct RecordView {
    std::vector<float>      vector;  // a copy of the stored components
    std::optional<Metadata> meta;    // present iff metadata was stored
};

// One result of a k-nearest-neighbor query: a record identifier and the
// distance computed for it under the collection's metric.
struct Neighbor {
    RecordId id;
    float    distance;
};

}  // namespace vectorvault

#endif  // VECTORVAULT_TYPES_HPP
