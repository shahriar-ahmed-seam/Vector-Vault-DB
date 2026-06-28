#ifndef VECTORVAULT_COLLECTION_HPP
#define VECTORVAULT_COLLECTION_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <shared_mutex>

#include "vectorvault/error.hpp"
#include "vectorvault/index.hpp"
#include "vectorvault/memory_allocator.hpp"
#include "vectorvault/snapshot_format.hpp"
#include "vectorvault/span.hpp"
#include "vectorvault/types.hpp"

namespace vectorvault {

// Computes distances under the collection's metric. Owned by the Engine and
// only referenced here, so a forward declaration suffices.
class DistanceCalculator;

// Reconstructs Collections from on-disk Snapshots. Befriended below so it can
// install loaded state without exposing those internals publicly.
class PersistenceManager;

// RAII owner of the read-only mapping backing a loaded collection's vectors.
// Held by unique_ptr with the destructor defined in the .cpp, so this header
// need not include the platform mapping headers.
class MmapRegion;

// A named container of Vector_Records sharing a fixed dimensionality and a
// single distance metric. Owns the record store, immutable configuration, and
// an optional index. A readers-writer lock guards access: reads (get/query)
// take a shared lock, mutations (insert/delete/build/save) take an exclusive
// lock.
//
// Created and owned by the Engine registry, which assigns each Collection a
// unique CollectionTag and a reference to the shared allocator; deleting the
// Collection releases every block carrying that tag.
class Collection {
public:
    // `allocator` and `calculator` are owned by the Engine and outlive every
    // collection. Defined out-of-line so the unique_ptr<MmapRegion> member can
    // be held by a forward-declared type.
    Collection(CollectionConfig config, MemoryAllocator& allocator, CollectionTag tag,
               const DistanceCalculator& calculator);

    // Defaulted in the .cpp so the unique_ptr<MmapRegion> member can be held by
    // a forward-declared type; a loaded collection's mapping is unmapped here.
    ~Collection();

    // Collections are owned in place and referenced through CollectionHandle;
    // copying or moving would invalidate those references and the per-collection
    // lock, so both are disabled.
    Collection(const Collection&) = delete;
    Collection& operator=(const Collection&) = delete;
    Collection(Collection&&) = delete;
    Collection& operator=(Collection&&) = delete;

    // The immutable configuration: name, dimensionality, and metric.
    const CollectionConfig& config() const { return config_; }

    // The current Vector_Record count.
    std::uint64_t count() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return record_count_;
    }

    // Inserts (or overwrites) a single Vector_Record. The request is validated
    // fully before any state changes, so a rejected insert leaves the
    // collection byte-for-byte unchanged:
    //   * empty id                           -> InvalidIdentifier
    //   * vec.size() != dimensionality       -> DimensionMismatch
    //   * any non-finite component (NaN/inf) -> InvalidValue
    //   * allocator exhausted                -> AllocationFailure
    // On success the components are copied into a freshly allocated 64-byte
    // aligned block. Inserting an existing id replaces the stored vector and
    // metadata without changing the count. `meta` left as std::nullopt records
    // an explicit absence of metadata.
    Status insert(const RecordId& id, span<const float> vec,
                  std::optional<Metadata> meta = std::nullopt);

    // Inserts a batch of 1 to 10000 Vector_Records atomically. Every record is
    // validated before any is stored; if any fails identifier, dimensionality,
    // or finite-value validation, no record is stored and an error identifying
    // the first invalid record is returned. A batch size outside [1, 10000] is
    // rejected with InvalidArgument. Allocation is staged so allocator
    // exhaustion mid-batch also leaves the collection unchanged. Duplicate ids
    // within the batch follow last-writer-wins, matching repeated single
    // inserts.
    Status insert_batch(span<const RecordInput> records);

    // Retrieves a Vector_Record by id. Returns a copy of the stored vector and
    // the optional metadata, or NotFound when no such record exists.
    Result<RecordView> get(const RecordId& id) const;

    // Removes a Vector_Record by id, releasing its vector block and
    // decrementing the count. Returns NotFound and leaves the collection
    // unchanged when no such record exists.
    Status remove(const RecordId& id);

    // Builds an Index of the given type over every stored Vector_Record,
    // replacing any existing index. Parameters are validated against their
    // ranges before any index is constructed or replaced, so a rejected request
    // leaves any existing index unchanged:
    //   * HNSW: `m` >= 2, `ef_construction` >= 1
    //   * IVF : `nlist` in [1, record count]; 0 selects the auto-default
    //   * any other IndexType (e.g. None) -> InvalidArgument
    // An out-of-range value yields InvalidParameter naming the offending
    // parameter. An empty collection produces an empty index. Runs under the
    // exclusive lock.
    Status build_index(IndexType type, const IndexParams& params);

    // The type of the active index, or IndexType::None when none has been built.
    IndexType index_type() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return index_type_;
    }

    // Runs a k-nearest-neighbor query for `q`, returning up to `k` Neighbors
    // ordered by ascending computed distance under the collection's metric,
    // ties broken by ascending identifier for a deterministic ordering. The
    // request is validated before any work:
    //   * k < 1                       -> InvalidArgument
    //   * q.size() != dimensionality  -> DimensionMismatch
    //   * any non-finite component    -> InvalidValue
    // When `k` exceeds the record count, every record is returned in order; a
    // query against an empty collection returns an empty result set. When an
    // index is present and `k` is below the record count, the index narrows the
    // candidate set and the candidates are re-ranked by exact distance;
    // otherwise (or if the index under-delivers) an exact brute-force scan is
    // used. A stored vector whose distance is undefined under the metric (e.g.
    // cosine on a zero-norm vector) is treated as infinitely far and sorted to
    // the back rather than aborting the query. Runs under the shared lock.
    Result<std::vector<Neighbor>> query(span<const float> q, std::uint32_t k,
                                        const QueryParams& params) const;

    // A listing entry describing this collection.
    CollectionInfo info() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return CollectionInfo{config_.name, config_.dimensionality, config_.metric,
                              record_count_};
    }

    // The allocator tag attributed to this collection, used by the Engine to
    // release the collection's memory on deletion.
    CollectionTag tag() const { return tag_; }

    // Captures a consistent, point-in-time view of this collection for the
    // Persistence_Manager to serialize. Runs under the shared lock so the
    // records, metadata, and index observed are mutually consistent. Records are
    // returned sorted by identifier ascending (matching the on-disk
    // record-directory ordering) and the index, if any, is captured as its
    // canonical serialized bytes, so the view supports a byte-identical
    // round-trip.
    CollectionSnapshotData snapshot_data() const;

private:
    // Guards the record store and record count: reads take a shared lock,
    // mutations take an exclusive lock.
    mutable std::shared_mutex mutex_;

    // Immutable after construction.
    CollectionConfig config_;

    // Shared allocator backing this collection's vector storage. Non-owning;
    // the Engine owns it and outlives every collection.
    MemoryAllocator* allocator_;

    CollectionTag tag_;

    // Shared distance calculator used to construct indexes. Non-owning; owned
    // by the Engine and outliving every collection.
    const DistanceCalculator* calculator_;

    // Maps each RecordId to its slot (aligned vector block + optional metadata).
    // record_count_ is kept in step under the exclusive lock.
    std::map<RecordId, RecordSlot> records_;
    std::uint64_t                  record_count_ = 0;

    // The active Index (HNSW/IVF), null until built; index_type_ records which
    // type is active. Read under the shared lock; rebuilt or updated under the
    // exclusive lock.
    std::unique_ptr<Index> index_;
    IndexType              index_type_ = IndexType::None;

    // For a collection reconstructed from a Snapshot, the read-only mapping that
    // backs its records' vector storage. Outlives every record slot pointing
    // into it and is released on destruction. Null for in-memory collections
    // (whose vectors live in allocator blocks). Installed by the
    // PersistenceManager during load.
    std::unique_ptr<MmapRegion> mapped_region_;

    // Snapshot load reconstructs private state directly (mapping, mmap-backed
    // record store, index) rather than replaying public inserts, so vector data
    // is mapped instead of copied. Friendship keeps that off the public surface.
    friend class PersistenceManager;
};

// A lightweight, non-owning reference to a Collection owned by the Engine
// registry, returned by create_collection and get_collection. Valid only while
// the underlying Collection remains registered; using a handle after the
// collection is deleted is undefined.
class CollectionHandle {
public:
    CollectionHandle() = default;
    explicit CollectionHandle(Collection* collection) : collection_(collection) {}

    // True when the handle refers to a live Collection.
    bool valid() const { return collection_ != nullptr; }
    explicit operator bool() const { return valid(); }

    // Access to the referenced Collection. Precondition: valid().
    Collection* get() const { return collection_; }
    Collection* operator->() const { return collection_; }
    Collection& operator*() const { return *collection_; }

private:
    Collection* collection_ = nullptr;  // non-owning; owned by the Engine
};

}  // namespace vectorvault

#endif  // VECTORVAULT_COLLECTION_HPP
