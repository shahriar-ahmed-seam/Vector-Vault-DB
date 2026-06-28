#ifndef VECTORVAULT_MEMORY_ALLOCATOR_HPP
#define VECTORVAULT_MEMORY_ALLOCATOR_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>

#include "vectorvault/error.hpp"
#include "vectorvault/types.hpp"

namespace vectorvault {

// Opaque per-collection identifier used to attribute allocator bytes to the
// collection that requested them. release_all(tag) frees every block carrying
// the tag so its attributed bytes return to 0 when a collection is deleted. A
// strong type (rather than a bare integer) keeps tags from being confused with
// sizes or other counters at call sites.
struct CollectionTag {
    std::uint64_t value = 0;

    constexpr CollectionTag() = default;
    constexpr explicit CollectionTag(std::uint64_t v) : value(v) {}

    friend constexpr bool operator==(CollectionTag a, CollectionTag b) {
        return a.value == b.value;
    }
    friend constexpr bool operator!=(CollectionTag a, CollectionTag b) {
        return a.value != b.value;
    }
    friend constexpr bool operator<(CollectionTag a, CollectionTag b) {
        return a.value < b.value;
    }
};

// Snapshot of allocator accounting at the time of the stats() call.
// `total_allocated_bytes` is granted bytes minus released bytes and is always
// non-negative.
struct AllocStats {
    std::size_t total_allocated_bytes = 0;  // granted - released
    std::size_t live_block_count = 0;       // number of outstanding blocks
};

// Arena/pool allocator that hands out 64-byte aligned blocks carved from a
// single pre-reserved, contiguous, 64-byte aligned arena. Released blocks go to
// an address-ordered free list with coalescing so freed space is reused.
//
// Accounting is reported in terms of the bytes callers requested, independent of
// the internal alignment rounding, so the reported total equals
// granted-minus-released request sizes.
//
// Internally thread-safe: all free-list and accounting operations are guarded by
// a single lightweight mutex.
class MemoryAllocator {
public:
    // Every returned block starts on a 64-byte boundary.
    static constexpr std::size_t kAlignment = 64;

    // Default arena capacity used when none is supplied.
    static constexpr std::size_t kDefaultCapacityBytes =
        static_cast<std::size_t>(64) * 1024 * 1024;

    // Reserves a contiguous arena of (capacity_bytes rounded down to a multiple
    // of kAlignment). The arena base is 64-byte aligned.
    explicit MemoryAllocator(std::size_t capacity_bytes = kDefaultCapacityBytes);
    ~MemoryAllocator();

    // The allocator owns a raw buffer and a mutex; copying/moving it would
    // invalidate outstanding blocks, so both are disabled.
    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;
    MemoryAllocator(MemoryAllocator&&) = delete;
    MemoryAllocator& operator=(MemoryAllocator&&) = delete;

    // Allocates a 64-byte aligned block of at least `bytes` attributed to `tag`.
    // On success the returned Block's `size` is the requested `bytes`. If the
    // arena cannot satisfy the request, returns AllocationFailure and leaves
    // every existing block and all reported totals unchanged.
    Result<Block> allocate(std::size_t bytes, CollectionTag tag);

    // Releases a block previously returned by allocate(): its bytes are removed
    // from the running totals and tag attribution, and the underlying space
    // returns to the free list for reuse. A null or foreign block is ignored.
    void release(Block block);

    // Releases every block currently attributed to `tag`, so the bytes
    // attributed to it return to 0.
    void release_all(CollectionTag tag);

    // Total bytes currently allocated across all tags.
    AllocStats stats() const;

    // Bytes currently allocated for a single tag.
    AllocStats stats(CollectionTag tag) const;

    // Usable arena capacity in bytes (multiple of kAlignment).
    std::size_t capacity_bytes() const { return capacity_bytes_; }

private:
    // Bookkeeping for one outstanding allocation.
    struct AllocRecord {
        std::size_t   offset;          // byte offset of the block within the arena
        std::size_t   rounded_bytes;   // arena footprint (multiple of kAlignment)
        std::size_t   requested_bytes; // bytes the caller asked for (accounted)
        CollectionTag tag;             // owning collection
    };

    // Rounds `bytes` up to the next multiple of kAlignment (0 stays 0).
    static std::size_t round_up(std::size_t bytes);

    // Returns [offset, offset+size) to the free list, merging it with any
    // immediately adjacent free spans. Caller must hold mutex_.
    void free_span(std::size_t offset, std::size_t size);

    mutable std::mutex mutex_;

    std::byte*  base_ = nullptr;    // 64-byte aligned arena base address
    std::size_t capacity_bytes_ = 0;  // usable arena size (multiple of kAlignment)

    // Free spans keyed by offset (ordered) to enable coalescing and an
    // address-ordered first-fit search: offset -> span size in bytes.
    std::map<std::size_t, std::size_t> free_by_offset_;

    // Outstanding allocations keyed by their arena offset.
    std::unordered_map<std::size_t, AllocRecord> live_;

    // Accounting in terms of requested bytes.
    std::size_t total_allocated_bytes_ = 0;
    std::size_t live_block_count_ = 0;

    // Per-tag attribution: tag -> requested bytes / block count.
    std::map<CollectionTag, std::size_t> tag_bytes_;
    std::map<CollectionTag, std::size_t> tag_blocks_;
};

}  // namespace vectorvault

#endif  // VECTORVAULT_MEMORY_ALLOCATOR_HPP
