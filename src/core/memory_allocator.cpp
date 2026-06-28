#include "vectorvault/memory_allocator.hpp"

#include <new>      // ::operator new/delete with std::align_val_t
#include <string>
#include <utility>

namespace vectorvault {

std::size_t MemoryAllocator::round_up(std::size_t bytes) {
    // Round up to the next multiple of kAlignment. Zero stays zero.
    const std::size_t mask = kAlignment - 1;
    return (bytes + mask) & ~mask;
}

MemoryAllocator::MemoryAllocator(std::size_t capacity_bytes) {
    // Trim the capacity down to a whole number of 64-byte chunks so that every
    // offset produced by the bump/first-fit logic is a multiple of kAlignment.
    capacity_bytes_ = capacity_bytes - (capacity_bytes % kAlignment);

    if (capacity_bytes_ > 0) {
        // Over-aligned allocation: the returned base is aligned to kAlignment,
        // and because every block footprint is a multiple of kAlignment, every
        // block start (base_ + offset) is therefore 64-byte aligned.
        base_ = static_cast<std::byte*>(
            ::operator new(capacity_bytes_, std::align_val_t{kAlignment}));

        // The whole arena starts life as one free span.
        free_by_offset_.emplace(std::size_t{0}, capacity_bytes_);
    }
}

MemoryAllocator::~MemoryAllocator() {
    if (base_ != nullptr) {
        ::operator delete(base_, capacity_bytes_, std::align_val_t{kAlignment});
    }
}

Result<Block> MemoryAllocator::allocate(std::size_t bytes, CollectionTag tag) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Internal footprint is rounded up to keep every block 64-byte aligned. A
    // zero-byte request still occupies one aligned chunk so it yields a valid,
    // aligned, distinct address, while its accounted size remains 0.
    std::size_t needed = round_up(bytes);
    if (needed == 0) {
        needed = kAlignment;
    }

    // Address-ordered first fit: scan free spans (ordered by offset) for the
    // first one large enough to hold the request.
    auto fit = free_by_offset_.end();
    for (auto it = free_by_offset_.begin(); it != free_by_offset_.end(); ++it) {
        if (it->second >= needed) {
            fit = it;
            break;
        }
    }

    // Exhaustion: no span can satisfy the request. Leave existing blocks and
    // the reported totals unchanged, and report an allocation failure.
    if (fit == free_by_offset_.end()) {
        return Result<Block>::error(
            ErrorCategory::AllocationFailure,
            "memory allocator exhausted: requested " + std::to_string(bytes) +
                " bytes (" + std::to_string(needed) + " after alignment)");
    }

    const std::size_t offset    = fit->first;
    const std::size_t span_size = fit->second;

    // Consume the span. Keep any remainder as a smaller free span so the freed
    // space stays reusable.
    free_by_offset_.erase(fit);
    if (span_size > needed) {
        free_by_offset_.emplace(offset + needed, span_size - needed);
    }

    // Record the live allocation and update accounting using the *requested*
    // byte count so stats reflect granted-minus-released request sizes.
    live_.emplace(offset, AllocRecord{offset, needed, bytes, tag});
    total_allocated_bytes_ += bytes;
    ++live_block_count_;
    tag_bytes_[tag]  += bytes;
    tag_blocks_[tag] += 1;

    Block block;
    block.data = base_ + offset;  // 64-byte aligned
    block.size = bytes;           // expose the requested size to the caller
    return Result<Block>{block};
}

// free_span coalesces with adjacent free spans. Caller holds mutex_.
void MemoryAllocator::free_span(std::size_t offset, std::size_t size) {
    if (size == 0) {
        return;
    }

    // Insert the span, then merge with an immediately preceding and/or
    // following free span to limit fragmentation.
    auto it = free_by_offset_.emplace(offset, size).first;

    // Coalesce with the following span if it is contiguous.
    auto next = std::next(it);
    if (next != free_by_offset_.end() && it->first + it->second == next->first) {
        it->second += next->second;
        free_by_offset_.erase(next);
    }

    // Coalesce with the preceding span if it is contiguous.
    if (it != free_by_offset_.begin()) {
        auto prev = std::prev(it);
        if (prev->first + prev->second == it->first) {
            prev->second += it->second;
            free_by_offset_.erase(it);
        }
    }
}

void MemoryAllocator::release(Block block) {
    if (block.data == nullptr || base_ == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Recover the arena offset from the block's address. Ignore blocks that do
    // not belong to this arena.
    auto* ptr = static_cast<std::byte*>(block.data);
    if (ptr < base_ || ptr >= base_ + capacity_bytes_) {
        return;
    }
    const std::size_t offset = static_cast<std::size_t>(ptr - base_);

    auto live_it = live_.find(offset);
    if (live_it == live_.end()) {
        return;  // not a live allocation (double release or foreign pointer)
    }

    const AllocRecord rec = live_it->second;

    // Roll back accounting using the originally requested byte count.
    total_allocated_bytes_ -= rec.requested_bytes;
    --live_block_count_;

    auto tb = tag_bytes_.find(rec.tag);
    if (tb != tag_bytes_.end()) {
        tb->second -= rec.requested_bytes;
        if (tb->second == 0) {
            tag_bytes_.erase(tb);
        }
    }
    auto tc = tag_blocks_.find(rec.tag);
    if (tc != tag_blocks_.end()) {
        tc->second -= 1;
        if (tc->second == 0) {
            tag_blocks_.erase(tc);
        }
    }

    live_.erase(live_it);
    free_span(rec.offset, rec.rounded_bytes);
}

void MemoryAllocator::release_all(CollectionTag tag) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = live_.begin(); it != live_.end();) {
        if (it->second.tag == tag) {
            const AllocRecord rec = it->second;
            total_allocated_bytes_ -= rec.requested_bytes;
            --live_block_count_;
            free_span(rec.offset, rec.rounded_bytes);
            it = live_.erase(it);
        } else {
            ++it;
        }
    }

    // The tag's attributed bytes return to 0.
    tag_bytes_.erase(tag);
    tag_blocks_.erase(tag);
}

AllocStats MemoryAllocator::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return AllocStats{total_allocated_bytes_, live_block_count_};
}

AllocStats MemoryAllocator::stats(CollectionTag tag) const {
    std::lock_guard<std::mutex> lock(mutex_);

    AllocStats out;
    auto tb = tag_bytes_.find(tag);
    if (tb != tag_bytes_.end()) {
        out.total_allocated_bytes = tb->second;
    }
    auto tc = tag_blocks_.find(tag);
    if (tc != tag_blocks_.end()) {
        out.live_block_count = tc->second;
    }
    return out;
}

}  // namespace vectorvault
