// MemoryAllocator tests: 64-byte block alignment, allocation accounting, and
// allocation-failure isolation.

#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>

#include "vectorvault/memory_allocator.hpp"

#include <cstdint>
#include <cstring>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

using namespace vectorvault;

namespace {

// True when `p` starts on a `alignment`-byte boundary.
bool is_aligned(const void* p, std::size_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(p) % alignment) == 0;
}

}  // namespace

// Feature: vector-vault-db, Property 15: Allocator returns 64-byte aligned vector blocks
// Validates: Requirements 7.2
TEST_CASE("Property 15: allocator returns 64-byte aligned blocks",
          "[allocator][property]") {
    const bool ok = rc::check(
        "every granted block starts on a 64-byte boundary",
        [](const std::vector<std::uint16_t>& request_sizes) {
            // Default 64 MiB arena: large enough that small requests succeed.
            MemoryAllocator alloc;
            const CollectionTag tag{1};

            for (std::uint16_t s : request_sizes) {
                const std::size_t bytes = static_cast<std::size_t>(s);
                auto result = alloc.allocate(bytes, tag);
                // Only successful allocations carry an alignment guarantee; an
                // exhausted arena is a valid outcome we skip.
                if (result.is_ok()) {
                    const Block& block = result.value();
                    RC_ASSERT(is_aligned(block.data, MemoryAllocator::kAlignment));
                    RC_ASSERT(block.size == bytes);
                }
            }
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 16: Allocator accounting equals allocated minus released
// Validates: Requirements 7.1, 7.3, 7.4
TEST_CASE("Property 16: allocator accounting equals allocated minus released",
          "[allocator][property]") {
    const bool ok = rc::check(
        "reported totals equal granted minus released; tag bytes return to 0 "
        "after release_all",
        [](const std::vector<std::tuple<bool, std::uint8_t, std::uint16_t>>& ops) {
            MemoryAllocator alloc;

            // Model: live blocks per tag and the requested bytes attributed to
            // each. The model is the source of truth checked against stats().
            std::map<std::uint64_t, std::vector<std::pair<Block, std::size_t>>> live;
            std::map<std::uint64_t, std::size_t> expected_tag_bytes;
            std::size_t expected_total = 0;

            // Recompute the per-tag and global expectation and assert agreement.
            auto check_invariant = [&]() {
                std::size_t sum = 0;
                for (const auto& [tag_value, blocks] : live) {
                    std::size_t tag_sum = 0;
                    for (const auto& entry : blocks) {
                        tag_sum += entry.second;
                    }
                    RC_ASSERT(alloc.stats(CollectionTag{tag_value})
                                  .total_allocated_bytes == tag_sum);
                    sum += tag_sum;
                }
                RC_ASSERT(alloc.stats().total_allocated_bytes == sum);
            };

            for (const auto& op : ops) {
                const bool do_alloc = std::get<0>(op);
                const std::uint64_t tag_value =
                    static_cast<std::uint64_t>(std::get<1>(op) % 4);  // few tags
                const std::size_t bytes =
                    static_cast<std::size_t>(std::get<2>(op));
                const CollectionTag tag{tag_value};

                if (do_alloc) {
                    auto result = alloc.allocate(bytes, tag);
                    if (result.is_ok()) {
                        live[tag_value].emplace_back(result.value(), bytes);
                        expected_total += bytes;
                        expected_tag_bytes[tag_value] += bytes;
                    }
                    // A failed allocation makes no change to the totals, so the
                    // model is left untouched and the invariant still holds.
                } else {
                    // Release the most recently allocated live block for this tag.
                    auto it = live.find(tag_value);
                    if (it != live.end() && !it->second.empty()) {
                        const auto entry = it->second.back();
                        it->second.pop_back();
                        alloc.release(entry.first);
                        expected_total -= entry.second;
                        expected_tag_bytes[tag_value] -= entry.second;
                    }
                }

                // total == granted - released at the time of the query.
                RC_ASSERT(alloc.stats().total_allocated_bytes == expected_total);
                check_invariant();
            }

            // After releasing each tag, its attributed bytes and block count
            // return to 0.
            for (const auto& [tag_value, blocks] : live) {
                (void)blocks;
                alloc.release_all(CollectionTag{tag_value});
                const AllocStats tag_stats = alloc.stats(CollectionTag{tag_value});
                RC_ASSERT(tag_stats.total_allocated_bytes == 0);
                RC_ASSERT(tag_stats.live_block_count == 0);
            }

            // With every tag released, the global totals are back to 0.
            RC_ASSERT(alloc.stats().total_allocated_bytes == 0);
            RC_ASSERT(alloc.stats().live_block_count == 0);
        });
    REQUIRE(ok);
}

// Unit test: a failed allocation leaves prior blocks and totals intact.
// Validates: Requirements 7.5
TEST_CASE("allocation failure leaves prior blocks and totals unchanged",
          "[allocator][unit]") {
    // A tiny arena (4 x 64-byte chunks) so exhaustion is easy to force.
    MemoryAllocator alloc(256);
    const CollectionTag tag{7};

    // Fill most of the arena: 64 + 128 = 192 bytes, leaving 64.
    auto first = alloc.allocate(64, tag);
    REQUIRE(first.is_ok());
    auto second = alloc.allocate(128, tag);
    REQUIRE(second.is_ok());

    void* const first_data = first.value().data;
    void* const second_data = second.value().data;

    const AllocStats before = alloc.stats();
    const AllocStats before_tag = alloc.stats(tag);
    REQUIRE(before.total_allocated_bytes == 192);
    REQUIRE(before.live_block_count == 2);

    // Request more than the arena can ever provide -> allocation failure.
    auto failed = alloc.allocate(1024, tag);
    REQUIRE(failed.is_error());
    REQUIRE(failed.category() == ErrorCategory::AllocationFailure);

    // Reported totals are unchanged by the failed request.
    const AllocStats after = alloc.stats();
    REQUIRE(after.total_allocated_bytes == before.total_allocated_bytes);
    REQUIRE(after.live_block_count == before.live_block_count);
    REQUIRE(alloc.stats(tag).total_allocated_bytes ==
            before_tag.total_allocated_bytes);
    REQUIRE(alloc.stats(tag).live_block_count == before_tag.live_block_count);

    // Prior blocks are unchanged: same addresses, still usable storage.
    REQUIRE(first.value().data == first_data);
    REQUIRE(second.value().data == second_data);
    std::memset(first_data, 0xAB, 64);
    std::memset(second_data, 0xCD, 128);

    // A failure that left totals intact must still allow clean teardown.
    alloc.release(first.value());
    alloc.release(second.value());
    REQUIRE(alloc.stats().total_allocated_bytes == 0);
    REQUIRE(alloc.stats().live_block_count == 0);
}
