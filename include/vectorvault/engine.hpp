#ifndef VECTORVAULT_ENGINE_HPP
#define VECTORVAULT_ENGINE_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "vectorvault/collection.hpp"
#include "vectorvault/distance.hpp"
#include "vectorvault/error.hpp"
#include "vectorvault/memory_allocator.hpp"
#include "vectorvault/persistence.hpp"
#include "vectorvault/types.hpp"

namespace vectorvault {

// Owns the registry of Collections and is the entry point for all operations.
// Owns the shared Memory_Allocator, assigns each Collection a unique allocator
// tag, and serializes registry mutations with an engine-level lock so creation,
// lookup, listing, and deletion are observed consistently.
class Engine {
public:
    // Constructs an Engine with a Memory_Allocator of the default capacity.
    Engine();

    // Constructs an Engine whose Memory_Allocator reserves the given capacity.
    explicit Engine(std::size_t allocator_capacity_bytes);

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    // Creates an empty Collection and returns a handle to it. Rejections create
    // no Collection and leave the registry unchanged:
    //   * name length not in [1, 255]             -> InvalidName
    //   * dimensionality not in [1, 65536]        -> InvalidDimensionality
    //   * metric not Euclidean/Cosine/DotProduct  -> InvalidMetric
    //   * name already exists                     -> NameConflict
    Result<CollectionHandle> create_collection(std::string_view name,
                                               std::uint32_t dimensionality,
                                               DistanceMetric metric);

    // Removes an existing Collection and releases its memory. Returns NotFound
    // if no such collection exists. If the (test-only) deletion fault-injection
    // hook signals a failure, the collection is left retrievable in its
    // pre-deletion state and DeletionFailed is returned.
    Status delete_collection(std::string_view name);

    // Returns one CollectionInfo per existing Collection, or an empty list when
    // none exist.
    std::vector<CollectionInfo> list_collections() const;

    // Returns a handle to the named Collection, or NotFound if it does not
    // exist.
    Result<CollectionHandle> get_collection(std::string_view name);

    // Saves the named Collection to `path` as a binary Snapshot. Returns
    // NotFound when no such collection exists, leaving the filesystem untouched.
    // The save is atomic: a failure leaves any pre-existing Snapshot at `path`
    // unmodified and returns WriteFailure.
    Status save_collection(std::string_view name,
                           const std::filesystem::path& path);

    // Loads a Snapshot from `path` and registers the reconstructed Collection
    // under its stored name. Vector data is read through a memory mapping. On a
    // load failure no collection is registered and the originating error
    // (SnapshotNotFound / UnsupportedVersion / Corruption) is returned. If a
    // collection with the snapshot's stored name already exists, the load is
    // rejected with NameConflict and the registry is left unchanged.
    Result<CollectionHandle> load_collection(const std::filesystem::path& path);

    // The shared Memory_Allocator backing every Collection. Exposed so tests can
    // observe allocation accounting (e.g. that a deleted collection's attributed
    // bytes return to 0).
    MemoryAllocator& allocator() { return allocator_; }
    const MemoryAllocator& allocator() const { return allocator_; }

    // Test-only deletion fault injector. When set, it is consulted at the start
    // of delete_collection; returning true aborts the deletion before any
    // change, leaving the collection retrievable and yielding DeletionFailed.
    // Unset in production, so deletions proceed normally.
    using DeleteFaultInjector = std::function<bool(std::string_view)>;
    void set_delete_fault_injector(DeleteFaultInjector injector) {
        std::lock_guard<std::mutex> lock(mutex_);
        delete_fault_injector_ = std::move(injector);
    }

private:
    mutable std::mutex mutex_;  // serializes registry mutations and lookups

    MemoryAllocator allocator_;

    // Shared distance calculator handed to every collection. Detects host SIMD
    // support once at construction and outlives every collection it backs.
    DistanceCalculator distance_calculator_;

    // Serializes collections to and from the on-disk Snapshot format. Stateless
    // for saving, so it is shared by every save_collection call.
    PersistenceManager persistence_;

    // Registry keyed by collection name. std::less<> (transparent comparator)
    // allows lookups directly from std::string_view without allocating a
    // std::string, and the ordered map yields a deterministic listing order.
    std::map<std::string, std::unique_ptr<Collection>, std::less<>> collections_;

    // Monotonic source of unique CollectionTags handed to each new collection.
    std::uint64_t next_tag_ = 1;

    // Test-only hook; unset in production (see set_delete_fault_injector).
    DeleteFaultInjector delete_fault_injector_;
};

}  // namespace vectorvault

#endif  // VECTORVAULT_ENGINE_HPP
