#ifndef VECTORVAULT_PERSISTENCE_HPP
#define VECTORVAULT_PERSISTENCE_HPP

#include <filesystem>
#include <memory>

#include "vectorvault/collection.hpp"
#include "vectorvault/error.hpp"
#include "vectorvault/snapshot_format.hpp"

namespace vectorvault {

// Serializes a Collection to the binary Snapshot format and reconstructs it.
// Save is atomic: the bytes are written to a temporary file in the target's
// directory, flushed durably, then atomically renamed over the target, so a
// failure before the rename leaves any pre-existing Snapshot untouched. The
// manager is stateless for saving and may be shared freely.
class PersistenceManager {
public:
    PersistenceManager() = default;

    // Serializes `collection` and writes it atomically to `path`. State is
    // captured under the collection's readers-writer lock and encoded
    // deterministically (records ascending by id, sorted metadata keys, zeroed
    // padding, canonical index region) so a save -> load -> save cycle is
    // byte-identical, then written via temp file + durable flush + atomic
    // rename. On any I/O failure the partial temp file is removed, any
    // pre-existing Snapshot at `path` is left unmodified, and WriteFailure is
    // returned.
    Status save(const Collection& collection, const std::filesystem::path& path);

    // Encodes a captured collection view into the in-memory Snapshot byte image
    // (header + name + record directory + padding + vector region + index
    // region, with the content CRC-64 filled in). Exposed so save and its tests
    // share one deterministic encoder. Pure function of its input.
    static std::vector<std::byte> encode(const CollectionSnapshotData& data);

    // Reconstructs a Collection from the Snapshot at `path`. Validates the file
    // in order before trusting any content: existence and minimum size, then
    // magic and header parse (SnapshotNotFound on failure), then the format
    // version (UnsupportedVersion), then the content checksum over the bytes
    // after the header (Corruption). Only after every check passes is the vector
    // region memory-mapped and the collection reconstructed, so its vector
    // components are read through the mapping rather than copied to the heap.
    //
    // The reconstructed Collection is bound to `allocator` (for subsequent
    // mutations), carries `tag` for allocator attribution, and computes
    // distances through `calculator`; these are owned by the Engine and outlive
    // the collection. Returns the collection on success, or an error Status
    // (SnapshotNotFound / UnsupportedVersion / Corruption) with no collection
    // created on failure.
    Result<std::unique_ptr<Collection>> load(const std::filesystem::path& path,
                                             MemoryAllocator& allocator,
                                             CollectionTag tag,
                                             const DistanceCalculator& calculator);
};

}  // namespace vectorvault

#endif  // VECTORVAULT_PERSISTENCE_HPP
