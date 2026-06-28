#ifndef VECTORVAULT_INDEX_HPP
#define VECTORVAULT_INDEX_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vectorvault/error.hpp"
#include "vectorvault/span.hpp"
#include "vectorvault/types.hpp"

namespace vectorvault {

// Construction-time parameters for both index types, flowing through
// Collection::build_index unchanged. Only the fields relevant to the chosen
// IndexType are consulted:
//   * HNSW reads `m` (max neighbours per node, >= 2) and `ef_construction`
//     (build-time exploration breadth, >= 1).
//   * IVF reads `nlist` (coarse k-means partitions, in [1, record count]);
//     a value of 0 requests the index's documented default (a heuristic derived
//     from the record count).
//
// Range validation against the record count happens in Collection::build_index;
// the index structures themselves defensively clamp to what the data can
// support so they never form more partitions than there are points.
struct IndexParams {
    std::uint32_t m               = 16;   // HNSW: max neighbours per node (>= 2)
    std::uint32_t ef_construction = 200;  // HNSW: build-time exploration breadth (>= 1)

    std::uint32_t nlist = 0;  // IVF: coarse partitions (1..record_count); 0 = auto

    // Seed for the deterministic randomness used during construction (HNSW level
    // assignment, IVF k-means initialisation) so a given input produces a
    // repeatable structure.
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
};

// Query-time tuning knobs for both index types; only the field relevant to the
// active IndexType is consulted:
//   * HNSW reads `ef_search`, the search-time exploration breadth (>= 1).
//   * IVF reads `nprobe`, the number of partitions probed per query
//     (in [1, nlist]).
struct QueryParams {
    std::uint32_t ef_search = 50;  // HNSW search breadth (>= 1)
    std::uint32_t nprobe    = 1;   // IVF partitions probed per query (1..nlist)
};

// One result of an index search: a record identifier and the distance computed
// for it under the collection's metric. The query path re-ranks Candidates and
// applies the ascending-identifier tie-break before surfacing them as Neighbors.
struct Candidate {
    RecordId id;
    float    distance;
};

// Common interface for the ANN index implementations (HnswIndex, IvfIndex). An
// Index owns its own copies of the vectors it has been given so it can compute
// distances during traversal, and it tracks membership so newly added records
// become searchable and removed records are excluded from every subsequent
// search.
//
// Distances are computed through a DistanceCalculator supplied at construction
// under a fixed DistanceMetric, matching the owning collection's metric so the
// index orders candidates the same way the exact distance does.
class Index {
public:
    virtual ~Index() = default;

    // Adds (or replaces) record `id` with vector `vec`, making it eligible to
    // appear in any search submitted after the call completes. Re-adding an
    // existing id replaces its stored vector; re-adding a previously removed id
    // revives it. Returns DimensionMismatch when vec.size() does not match the
    // index dimensionality and InvalidIdentifier when `id` is empty.
    virtual Status add(const RecordId& id, span<const float> vec) = 0;

    // Removes record `id` so it is excluded from every search submitted after
    // the call completes. Removal is logical (tombstoning): the record is marked
    // deleted and skipped when results are collected, while the surrounding
    // structure is preserved. Returns NotFound when no live record with that id
    // is present.
    virtual Status remove(const RecordId& id) = 0;

    // Returns up to `k` nearest Candidates to query vector `q` under the index's
    // metric, ordered by ascending computed distance. Tombstoned records are
    // never returned. The result is a best-effort (approximate) set; the caller
    // applies the final ascending-distance / ascending-identifier ordering.
    virtual std::vector<Candidate> search(span<const float> q, std::uint32_t k,
                                          const QueryParams& params) const = 0;

    // The number of live (non-tombstoned) records currently in the index.
    virtual std::uint64_t size() const = 0;

    // The concrete index type, so a serialized snapshot is self-describing and
    // the loader can dispatch to the right reconstruction path.
    virtual IndexType type() const = 0;

    // Serializes the index into a canonical, deterministic byte stream for the
    // Snapshot index region. The encoding uses a fixed node ordering (ascending
    // identifier) and fixed-width little-endian fields so a save -> load -> save
    // cycle is byte-identical. Tombstoned records are excluded so the persisted
    // membership matches the live records.
    virtual std::vector<std::byte> serialize() const = 0;
};

}  // namespace vectorvault

#endif  // VECTORVAULT_INDEX_HPP
