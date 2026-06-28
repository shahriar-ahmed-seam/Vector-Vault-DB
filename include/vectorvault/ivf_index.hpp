#ifndef VECTORVAULT_IVF_INDEX_HPP
#define VECTORVAULT_IVF_INDEX_HPP

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "vectorvault/distance.hpp"
#include "vectorvault/index.hpp"
#include "vectorvault/types.hpp"

namespace vectorvault {

// Inverted File index with coarse k-means quantization. Partitions the vector
// space into `nlist` cells; each cell has a centroid, and every record is
// assigned to its nearest centroid, forming the cell's posting list. A query
// computes its distance to every centroid, probes the `nprobe` nearest cells,
// and scans only the records there, trading a little recall for far fewer
// comparisons.
//
// k-means clustering uses squared-L2 over the raw vectors as the coarse
// quantizer regardless of the collection's metric (standard IVF practice); the
// final ranking of probed candidates uses the collection's actual metric
// through the DistanceCalculator so reported distances match an exact
// computation.
//
// Construction parameter (IndexParams): `nlist`, in [1, record_count]; 0
// selects a heuristic default. Query parameter (QueryParams): `nprobe`, in
// [1, nlist]. Out-of-range rejection happens at the collection boundary; the
// index defensively clamps the effective partition and probe counts to what the
// current data supports (no more cells than points, no probing more cells than
// exist).
//
// Because k-means needs the full data set, the partitioning is (re)built lazily
// on the first search after any mutation: add/remove mark the index dirty and a
// subsequent search rebuilds the centroids and inverted lists. Deletions are
// tombstoned and skipped during clustering and result collection, so a deleted
// record is excluded from every subsequent query.
//
// The index owns a copy of every vector. It is not internally synchronised; the
// owning Collection serialises access through its readers-writer lock.
class IvfIndex : public Index {
public:
    // Constructs an empty index over `dimensionality`-component vectors,
    // computing final distances through `calculator` under `metric`.
    // `calculator` must outlive the index. The requested partition count and
    // k-means seed are read from `params` (nlist, seed).
    IvfIndex(std::uint32_t dimensionality, DistanceMetric metric,
             const DistanceCalculator& calculator, const IndexParams& params);

    Status add(const RecordId& id, span<const float> vec) override;
    Status remove(const RecordId& id) override;
    std::vector<Candidate> search(span<const float> q, std::uint32_t k,
                                  const QueryParams& params) const override;
    std::uint64_t size() const override { return live_count_; }
    IndexType type() const override { return IndexType::IVF; }
    std::vector<std::byte> serialize() const override;

private:
    // One stored record: identifier, vector copy, and tombstone flag.
    struct Entry {
        RecordId           id;
        std::vector<float> vector;
        bool               deleted = false;
    };

    // Final-ranking distance from `q` to `vec`. A metric error (e.g. cosine on a
    // zero-norm vector) maps to +infinity.
    float metric_distance(span<const float> q, span<const float> vec) const;

    // Rebuilds the centroids and inverted lists from the current live entries
    // via Lloyd's k-means, but only when the structure is marked dirty. Const
    // because it refreshes mutable caches behind a logically read-only search.
    void ensure_partitions_built() const;

    std::uint32_t       dimensionality_;
    DistanceMetric      metric_;
    const DistanceCalculator* calculator_;  // non-owning; outlives this index

    std::uint32_t requested_nlist_;  // nlist from IndexParams (0 = auto)
    std::uint64_t seed_;             // k-means initialisation seed

    std::vector<Entry>                          entries_;       // all records (by slot)
    std::unordered_map<RecordId, std::size_t>   id_to_index_;   // id -> entries_ slot
    std::uint64_t                               live_count_ = 0;

    // Lazily (re)built partitioning, refreshed by ensure_partitions_built().
    mutable bool                              dirty_ = true;
    mutable std::vector<std::vector<float>>   centroids_;   // [cell] -> centroid vector
    mutable std::vector<std::vector<std::uint32_t>> inverted_lists_;  // [cell] -> entry slots
};

}  // namespace vectorvault

#endif  // VECTORVAULT_IVF_INDEX_HPP
