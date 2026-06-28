#ifndef VECTORVAULT_HNSW_INDEX_HPP
#define VECTORVAULT_HNSW_INDEX_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "vectorvault/distance.hpp"
#include "vectorvault/index.hpp"
#include "vectorvault/types.hpp"

namespace vectorvault {

// Hierarchical Navigable Small World graph. Each node is placed on layers 0..L,
// where its top layer L is drawn from an exponentially decaying distribution so
// higher layers are sparse "express lanes" and layer 0 holds every node. A
// search descends greedily through the upper layers to find a good entry point,
// then explores layer 0 with a breadth of `ef_search` to gather the nearest
// candidates.
//
// Construction parameters (IndexParams): `m` (max neighbours per node above
// layer 0; layer 0 keeps up to 2*m) and `ef_construction` (insertion
// exploration breadth). Query parameter (QueryParams): `ef_search`. All are
// clamped to safe minimums (m >= 2, ef_construction >= 1, ef_search >= 1) so the
// structure is always well-formed; out-of-range rejection happens at the
// collection boundary.
//
// Deletions are tombstoned: a removed node is marked deleted and still routes
// traversals (preserving connectivity) but is never returned in a result, so a
// deleted record is excluded from every subsequent query.
//
// The index owns a copy of every vector so it can compute distances through the
// supplied DistanceCalculator under a fixed metric. It is not internally
// synchronised; the owning Collection serialises access through its
// readers-writer lock.
class HnswIndex : public Index {
public:
    // Constructs an empty index over `dimensionality`-component vectors,
    // computing distances through `calculator` under `metric`. `calculator`
    // must outlive the index. Construction parameters (m, ef_construction, seed)
    // are read from `params` and clamped to their safe minimums.
    HnswIndex(std::uint32_t dimensionality, DistanceMetric metric,
              const DistanceCalculator& calculator, const IndexParams& params);

    Status add(const RecordId& id, span<const float> vec) override;
    Status remove(const RecordId& id) override;
    std::vector<Candidate> search(span<const float> q, std::uint32_t k,
                                  const QueryParams& params) const override;
    std::uint64_t size() const override { return live_count_; }
    IndexType type() const override { return IndexType::HNSW; }
    std::vector<std::byte> serialize() const override;

    // Reconstructs an index from a Snapshot's serialized index region.
    // serialize() persists the full graph (canonical node order, top layers,
    // neighbour adjacency) but NOT the vectors, which live in the Snapshot's
    // memory-mapped vector region; `vector_for` supplies each node's components
    // by identifier (a pointer into that mapping). The reconstructed structure
    // re-serializes byte-identically and contains exactly the persisted live
    // membership. Returns the rebuilt index on success; on a malformed or
    // inconsistent region it sets `out_ok` to false and returns nullptr.
    static std::unique_ptr<HnswIndex> deserialize(
        std::uint32_t dimensionality, DistanceMetric metric,
        const DistanceCalculator& calculator, const std::byte* data,
        std::size_t size,
        const std::function<const float*(const RecordId&)>& vector_for,
        bool& out_ok);

private:
    // One graph node: identifier, stored vector copy, top layer, tombstone flag,
    // and per-layer adjacency (neighbors[layer] holds internal node ids).
    struct Node {
        RecordId                                id;
        std::vector<float>                      vector;
        int                                     top_layer = 0;
        bool                                    deleted   = false;
        std::vector<std::vector<std::uint32_t>> neighbors;  // [layer] -> internal ids
    };

    // Distance from query `q` to node `node_id`. A metric error (e.g. cosine on
    // a zero-norm vector) maps to +infinity so the node sorts to the back rather
    // than aborting the traversal.
    float distance_to(span<const float> q, std::uint32_t node_id) const;

    // Distance between two stored nodes by internal id, used for the pairwise
    // candidate-to-candidate distances the neighbour heuristic needs. Shares the
    // +infinity-on-error convention with distance_to.
    float node_distance(std::uint32_t a, std::uint32_t b) const;

    // Malkov & Yashunin Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC): selects up to
    // `m` diverse neighbours from `candidates`, whose `.first` is each
    // candidate's distance to the base element. Iterating nearest-first, a
    // candidate `e` is accepted into the result only if it is closer to the base
    // element than to every already-accepted neighbour; otherwise it is
    // discarded. Candidates are ordered by ascending distance with ties broken by
    // ascending internal id, so the selection is deterministic.
    std::vector<std::uint32_t> select_neighbors_heuristic(
        std::vector<std::pair<float, std::uint32_t>> candidates,
        std::uint32_t m) const;

    // Draws a node's top layer from the exponential level distribution.
    int random_top_layer();

    // Maximum neighbour count permitted on `layer` (2*m on layer 0, m above).
    std::uint32_t max_neighbors_for_layer(int layer) const {
        return layer == 0 ? max_neighbors_layer0_ : max_neighbors_upper_;
    }

    // Greedy descent on a single upper layer: from `entry`, repeatedly hop to
    // the neighbour closest to `q` until none improves, returning the closest
    // node found. Used to find an entry point for the next lower layer.
    std::uint32_t greedy_descend(span<const float> q, std::uint32_t entry,
                                 int layer) const;

    // Best-first exploration of `layer` from `entry_points`, returning up to
    // `ef` nearest internal ids paired with their distance to `q`. Used during
    // insertion (ef = ef_construction) and search (ef = ef_search). All visited
    // nodes (including tombstoned ones) drive the traversal so connectivity is
    // preserved; callers filter tombstones from the result.
    std::vector<std::pair<float, std::uint32_t>> search_layer(
        span<const float> q, const std::vector<std::uint32_t>& entry_points,
        std::uint32_t ef, int layer) const;

    // Connects `node_id` to a diverse subset (Algorithm 4) of `candidates` on
    // `layer`, adding reciprocal edges and re-running the heuristic over any
    // neighbour whose adjacency list overflows the layer limit.
    void connect_node(std::uint32_t node_id,
                      std::vector<std::pair<float, std::uint32_t>> candidates,
                      int layer);

    std::uint32_t       dimensionality_;
    DistanceMetric      metric_;
    const DistanceCalculator* calculator_;  // non-owning; outlives this index

    std::uint32_t max_neighbors_upper_;   // m (layers above 0)
    std::uint32_t max_neighbors_layer0_;  // 2*m (layer 0)
    std::uint32_t ef_construction_;       // build-time exploration breadth
    double        level_scale_;           // mL = 1 / ln(m), for level sampling
    std::uint64_t rng_seed_;              // construction seed (persisted)

    std::vector<Node>                              nodes_;          // by internal id
    std::unordered_map<RecordId, std::uint32_t>    id_to_internal_;
    int                                            entry_point_ = -1;  // internal id or -1
    int                                            max_level_   = -1;  // highest occupied layer
    std::uint64_t                                  live_count_  = 0;   // non-tombstoned nodes

    mutable std::mt19937_64 rng_;  // deterministic level sampling (seeded from params)
};

}  // namespace vectorvault

#endif  // VECTORVAULT_HNSW_INDEX_HPP
