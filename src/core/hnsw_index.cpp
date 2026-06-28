#include "vectorvault/hnsw_index.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vectorvault/snapshot_format.hpp"

namespace vectorvault {

namespace {

// Sentinel distance returned for vectors that cannot be scored.
constexpr double kInfinityF = std::numeric_limits<float>::infinity();

}  // namespace

HnswIndex::HnswIndex(std::uint32_t dimensionality, DistanceMetric metric,
                     const DistanceCalculator& calculator, const IndexParams& params)
    : dimensionality_(dimensionality),
      metric_(metric),
      calculator_(&calculator),
      // Clamp m to its documented minimum of 2 so the structure stays
      // well-formed even if an unvalidated value reaches the index.
      max_neighbors_upper_(params.m < 2 ? 2u : params.m),
      max_neighbors_layer0_(max_neighbors_upper_ * 2u),
      // ef_construction must explore at least one candidate.
      ef_construction_(params.ef_construction < 1 ? 1u : params.ef_construction),
      // Level scale mL = 1 / ln(m); a higher m yields shallower graphs.
      level_scale_(1.0 / std::log(static_cast<double>(max_neighbors_upper_))),
      rng_seed_(params.seed),
      rng_(params.seed) {}

float HnswIndex::distance_to(span<const float> q, std::uint32_t node_id) const {
    const std::vector<float>& v = nodes_[node_id].vector;
    Result<float> r = calculator_->distance(metric_, q, span<const float>(v));
    return r.is_ok() ? r.value() : kInfinityF;
}

float HnswIndex::node_distance(std::uint32_t a, std::uint32_t b) const {
    const std::vector<float>& va = nodes_[a].vector;
    return distance_to(span<const float>(va), b);
}

std::vector<std::uint32_t> HnswIndex::select_neighbors_heuristic(
    std::vector<std::pair<float, std::uint32_t>> candidates,
    std::uint32_t m) const {
    // Order candidates nearest-first by distance to the base element, breaking
    // ties on ascending internal id so the selection is deterministic.
    std::sort(candidates.begin(), candidates.end(),
              [](const std::pair<float, std::uint32_t>& x,
                 const std::pair<float, std::uint32_t>& y) {
                  if (x.first != y.first) {
                      return x.first < y.first;
                  }
                  return x.second < y.second;
              });

    std::vector<std::uint32_t> selected;
    selected.reserve(m);
    for (const auto& cand : candidates) {
        if (selected.size() >= m) {
            break;
        }
        const float dist_to_base = cand.first;
        const std::uint32_t e = cand.second;

        // Accept e only if it is closer to the base element than to every
        // already-accepted neighbour; otherwise discard it as redundant.
        bool keep = true;
        for (std::uint32_t r : selected) {
            if (node_distance(e, r) <= dist_to_base) {
                keep = false;
                break;
            }
        }
        if (keep) {
            selected.push_back(e);
        }
    }
    return selected;
}

int HnswIndex::random_top_layer() {
    // Standard HNSW level assignment: floor(-ln(U) * mL), U ~ Uniform(0, 1].
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng_);
    if (u <= 0.0) {
        u = std::numeric_limits<double>::min();
    }
    const double level = -std::log(u) * level_scale_;
    return static_cast<int>(level);
}

std::uint32_t HnswIndex::greedy_descend(span<const float> q, std::uint32_t entry,
                                        int layer) const {
    std::uint32_t current = entry;
    float current_dist = distance_to(q, current);

    bool improved = true;
    while (improved) {
        improved = false;
        const std::vector<std::uint32_t>& neighbors = nodes_[current].neighbors[layer];
        for (std::uint32_t neighbor : neighbors) {
            const float d = distance_to(q, neighbor);
            if (d < current_dist) {
                current_dist = d;
                current = neighbor;
                improved = true;
            }
        }
    }
    return current;
}

std::vector<std::pair<float, std::uint32_t>> HnswIndex::search_layer(
    span<const float> q, const std::vector<std::uint32_t>& entry_points,
    std::uint32_t ef, int layer) const {
    // Min-heap of (distance, id): the next closest unexplored candidate.
    using Item = std::pair<float, std::uint32_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> candidates;
    // Max-heap of (distance, id): the current ef nearest found (farthest on top).
    std::priority_queue<Item, std::vector<Item>, std::less<Item>> result;

    std::unordered_set<std::uint32_t> visited;
    visited.reserve(entry_points.size() * 4 + 16);

    for (std::uint32_t ep : entry_points) {
        const float d = distance_to(q, ep);
        candidates.emplace(d, ep);
        result.emplace(d, ep);
        visited.insert(ep);
    }
    while (result.size() > ef) {
        result.pop();
    }

    while (!candidates.empty()) {
        const Item nearest = candidates.top();
        candidates.pop();

        // Stop when the closest remaining candidate is farther than the worst
        // result we already hold and the result set is already full.
        if (!result.empty() && nearest.first > result.top().first &&
            result.size() >= ef) {
            break;
        }

        const std::vector<std::uint32_t>& neighbors = nodes_[nearest.second].neighbors[layer];
        for (std::uint32_t neighbor : neighbors) {
            if (visited.count(neighbor) != 0) {
                continue;
            }
            visited.insert(neighbor);
            const float d = distance_to(q, neighbor);
            if (result.size() < ef || d < result.top().first) {
                candidates.emplace(d, neighbor);
                result.emplace(d, neighbor);
                if (result.size() > ef) {
                    result.pop();
                }
            }
        }
    }

    std::vector<Item> out;
    out.reserve(result.size());
    while (!result.empty()) {
        out.push_back(result.top());
        result.pop();
    }
    // `out` is currently farthest-first; reverse to nearest-first.
    std::reverse(out.begin(), out.end());
    return out;
}

void HnswIndex::connect_node(std::uint32_t node_id,
                             std::vector<std::pair<float, std::uint32_t>> candidates,
                             int layer) {
    const std::uint32_t limit = max_neighbors_for_layer(layer);

    // Candidates carry their distance to the new node (the base element); pick a
    // diverse subset via the selection heuristic rather than the closest `limit`.
    const std::vector<std::uint32_t> selected =
        select_neighbors_heuristic(std::move(candidates), limit);

    std::vector<std::uint32_t>& own = nodes_[node_id].neighbors[layer];
    for (std::uint32_t other : selected) {
        if (other == node_id) {
            continue;
        }
        own.push_back(other);

        // Add the reciprocal edge. If it overflows the neighbour's list, re-run
        // the heuristic over that list using the neighbour's own vector as the
        // base element, keeping a diverse subset instead of the closest `limit`.
        std::vector<std::uint32_t>& other_list = nodes_[other].neighbors[layer];
        other_list.push_back(node_id);
        if (other_list.size() > limit) {
            std::vector<std::pair<float, std::uint32_t>> other_candidates;
            other_candidates.reserve(other_list.size());
            for (std::uint32_t nb : other_list) {
                other_candidates.emplace_back(node_distance(other, nb), nb);
            }
            other_list = select_neighbors_heuristic(std::move(other_candidates),
                                                    limit);
        }
    }
}

Status HnswIndex::add(const RecordId& id, span<const float> vec) {
    if (id.empty()) {
        return Status::error(ErrorCategory::InvalidIdentifier,
                             "index add requires a non-empty identifier");
    }
    if (vec.size() != dimensionality_) {
        return Status::error(
            ErrorCategory::DimensionMismatch,
            "index add expected a vector of dimensionality " +
                std::to_string(dimensionality_) + " but got " +
                std::to_string(vec.size()));
    }

    // Re-adding an existing id replaces its stored vector and revives it if it
    // had been tombstoned. Existing graph edges are retained, which keeps the
    // structure connected for an approximate index.
    auto it = id_to_internal_.find(id);
    if (it != id_to_internal_.end()) {
        Node& node = nodes_[it->second];
        node.vector.assign(vec.begin(), vec.end());
        if (node.deleted) {
            node.deleted = false;
            ++live_count_;
        }
        return Status::ok();
    }

    const int node_level = random_top_layer();
    const std::uint32_t node_id = static_cast<std::uint32_t>(nodes_.size());

    Node node;
    node.id = id;
    node.vector.assign(vec.begin(), vec.end());
    node.top_layer = node_level;
    node.neighbors.resize(static_cast<std::size_t>(node_level) + 1);
    nodes_.push_back(std::move(node));
    id_to_internal_.emplace(id, node_id);
    ++live_count_;

    span<const float> q(nodes_[node_id].vector);

    // First node ever: it becomes the entry point and there is nothing to link.
    if (entry_point_ < 0) {
        entry_point_ = static_cast<int>(node_id);
        max_level_ = node_level;
        return Status::ok();
    }

    std::uint32_t current = static_cast<std::uint32_t>(entry_point_);

    // Descend greedily through the layers above the new node's top layer.
    for (int layer = max_level_; layer > node_level; --layer) {
        current = greedy_descend(q, current, layer);
    }

    // From the new node's top layer down to 0, explore, select, and connect.
    const int start_layer = std::min(node_level, max_level_);
    for (int layer = start_layer; layer >= 0; --layer) {
        std::vector<std::pair<float, std::uint32_t>> found =
            search_layer(q, {current}, ef_construction_, layer);
        connect_node(node_id, found, layer);
        if (!found.empty()) {
            current = found.front().second;  // nearest, entry for the next layer
        }
    }

    // A taller node becomes the new entry point at the new maximum level.
    if (node_level > max_level_) {
        entry_point_ = static_cast<int>(node_id);
        max_level_ = node_level;
    }
    return Status::ok();
}

Status HnswIndex::remove(const RecordId& id) {
    auto it = id_to_internal_.find(id);
    if (it == id_to_internal_.end()) {
        return Status::error(ErrorCategory::NotFound,
                             "index has no record with id '" + id + "'");
    }
    Node& node = nodes_[it->second];
    if (node.deleted) {
        return Status::error(ErrorCategory::NotFound,
                             "index has no live record with id '" + id + "'");
    }
    // Tombstone: keep the node for routing but exclude it from results.
    node.deleted = true;
    --live_count_;
    return Status::ok();
}

std::vector<Candidate> HnswIndex::search(span<const float> q, std::uint32_t k,
                                         const QueryParams& params) const {
    if (entry_point_ < 0 || k == 0 || live_count_ == 0) {
        return {};
    }

    std::uint32_t current = static_cast<std::uint32_t>(entry_point_);
    for (int layer = max_level_; layer > 0; --layer) {
        current = greedy_descend(q, current, layer);
    }

    // Explore layer 0 with a breadth of at least ef_search and at least k so a
    // small ef never starves the result of k live candidates.
    std::uint32_t ef = std::max<std::uint32_t>(params.ef_search < 1 ? 1u : params.ef_search, k);
    std::vector<std::pair<float, std::uint32_t>> found = search_layer(q, {current}, ef, 0);

    std::vector<Candidate> out;
    out.reserve(found.size());
    for (const auto& item : found) {
        const Node& node = nodes_[item.second];
        if (node.deleted) {
            continue;  // tombstoned records are never returned
        }
        out.push_back(Candidate{node.id, item.first});
    }

    // Order by ascending distance, ties broken by ascending identifier so the
    // candidate ordering is deterministic.
    std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
        if (a.distance != b.distance) {
            return a.distance < b.distance;
        }
        return a.id < b.id;
    });
    if (out.size() > k) {
        out.resize(k);
    }
    return out;
}

std::vector<std::byte> HnswIndex::serialize() const {
    // Canonical node ordering for a byte-identical round-trip: keep only live
    // (non-tombstoned) nodes so the persisted membership matches the live
    // records, and order them by ascending identifier. The position in this
    // sorted order is the node's canonical id; neighbour references are remapped
    // to these canonical ids, dropping edges to tombstoned nodes (their
    // endpoints are gone from the persisted graph).
    std::vector<std::uint32_t> live;  // internal ids of live nodes
    live.reserve(nodes_.size());
    for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
        if (!nodes_[i].deleted) {
            live.push_back(i);
        }
    }
    std::sort(live.begin(), live.end(), [&](std::uint32_t a, std::uint32_t b) {
        return nodes_[a].id < nodes_[b].id;
    });

    // internal id -> canonical id (defined only for live nodes).
    std::unordered_map<std::uint32_t, std::uint32_t> canonical;
    canonical.reserve(live.size() * 2 + 1);
    for (std::uint32_t c = 0; c < live.size(); ++c) {
        canonical.emplace(live[c], c);
    }

    // Derive the entry point and max level from the live set so they remain
    // valid even when the original entry point was tombstoned. The entry point
    // is the live node with the highest top layer, ties broken by the smallest
    // canonical id (i.e. smallest identifier) for determinism.
    constexpr std::uint32_t kNone = 0xFFFFFFFFu;
    std::uint32_t entry_canonical = kNone;
    std::int32_t  max_level = -1;
    for (std::uint32_t c = 0; c < live.size(); ++c) {
        const int top = nodes_[live[c]].top_layer;
        if (top > max_level) {
            max_level = top;
            entry_canonical = c;
        }
    }

    snapshot::ByteWriter w;
    w.put_u32(max_neighbors_upper_);   // m
    w.put_u32(ef_construction_);       // ef_construction
    w.put_u64(static_cast<std::uint64_t>(rng_seed_));  // construction seed
    w.put_u32(static_cast<std::uint32_t>(live.size()));
    w.put_u32(static_cast<std::uint32_t>(max_level));  // -1 -> 0xFFFFFFFF
    w.put_u32(entry_canonical);

    for (std::uint32_t c = 0; c < live.size(); ++c) {
        const Node& node = nodes_[live[c]];
        w.put_string(node.id);
        w.put_u32(static_cast<std::uint32_t>(node.top_layer));
        // Layers 0..top_layer, each with its (canonical, live-only, sorted,
        // deduplicated) neighbour list.
        for (int layer = 0; layer <= node.top_layer; ++layer) {
            std::vector<std::uint32_t> remapped;
            if (layer < static_cast<int>(node.neighbors.size())) {
                remapped.reserve(node.neighbors[layer].size());
                for (std::uint32_t nb : node.neighbors[layer]) {
                    auto it = canonical.find(nb);
                    if (it != canonical.end()) {
                        remapped.push_back(it->second);
                    }
                }
            }
            std::sort(remapped.begin(), remapped.end());
            remapped.erase(std::unique(remapped.begin(), remapped.end()),
                           remapped.end());
            w.put_u32(static_cast<std::uint32_t>(remapped.size()));
            for (std::uint32_t cid : remapped) {
                w.put_u32(cid);
            }
        }
    }

    return std::move(w.bytes());
}

std::unique_ptr<HnswIndex> HnswIndex::deserialize(
    std::uint32_t dimensionality, DistanceMetric metric,
    const DistanceCalculator& calculator, const std::byte* data,
    std::size_t size,
    const std::function<const float*(const RecordId&)>& vector_for,
    bool& out_ok) {
    out_ok = false;
    snapshot::ByteReader r(data, size);

    // Header mirrors serialize(): m, ef_construction, seed, live count, max
    // level, and the entry-point's canonical id.
    const std::uint32_t m               = r.get_u32();
    const std::uint32_t ef_construction = r.get_u32();
    const std::uint64_t seed            = r.get_u64();
    const std::uint32_t count           = r.get_u32();
    const std::uint32_t stored_max_lvl  = r.get_u32();  // 0xFFFFFFFF when empty
    const std::uint32_t entry_canonical = r.get_u32();  // 0xFFFFFFFF when none
    if (!r.ok()) {
        return nullptr;
    }

    // Rebuild with the persisted construction parameters so a subsequent
    // re-serialize emits identical m / ef_construction / seed fields.
    IndexParams params;
    params.m               = m;
    params.ef_construction = ef_construction;
    params.seed            = seed;
    auto index = std::make_unique<HnswIndex>(dimensionality, metric, calculator,
                                             params);

    // The canonical id written by serialize() is the position in ascending-id
    // order, so reading nodes in sequence makes canonical id == internal id.
    index->nodes_.reserve(count);
    index->id_to_internal_.reserve(count * 2 + 1);
    for (std::uint32_t c = 0; c < count; ++c) {
        Node node;
        node.id        = r.get_string();
        node.top_layer = static_cast<int>(r.get_u32());
        if (!r.ok() || node.top_layer < 0) {
            return nullptr;
        }

        // A node's vector is not in the index region; resolve it from the
        // record store. A missing id means the region is inconsistent with the
        // record directory.
        const float* components = vector_for ? vector_for(node.id) : nullptr;
        if (components == nullptr) {
            return nullptr;
        }
        node.vector.assign(components, components + dimensionality);

        node.deleted = false;
        node.neighbors.resize(static_cast<std::size_t>(node.top_layer) + 1);
        for (int layer = 0; layer <= node.top_layer; ++layer) {
            const std::uint32_t degree = r.get_u32();
            if (!r.ok()) {
                return nullptr;
            }
            std::vector<std::uint32_t>& adj = node.neighbors[layer];
            adj.reserve(degree);
            for (std::uint32_t e = 0; e < degree; ++e) {
                const std::uint32_t nb = r.get_u32();
                // Neighbour ids are canonical ids in [0, count); reject any that
                // fall outside the live set as corruption.
                if (!r.ok() || nb >= count) {
                    return nullptr;
                }
                adj.push_back(nb);
            }
        }

        if (index->id_to_internal_.count(node.id) != 0) {
            return nullptr;  // duplicate id in the region
        }
        index->id_to_internal_.emplace(node.id, c);
        index->nodes_.push_back(std::move(node));
    }

    // Every byte of the region must have been consumed for a well-formed index.
    if (!r.ok() || r.remaining() != 0) {
        return nullptr;
    }

    index->live_count_ = count;
    index->max_level_ =
        (stored_max_lvl == 0xFFFFFFFFu) ? -1 : static_cast<int>(stored_max_lvl);
    if (count == 0) {
        index->entry_point_ = -1;
    } else if (entry_canonical == 0xFFFFFFFFu || entry_canonical >= count) {
        return nullptr;  // a non-empty graph must name a valid entry point
    } else {
        index->entry_point_ = static_cast<int>(entry_canonical);
    }

    out_ok = true;
    return index;
}

}  // namespace vectorvault
