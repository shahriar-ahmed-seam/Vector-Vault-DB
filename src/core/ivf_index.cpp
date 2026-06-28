#include "vectorvault/ivf_index.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "vectorvault/distance.hpp"
#include "vectorvault/snapshot_format.hpp"

namespace vectorvault {

namespace {

constexpr float kInfinityF = std::numeric_limits<float>::infinity();

// Maximum Lloyd's iterations when refining centroids. A small fixed cap keeps
// (re)builds bounded; coarse quantization does not need an exact optimum.
constexpr int kMaxKMeansIterations = 12;

// Squared-L2 between two equal-length vectors; the coarse quantizer used for
// k-means assignment regardless of the collection's metric.
float squared_l2(const float* a, const float* b, std::size_t n) {
    return kernels::scalar_l2_squared(a, b, n);
}

}  // namespace

IvfIndex::IvfIndex(std::uint32_t dimensionality, DistanceMetric metric,
                   const DistanceCalculator& calculator, const IndexParams& params)
    : dimensionality_(dimensionality),
      metric_(metric),
      calculator_(&calculator),
      requested_nlist_(params.nlist),
      seed_(params.seed) {}

float IvfIndex::metric_distance(span<const float> q, span<const float> vec) const {
    Result<float> r = calculator_->distance(metric_, q, vec);
    return r.is_ok() ? r.value() : kInfinityF;
}

Status IvfIndex::add(const RecordId& id, span<const float> vec) {
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

    // Re-adding an existing id replaces its vector and revives a tombstone. Any
    // mutation invalidates the cached partitioning.
    auto it = id_to_index_.find(id);
    if (it != id_to_index_.end()) {
        Entry& entry = entries_[it->second];
        entry.vector.assign(vec.begin(), vec.end());
        if (entry.deleted) {
            entry.deleted = false;
            ++live_count_;
        }
        dirty_ = true;
        return Status::ok();
    }

    Entry entry;
    entry.id = id;
    entry.vector.assign(vec.begin(), vec.end());
    const std::size_t slot = entries_.size();
    entries_.push_back(std::move(entry));
    id_to_index_.emplace(id, slot);
    ++live_count_;
    dirty_ = true;
    return Status::ok();
}

Status IvfIndex::remove(const RecordId& id) {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) {
        return Status::error(ErrorCategory::NotFound,
                             "index has no record with id '" + id + "'");
    }
    Entry& entry = entries_[it->second];
    if (entry.deleted) {
        return Status::error(ErrorCategory::NotFound,
                             "index has no live record with id '" + id + "'");
    }
    // Tombstone so the record is skipped during clustering and result
    // collection alike.
    entry.deleted = true;
    --live_count_;
    dirty_ = true;
    return Status::ok();
}

void IvfIndex::ensure_partitions_built() const {
    if (!dirty_) {
        return;
    }

    centroids_.clear();
    inverted_lists_.clear();

    // Collect the live entry slots; tombstoned records take no part in
    // clustering.
    std::vector<std::uint32_t> live;
    live.reserve(entries_.size());
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (!entries_[i].deleted) {
            live.push_back(static_cast<std::uint32_t>(i));
        }
    }

    if (live.empty()) {
        dirty_ = false;
        return;
    }

    // Effective partition count: the requested nlist (or a heuristic default of
    // ~sqrt(live_count) when 0/auto), clamped to [1, live_count] since there
    // cannot be more cells than points.
    std::uint32_t k = requested_nlist_;
    if (k == 0) {
        const auto heuristic = static_cast<std::uint32_t>(
            std::sqrt(static_cast<double>(live.size())));
        k = heuristic < 1 ? 1u : heuristic;
    }
    if (k > live.size()) {
        k = static_cast<std::uint32_t>(live.size());
    }
    if (k < 1) {
        k = 1;
    }

    const std::size_t dim = dimensionality_;

    // Deterministic initialisation: shuffle the live slots with the seeded RNG
    // and take the first k distinct vectors as initial centroids.
    std::vector<std::uint32_t> shuffled = live;
    std::mt19937_64 rng(seed_);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);

    centroids_.assign(k, std::vector<float>(dim, 0.0f));
    for (std::uint32_t c = 0; c < k; ++c) {
        const std::vector<float>& src = entries_[shuffled[c]].vector;
        std::copy(src.begin(), src.end(), centroids_[c].begin());
    }

    std::vector<std::uint32_t> assignment(live.size(), 0);

    // Lloyd's iterations: assign each live vector to its nearest centroid, then
    // recompute each centroid as the mean of its assigned vectors.
    for (int iter = 0; iter < kMaxKMeansIterations; ++iter) {
        bool changed = false;

        for (std::size_t i = 0; i < live.size(); ++i) {
            const std::vector<float>& v = entries_[live[i]].vector;
            float best = kInfinityF;
            std::uint32_t best_c = 0;
            for (std::uint32_t c = 0; c < k; ++c) {
                const float d = squared_l2(v.data(), centroids_[c].data(), dim);
                if (d < best) {
                    best = d;
                    best_c = c;
                }
            }
            if (assignment[i] != best_c) {
                assignment[i] = best_c;
                changed = true;
            }
        }

        // Recompute centroids as the mean of their members.
        std::vector<std::vector<double>> sums(k, std::vector<double>(dim, 0.0));
        std::vector<std::uint64_t> counts(k, 0);
        for (std::size_t i = 0; i < live.size(); ++i) {
            const std::vector<float>& v = entries_[live[i]].vector;
            const std::uint32_t c = assignment[i];
            std::vector<double>& acc = sums[c];
            for (std::size_t d = 0; d < dim; ++d) {
                acc[d] += static_cast<double>(v[d]);
            }
            ++counts[c];
        }
        for (std::uint32_t c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                continue;  // keep an empty cell's centroid where it is
            }
            for (std::size_t d = 0; d < dim; ++d) {
                centroids_[c][d] = static_cast<float>(sums[c][d] /
                                                      static_cast<double>(counts[c]));
            }
        }

        if (!changed && iter > 0) {
            break;  // assignments stable: converged
        }
    }

    // Materialise the inverted lists from the final assignment.
    inverted_lists_.assign(k, {});
    for (std::size_t i = 0; i < live.size(); ++i) {
        inverted_lists_[assignment[i]].push_back(live[i]);
    }

    dirty_ = false;
}

std::vector<Candidate> IvfIndex::search(span<const float> q, std::uint32_t k,
                                        const QueryParams& params) const {
    if (k == 0 || live_count_ == 0) {
        return {};
    }

    ensure_partitions_built();

    const std::uint32_t nlist = static_cast<std::uint32_t>(centroids_.size());
    if (nlist == 0) {
        return {};
    }

    // Effective probe count: nprobe clamped to [1, nlist].
    std::uint32_t nprobe = params.nprobe < 1 ? 1u : params.nprobe;
    if (nprobe > nlist) {
        nprobe = nlist;
    }

    // Rank cells by squared-L2 from the query to each centroid (coarse
    // quantizer), then probe the nprobe nearest cells.
    std::vector<std::pair<float, std::uint32_t>> cell_dists;
    cell_dists.reserve(nlist);
    for (std::uint32_t c = 0; c < nlist; ++c) {
        const float d = squared_l2(q.data(), centroids_[c].data(), dimensionality_);
        cell_dists.emplace_back(d, c);
    }
    std::partial_sort(cell_dists.begin(),
                      cell_dists.begin() + static_cast<std::ptrdiff_t>(nprobe),
                      cell_dists.end());

    // Scan the probed cells, scoring live records under the collection metric.
    std::vector<Candidate> out;
    for (std::uint32_t p = 0; p < nprobe; ++p) {
        const std::uint32_t cell = cell_dists[p].second;
        for (std::uint32_t slot : inverted_lists_[cell]) {
            const Entry& entry = entries_[slot];
            if (entry.deleted) {
                continue;  // excluded from every query
            }
            const float dist = metric_distance(q, span<const float>(entry.vector));
            out.push_back(Candidate{entry.id, dist});
        }
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

std::vector<std::byte> IvfIndex::serialize() const {
    // The IVF partitioning is a deterministic function of the live record set
    // (seeded k-means over squared-L2), and the live record set is exactly the
    // collection's record directory. So the index region need only persist the
    // construction parameters; snapshot load re-adds every record from the
    // directory and rebuilds the identical partitioning. This keeps the region
    // canonical and fixed-width for a byte-identical round-trip and reconstructs
    // matching index membership.
    snapshot::ByteWriter w;
    w.put_u32(requested_nlist_);
    w.put_u64(seed_);
    return std::move(w.bytes());
}

}  // namespace vectorvault
