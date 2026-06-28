#include "vectorvault/collection.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vectorvault/distance.hpp"
#include "vectorvault/hnsw_index.hpp"
#include "vectorvault/ivf_index.hpp"
#include "vectorvault/mmap_region.hpp"

namespace vectorvault {

namespace {

bool is_valid_id(const RecordId& id) { return !id.empty(); }

// Index of the first non-finite component (NaN or +/-inf), or the span size
// when every component is finite. std::isfinite rejects NaN and the infinities.
std::size_t first_non_finite(span<const float> vec) {
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (!std::isfinite(vec[i])) {
            return i;
        }
    }
    return vec.size();
}

void store_components(const Block& block, span<const float> vec) {
    std::memcpy(block.data, vec.data(), vec.size() * sizeof(float));
}

// Maximum number of records accepted in a single batch.
constexpr std::size_t kMaxBatchSize = 10000;

}  // namespace

// Defined out-of-line, where MmapRegion is complete, so the header can hold the
// loaded mapping through a forward-declared type. Load installs the mapped
// state afterwards through friend access.
Collection::Collection(CollectionConfig config, MemoryAllocator& allocator,
                       CollectionTag tag, const DistanceCalculator& calculator)
    : config_(std::move(config)),
      allocator_(&allocator),
      tag_(tag),
      calculator_(&calculator) {}

// Defaulted here, where MmapRegion is complete, so the unique_ptr member can
// hold a forward-declared type in the header. Destroying a loaded collection
// releases its memory mapping; allocator-backed blocks of in-memory collections
// are released separately by the Engine on deletion.
Collection::~Collection() = default;

Status Collection::insert(const RecordId& id, span<const float> vec,
                          std::optional<Metadata> meta) {
    if (!is_valid_id(id)) {
        return Status::error(ErrorCategory::InvalidIdentifier,
                             "record identifier must be non-empty");
    }
    if (vec.size() != config_.dimensionality) {
        return Status::error(
            ErrorCategory::DimensionMismatch,
            "vector has " + std::to_string(vec.size()) +
                " components but the collection dimensionality is " +
                std::to_string(config_.dimensionality));
    }
    if (const std::size_t bad = first_non_finite(vec); bad != vec.size()) {
        return Status::error(
            ErrorCategory::InvalidValue,
            "vector component at index " + std::to_string(bad) +
                " is not a finite number");
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Allocate before mutating: if the allocator is exhausted the store is
    // untouched and the insert changes nothing.
    const std::size_t bytes =
        static_cast<std::size_t>(config_.dimensionality) * sizeof(float);
    Result<Block> block = allocator_->allocate(bytes, tag_);
    if (block.is_error()) {
        return block.status();
    }
    store_components(block.value(), vec);

    auto it = records_.find(id);
    if (it != records_.end()) {
        // Overwrite: replace the slot, release the old block; count is unchanged.
        allocator_->release(it->second.vector_block);
        it->second.vector_block = block.value();
        it->second.meta         = std::move(meta);
        // Inputs were validated above, so the index add cannot fail.
        if (index_) {
            index_->add(id, vec);
        }
        return Status::ok();
    }

    records_.emplace(id, RecordSlot{block.value(), std::move(meta)});
    ++record_count_;
    if (index_) {
        index_->add(id, vec);
    }
    return Status::ok();
}

Status Collection::insert_batch(span<const RecordInput> records) {
    if (records.empty() || records.size() > kMaxBatchSize) {
        return Status::error(
            ErrorCategory::InvalidArgument,
            "batch must contain 1 to 10000 records (got " +
                std::to_string(records.size()) + ")");
    }

    // Phase 1: validate every record before mutating anything, so an invalid
    // batch stores nothing. The first invalid record determines the error.
    for (std::size_t i = 0; i < records.size(); ++i) {
        const RecordInput& rec = records[i];
        if (!is_valid_id(rec.id)) {
            return Status::error(
                ErrorCategory::InvalidIdentifier,
                "record at batch index " + std::to_string(i) +
                    " has an empty identifier");
        }
        const span<const float> vec(rec.vector.data(), rec.vector.size());
        if (vec.size() != config_.dimensionality) {
            return Status::error(
                ErrorCategory::DimensionMismatch,
                "record at batch index " + std::to_string(i) + " has " +
                    std::to_string(vec.size()) +
                    " components but the collection dimensionality is " +
                    std::to_string(config_.dimensionality));
        }
        if (const std::size_t bad = first_non_finite(vec); bad != vec.size()) {
            return Status::error(
                ErrorCategory::InvalidValue,
                "record at batch index " + std::to_string(i) +
                    " has a non-finite component at index " +
                    std::to_string(bad));
        }
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Phase 2: stage every allocation up front. If any allocation fails, undo
    // the ones already made and return, keeping the batch atomic under
    // allocator exhaustion.
    const std::size_t bytes =
        static_cast<std::size_t>(config_.dimensionality) * sizeof(float);
    std::vector<Block> staged;
    staged.reserve(records.size());
    for (std::size_t i = 0; i < records.size(); ++i) {
        Result<Block> block = allocator_->allocate(bytes, tag_);
        if (block.is_error()) {
            for (const Block& b : staged) {
                allocator_->release(b);
            }
            return block.status();
        }
        store_components(block.value(),
                         span<const float>(records[i].vector.data(),
                                           records[i].vector.size()));
        staged.push_back(block.value());
    }

    // Phase 3: commit. Every allocation has succeeded, so no step here can fail.
    // Duplicate ids (within the batch or already present) overwrite the prior
    // slot, releasing its block and leaving the count unchanged; fresh ids
    // increment the count.
    for (std::size_t i = 0; i < records.size(); ++i) {
        const RecordInput& rec = records[i];
        auto it = records_.find(rec.id);
        if (it != records_.end()) {
            allocator_->release(it->second.vector_block);
            it->second.vector_block = staged[i];
            it->second.meta         = rec.meta;
        } else {
            records_.emplace(rec.id, RecordSlot{staged[i], rec.meta});
            ++record_count_;
        }
        // Inputs were validated in phase 1, so the index add cannot fail.
        if (index_) {
            index_->add(rec.id,
                        span<const float>(rec.vector.data(), rec.vector.size()));
        }
    }
    return Status::ok();
}

Result<RecordView> Collection::get(const RecordId& id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto it = records_.find(id);
    if (it == records_.end()) {
        return Result<RecordView>::error(
            ErrorCategory::NotFound,
            "no record with identifier '" + id + "' exists");
    }

    // Copy the components out of the aligned block; metadata is surfaced exactly
    // as stored, preserving the present/absent distinction.
    const RecordSlot& slot = it->second;
    const float* components = static_cast<const float*>(slot.vector_block.data);
    RecordView view;
    view.vector.assign(components, components + config_.dimensionality);
    view.meta = slot.meta;
    return Result<RecordView>{std::move(view)};
}

Status Collection::remove(const RecordId& id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = records_.find(id);
    if (it == records_.end()) {
        return Status::error(ErrorCategory::NotFound,
                             "no record with identifier '" + id + "' exists");
    }

    // Drop the record, then exclude it from the index. It was live in the store,
    // so it is live in the index.
    allocator_->release(it->second.vector_block);
    records_.erase(it);
    --record_count_;
    if (index_) {
        index_->remove(id);
    }
    return Status::ok();
}

Status Collection::build_index(IndexType type, const IndexParams& params) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Validate parameters before constructing any index, so a rejected request
    // leaves an existing index unchanged. IVF's nlist range depends on the
    // record count, so validation runs under the lock where it is known.
    switch (type) {
        case IndexType::HNSW:
            // m >= 2 and ef_construction >= 1; ef_search is validated at query
            // time.
            if (params.m < 2) {
                return Status::error(
                    ErrorCategory::InvalidParameter,
                    "HNSW parameter 'm' must be an integer of at least 2 (got " +
                        std::to_string(params.m) + ")");
            }
            if (params.ef_construction < 1) {
                return Status::error(
                    ErrorCategory::InvalidParameter,
                    "HNSW parameter 'ef_construction' must be an integer of at "
                    "least 1 (got " +
                        std::to_string(params.ef_construction) + ")");
            }
            break;
        case IndexType::IVF:
            // nlist must be in [1, record count]; 0 selects the auto-default.
            // nprobe is validated at query time.
            if (params.nlist != 0 && params.nlist > record_count_) {
                return Status::error(
                    ErrorCategory::InvalidParameter,
                    "IVF parameter 'nlist' must be an integer in [1, " +
                        std::to_string(record_count_) + "] (got " +
                        std::to_string(params.nlist) + ")");
            }
            break;
        case IndexType::None:
        default:
            return Status::error(
                ErrorCategory::InvalidArgument,
                "index type must be HNSW or IVF");
    }

    // Construct into a local so a mid-build failure cannot leave a partial index
    // in place of the existing one.
    std::unique_ptr<Index> new_index;
    if (type == IndexType::HNSW) {
        new_index = std::make_unique<HnswIndex>(
            config_.dimensionality, config_.metric, *calculator_, params);
    } else {  // IndexType::IVF
        new_index = std::make_unique<IvfIndex>(
            config_.dimensionality, config_.metric, *calculator_, params);
    }

    // Populate the new index with every live record. Records were validated on
    // insertion, so add() does not fail here.
    for (const auto& entry : records_) {
        const float* components =
            static_cast<const float*>(entry.second.vector_block.data);
        Status added = new_index->add(
            entry.first, span<const float>(components, config_.dimensionality));
        if (added.is_error()) {
            // Defensive: should not occur for already-validated records.
            return added;
        }
    }

    // Replace any existing index atomically under the exclusive lock.
    index_      = std::move(new_index);
    index_type_ = type;
    return Status::ok();
}

Result<std::vector<Neighbor>> Collection::query(span<const float> q,
                                                std::uint32_t k,
                                                const QueryParams& params) const {
    // k is unsigned in the signature, so a non-integer k cannot reach this layer
    // (the Python binding handles that); only k < 1 is rejected here.
    if (k < 1) {
        return Result<std::vector<Neighbor>>::error(
            ErrorCategory::InvalidArgument,
            "k must be an integer of at least 1 (got " + std::to_string(k) +
                ")");
    }
    if (q.size() != config_.dimensionality) {
        return Result<std::vector<Neighbor>>::error(
            ErrorCategory::DimensionMismatch,
            "query vector has " + std::to_string(q.size()) +
                " components but the collection dimensionality is " +
                std::to_string(config_.dimensionality));
    }
    if (const std::size_t bad = first_non_finite(q); bad != q.size()) {
        return Result<std::vector<Neighbor>>::error(
            ErrorCategory::InvalidValue,
            "query vector component at index " + std::to_string(bad) +
                " is not a finite number");
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<Neighbor> results;

    if (record_count_ == 0) {
        return Result<std::vector<Neighbor>>{std::move(results)};
    }

    // Exact distance from q to a stored record under the collection metric. An
    // undefined distance (chiefly cosine on a zero-norm stored vector) is
    // reported as +infinity so the record sorts to the back yet stays
    // returnable when k covers the collection.
    auto exact_distance = [&](const RecordSlot& slot) -> float {
        const float* components =
            static_cast<const float*>(slot.vector_block.data);
        Result<float> d = calculator_->distance(
            config_.metric, q,
            span<const float>(components, config_.dimensionality));
        if (d.is_error()) {
            return std::numeric_limits<float>::infinity();
        }
        return d.value();
    };

    // Choose between an exact brute-force scan and an index-assisted scan. When
    // k covers the whole collection, brute force returns every record directly;
    // we also brute force when no index exists. Otherwise the index narrows the
    // candidate set and we re-rank those candidates by exact distance, so the
    // reported distances and ordering stay exact even though the index is
    // approximate.
    const bool use_index = index_ && index_type_ != IndexType::None &&
                           static_cast<std::uint64_t>(k) < record_count_;

    if (use_index) {
        const std::vector<Candidate> candidates = index_->search(q, k, params);

        // Re-rank the index's candidates by their exact distance, dropping any
        // stale (already-removed) or duplicate ids.
        std::unordered_set<RecordId> seen;
        seen.reserve(candidates.size());
        for (const Candidate& c : candidates) {
            auto it = records_.find(c.id);
            if (it == records_.end() || !seen.insert(c.id).second) {
                continue;
            }
            results.push_back(Neighbor{it->first, exact_distance(it->second)});
        }

        // If the approximate structure returned fewer live candidates than we
        // are able to return (< min(k, record_count)), fall back to an exact
        // brute-force scan so coverage and exactness still hold.
        const std::uint64_t want = std::min<std::uint64_t>(k, record_count_);
        if (static_cast<std::uint64_t>(results.size()) < want) {
            results.clear();
            results.reserve(static_cast<std::size_t>(record_count_));
            for (const auto& entry : records_) {
                results.push_back(
                    Neighbor{entry.first, exact_distance(entry.second)});
            }
        }
    } else {
        // Brute force: compute the exact distance to every stored record.
        results.reserve(static_cast<std::size_t>(record_count_));
        for (const auto& entry : records_) {
            results.push_back(
                Neighbor{entry.first, exact_distance(entry.second)});
        }
    }

    // Ascending distance, ties broken by ascending identifier for determinism.
    // All distances are finite or +infinity (never NaN), so this is a valid
    // strict-weak ordering.
    std::sort(results.begin(), results.end(),
              [](const Neighbor& a, const Neighbor& b) {
                  if (a.distance != b.distance) {
                      return a.distance < b.distance;
                  }
                  return a.id < b.id;
              });

    // Return up to k results; when k exceeds the record count this keeps every
    // record.
    if (static_cast<std::uint64_t>(results.size()) > k) {
        results.resize(k);
    }

    return Result<std::vector<Neighbor>>{std::move(results)};
}

CollectionSnapshotData Collection::snapshot_data() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    CollectionSnapshotData data;
    data.config     = config_;
    data.index_type = index_type_;

    // records_ is a std::map keyed by RecordId, so iteration is already in
    // ascending-id order — the canonical record-directory order. Components are
    // copied out of each aligned block, preserving metadata presence/absence.
    data.records.reserve(records_.size());
    for (const auto& entry : records_) {
        const RecordSlot& slot = entry.second;
        const float* components =
            static_cast<const float*>(slot.vector_block.data);
        CollectionSnapshotData::Record rec;
        rec.id = entry.first;
        rec.vector.assign(components, components + config_.dimensionality);
        rec.meta = slot.meta;
        data.records.push_back(std::move(rec));
    }

    // Capture the index as canonical serialized bytes under the lock so it is
    // consistent with the records above.
    if (index_ && index_type_ != IndexType::None) {
        data.index_blob = index_->serialize();
    }

    return data;
}

}  // namespace vectorvault
