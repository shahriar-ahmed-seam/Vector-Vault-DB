#include "vectorvault/engine.hpp"

#include <string>
#include <utility>

namespace vectorvault {

namespace {

bool is_valid_name(std::string_view name) {
    return !name.empty() && name.size() <= 255;
}

bool is_valid_dimensionality(std::uint32_t dimensionality) {
    return dimensionality >= 1 && dimensionality <= 65536;
}

// Checked by underlying value so a value fabricated outside the defined set
// (e.g. crossing the Python boundary) is rejected rather than silently accepted.
bool is_valid_metric(DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::Euclidean:
        case DistanceMetric::Cosine:
        case DistanceMetric::DotProduct:
            return true;
    }
    return false;
}

}  // namespace

Engine::Engine() : allocator_(), distance_calculator_(DistanceCalculator::create()) {}

Engine::Engine(std::size_t allocator_capacity_bytes)
    : allocator_(allocator_capacity_bytes),
      distance_calculator_(DistanceCalculator::create()) {}

Result<CollectionHandle> Engine::create_collection(std::string_view name,
                                                   std::uint32_t dimensionality,
                                                   DistanceMetric metric) {
    // Validate fully before touching the registry so every rejection creates no
    // Collection.
    if (!is_valid_name(name)) {
        return Result<CollectionHandle>::error(
            ErrorCategory::InvalidName,
            "collection name must be 1 to 255 characters (got " +
                std::to_string(name.size()) + ")");
    }
    if (!is_valid_dimensionality(dimensionality)) {
        return Result<CollectionHandle>::error(
            ErrorCategory::InvalidDimensionality,
            "dimensionality must be in [1, 65536] (got " +
                std::to_string(dimensionality) + ")");
    }
    if (!is_valid_metric(metric)) {
        return Result<CollectionHandle>::error(
            ErrorCategory::InvalidMetric,
            "distance metric must be Euclidean, Cosine, or DotProduct (got " +
                std::to_string(static_cast<unsigned>(
                    static_cast<std::uint8_t>(metric))) +
                ")");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (collections_.find(name) != collections_.end()) {
        return Result<CollectionHandle>::error(
            ErrorCategory::NameConflict,
            "a collection named '" + std::string(name) + "' already exists");
    }

    // Each collection gets a unique allocator tag for per-collection byte
    // attribution and release-on-delete.
    CollectionConfig config;
    config.name           = std::string(name);
    config.dimensionality = dimensionality;
    config.metric         = metric;

    const CollectionTag tag{next_tag_++};
    auto collection = std::make_unique<Collection>(std::move(config), allocator_,
                                                   tag, distance_calculator_);
    Collection* raw = collection.get();

    collections_.emplace(std::string(name), std::move(collection));

    return Result<CollectionHandle>{CollectionHandle{raw}};
}

Status Engine::delete_collection(std::string_view name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = collections_.find(name);
    if (it == collections_.end()) {
        return Status::error(
            ErrorCategory::NotFound,
            "no collection named '" + std::string(name) + "' exists");
    }

    // Deletion fault injection for tests: if the hook signals a failure, abort
    // before any change so the collection stays retrievable.
    if (delete_fault_injector_ && delete_fault_injector_(name)) {
        return Status::error(
            ErrorCategory::DeletionFailed,
            "deletion of collection '" + std::string(name) +
                "' failed before completion");
    }

    // Release every block attributed to this collection, then remove it from
    // the registry.
    const CollectionTag tag = it->second->tag();
    allocator_.release_all(tag);
    collections_.erase(it);

    return Status::ok();
}

std::vector<CollectionInfo> Engine::list_collections() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<CollectionInfo> infos;
    infos.reserve(collections_.size());
    for (const auto& entry : collections_) {
        infos.push_back(entry.second->info());
    }
    // One entry per collection in name order; empty when none exist.
    return infos;
}

Result<CollectionHandle> Engine::get_collection(std::string_view name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = collections_.find(name);
    if (it == collections_.end()) {
        return Result<CollectionHandle>::error(
            ErrorCategory::NotFound,
            "no collection named '" + std::string(name) + "' exists");
    }
    return Result<CollectionHandle>{CollectionHandle{it->second.get()}};
}

Status Engine::save_collection(std::string_view name,
                               const std::filesystem::path& path) {
    // Hold the registry lock across the save so the resolved collection cannot
    // be deleted out from under the serializer. The collection captures its own
    // consistent view under its readers-writer lock during serialization.
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = collections_.find(name);
    if (it == collections_.end()) {
        return Status::error(
            ErrorCategory::NotFound,
            "no collection named '" + std::string(name) + "' exists");
    }

    return persistence_.save(*it->second, path);
}

Result<CollectionHandle> Engine::load_collection(
    const std::filesystem::path& path) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Reserve a unique allocator tag for the reconstructed collection. A loaded
    // collection's vector data is memory-mapped (not allocator-backed), so the
    // tag carries no bytes until a later mutation, but it must still be unique
    // for that future attribution.
    const CollectionTag tag{next_tag_++};

    Result<std::unique_ptr<Collection>> loaded =
        persistence_.load(path, allocator_, tag, distance_calculator_);
    if (loaded.is_error()) {
        // Nothing was registered or allocated under the tag; surface the
        // validation error.
        return Result<CollectionHandle>{loaded.status()};
    }

    std::unique_ptr<Collection> collection = std::move(loaded.value());
    const std::string& name = collection->config().name;

    // Reject a load whose stored name collides with an existing collection. The
    // loaded collection is dropped here (its mapping is unmapped by its
    // destructor); release any tag-attributed bytes defensively, though a
    // freshly loaded collection has none.
    if (collections_.find(name) != collections_.end()) {
        allocator_.release_all(tag);
        return Result<CollectionHandle>::error(
            ErrorCategory::NameConflict,
            "a collection named '" + name + "' already exists");
    }

    Collection* raw = collection.get();
    collections_.emplace(name, std::move(collection));
    return Result<CollectionHandle>{CollectionHandle{raw}};
}

}  // namespace vectorvault
