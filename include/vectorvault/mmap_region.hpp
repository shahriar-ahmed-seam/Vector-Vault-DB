#ifndef VECTORVAULT_MMAP_REGION_HPP
#define VECTORVAULT_MMAP_REGION_HPP

#include <cstddef>
#include <filesystem>
#include <memory>

namespace vectorvault {

// Cross-platform RAII owner of a read-only memory mapping of a file. Snapshot
// load maps the vector region rather than copying it into the heap: MmapRegion
// maps an entire file read-only into the address space and unmaps it on
// destruction. A loaded Collection holds one for its whole lifetime and backs
// each record's vector storage with a pointer into data(), so vector reads
// resolve through the mapping (and the OS page cache).
//
// The OS mapping base is page-aligned and the snapshot lays the vector region
// out on a 64-byte boundary, so vectors read from the mapping are 64-byte
// aligned and safe for SIMD reads.
//
// Move-only: copying a mapping would risk a double unmap. The underlying bytes
// are read-only; callers must never write through data().
class MmapRegion {
public:
    MmapRegion() = default;
    ~MmapRegion();

    MmapRegion(const MmapRegion&) = delete;
    MmapRegion& operator=(const MmapRegion&) = delete;
    MmapRegion(MmapRegion&& other) noexcept;
    MmapRegion& operator=(MmapRegion&& other) noexcept;

    // Maps the entire file at `path` read-only and returns the owning region,
    // or nullptr if it cannot be opened or mapped (e.g. it does not exist, is
    // not readable, or is empty). The caller has already verified existence and
    // a minimum size, so a nullptr return is treated as an unparseable snapshot.
    static std::unique_ptr<MmapRegion> map_file_readonly(
        const std::filesystem::path& path);

    // The start of the mapped bytes (read-only), or nullptr when not mapped.
    const std::byte* data() const { return data_; }

    // The number of mapped bytes (the file size).
    std::size_t size() const { return size_; }

    // True when a mapping is held.
    bool valid() const { return data_ != nullptr; }

private:
    // Releases the mapping and associated OS handles, returning to the empty
    // state. Safe to call on an already-empty region.
    void reset() noexcept;

    const std::byte* data_ = nullptr;  // mapped view start (read-only)
    std::size_t      size_ = 0;        // mapped length in bytes

#if defined(_WIN32)
    void* file_handle_    = nullptr;  // HANDLE from CreateFileW (or INVALID)
    void* mapping_handle_ = nullptr;  // HANDLE from CreateFileMappingW
#else
    int         fd_       = -1;       // open file descriptor
    void*       map_addr_ = nullptr;  // base passed to munmap
    std::size_t map_len_  = 0;        // length passed to munmap
#endif
};

}  // namespace vectorvault

#endif  // VECTORVAULT_MMAP_REGION_HPP
