#include "vectorvault/mmap_region.hpp"

#include <utility>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>     // open
#  include <sys/mman.h>  // mmap, munmap
#  include <sys/stat.h>  // fstat
#  include <unistd.h>    // close
#endif

namespace vectorvault {

MmapRegion::MmapRegion(MmapRegion&& other) noexcept {
    *this = std::move(other);
}

MmapRegion& MmapRegion::operator=(MmapRegion&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = other.data_;
        size_ = other.size_;
#if defined(_WIN32)
        file_handle_    = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_    = nullptr;
        other.mapping_handle_ = nullptr;
#else
        fd_       = other.fd_;
        map_addr_ = other.map_addr_;
        map_len_  = other.map_len_;
        other.fd_       = -1;
        other.map_addr_ = nullptr;
        other.map_len_  = 0;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

MmapRegion::~MmapRegion() { reset(); }

// reset releases the mapping and OS handles.
void MmapRegion::reset() noexcept {
#if defined(_WIN32)
    if (data_ != nullptr) {
        ::UnmapViewOfFile(const_cast<std::byte*>(data_));
    }
    if (mapping_handle_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(mapping_handle_));
        mapping_handle_ = nullptr;
    }
    if (file_handle_ != nullptr && file_handle_ != INVALID_HANDLE_VALUE) {
        ::CloseHandle(static_cast<HANDLE>(file_handle_));
        file_handle_ = nullptr;
    }
#else
    if (map_addr_ != nullptr && map_len_ != 0) {
        ::munmap(map_addr_, map_len_);
        map_addr_ = nullptr;
        map_len_  = 0;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    data_ = nullptr;
    size_ = 0;
}

// map_file_readonly creates a platform-specific read-only mapping.
std::unique_ptr<MmapRegion> MmapRegion::map_file_readonly(
    const std::filesystem::path& path) {
    auto region = std::make_unique<MmapRegion>();

#if defined(_WIN32)
    // Open the file for reading, allowing other readers concurrent access.
    HANDLE file = ::CreateFileW(path.wstring().c_str(), GENERIC_READ,
                                FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return nullptr;
    }

    LARGE_INTEGER file_size;
    if (::GetFileSizeEx(file, &file_size) == 0 || file_size.QuadPart <= 0) {
        ::CloseHandle(file);
        return nullptr;
    }

    // Create a read-only file mapping covering the whole file, then map a view.
    HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0,
                                          nullptr);
    if (mapping == nullptr) {
        ::CloseHandle(file);
        return nullptr;
    }

    void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        ::CloseHandle(mapping);
        ::CloseHandle(file);
        return nullptr;
    }

    region->file_handle_    = file;
    region->mapping_handle_ = mapping;
    region->data_           = static_cast<const std::byte*>(view);
    region->size_           = static_cast<std::size_t>(file_size.QuadPart);
    return region;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return nullptr;
    }

    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return nullptr;
    }
    const std::size_t len = static_cast<std::size_t>(st.st_size);

    // MAP_PRIVATE is sufficient for a read-only view and never writes back.
    void* addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        ::close(fd);
        return nullptr;
    }

    region->fd_       = fd;
    region->map_addr_ = addr;
    region->map_len_  = len;
    region->data_     = static_cast<const std::byte*>(addr);
    region->size_     = len;
    return region;
#endif
}

}  // namespace vectorvault
