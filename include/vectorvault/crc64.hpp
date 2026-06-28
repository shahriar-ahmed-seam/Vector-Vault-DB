#ifndef VECTORVAULT_CRC64_HPP
#define VECTORVAULT_CRC64_HPP

#include <cstddef>
#include <cstdint>

namespace vectorvault {

// CRC-64/ECMA-182 over the snapshot content (every byte after the fixed
// header), used by load to detect corruption. Standard parameters:
//   polynomial 0x42F0E1EBA9EA3693 (non-reflected), init 0, no reflect, no xor.
//
// May be called incrementally: pass the running value back in as `crc` to
// continue across several buffers. A `crc` of 0 starts a fresh checksum.
std::uint64_t crc64_ecma(const void* data, std::size_t length,
                         std::uint64_t crc = 0);

}  // namespace vectorvault

#endif  // VECTORVAULT_CRC64_HPP
