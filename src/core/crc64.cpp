#include "vectorvault/crc64.hpp"

namespace vectorvault {

namespace {

// CRC-64/ECMA-182 polynomial in normal (non-reflected) form.
constexpr std::uint64_t kPoly = 0x42F0E1EBA9EA3693ULL;

// Builds the 256-entry lookup table for the non-reflected, MSB-first CRC. Each
// entry is the CRC of the byte value `i` placed in the top byte. Computed once
// on first use; the computation is deterministic, so checksums are stable
// across platforms and runs.
struct Crc64Table {
    std::uint64_t entries[256];

    Crc64Table() {
        for (int i = 0; i < 256; ++i) {
            std::uint64_t crc = static_cast<std::uint64_t>(i) << 56;
            for (int bit = 0; bit < 8; ++bit) {
                if (crc & 0x8000000000000000ULL) {
                    crc = (crc << 1) ^ kPoly;
                } else {
                    crc <<= 1;
                }
            }
            entries[i] = crc;
        }
    }
};

const Crc64Table& table() {
    static const Crc64Table kTable;
    return kTable;
}

}  // namespace

std::uint64_t crc64_ecma(const void* data, std::size_t length,
                         std::uint64_t crc) {
    const Crc64Table& t = table();
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < length; ++i) {
        const std::uint8_t index =
            static_cast<std::uint8_t>((crc >> 56) ^ bytes[i]);
        crc = (crc << 8) ^ t.entries[index];
    }
    return crc;
}

}  // namespace vectorvault
