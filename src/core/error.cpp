#include "vectorvault/error.hpp"

#include "vectorvault/types.hpp"

namespace vectorvault {

const char* to_string(ErrorCategory category) {
    switch (category) {
        case ErrorCategory::NameConflict:          return "NameConflict";
        case ErrorCategory::InvalidName:           return "InvalidName";
        case ErrorCategory::InvalidDimensionality: return "InvalidDimensionality";
        case ErrorCategory::InvalidMetric:         return "InvalidMetric";
        case ErrorCategory::NotFound:              return "NotFound";
        case ErrorCategory::DimensionMismatch:     return "DimensionMismatch";
        case ErrorCategory::InvalidValue:          return "InvalidValue";
        case ErrorCategory::InvalidIdentifier:     return "InvalidIdentifier";
        case ErrorCategory::InvalidParameter:      return "InvalidParameter";
        case ErrorCategory::InvalidArgument:       return "InvalidArgument";
        case ErrorCategory::UndefinedDistance:     return "UndefinedDistance";
        case ErrorCategory::AllocationFailure:     return "AllocationFailure";
        case ErrorCategory::WriteFailure:          return "WriteFailure";
        case ErrorCategory::SnapshotNotFound:      return "SnapshotNotFound";
        case ErrorCategory::UnsupportedVersion:    return "UnsupportedVersion";
        case ErrorCategory::Corruption:            return "Corruption";
        case ErrorCategory::DeletionFailed:        return "DeletionFailed";
    }
    return "Unknown";
}

}  // namespace vectorvault
