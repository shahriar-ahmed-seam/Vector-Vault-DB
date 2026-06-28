#ifndef VECTORVAULT_ERROR_HPP
#define VECTORVAULT_ERROR_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace vectorvault {

// Every recoverable failure in the Core_Engine is reported through a Status
// (or Result<T>) carrying one of these categories plus a human-readable cause.
// The Python_Binding maps each category onto a corresponding exception subclass.
//
// The backing type is fixed at uint16_t so the value stores compactly and stays
// stable across translation units.
enum class ErrorCategory : std::uint16_t {
    NameConflict,           // duplicate collection name
    InvalidName,            // name empty or > 255 chars
    InvalidDimensionality,  // dimensionality outside [1, 65536]
    InvalidMetric,          // unknown distance metric
    NotFound,               // missing collection or record
    DimensionMismatch,      // vector length != dimensionality
    InvalidValue,           // non-finite vector component
    InvalidIdentifier,      // empty record identifier
    InvalidParameter,       // index parameter out of range
    InvalidArgument,        // invalid k
    UndefinedDistance,      // cosine on a zero-norm vector
    AllocationFailure,      // allocator exhausted
    WriteFailure,           // snapshot save write failed
    SnapshotNotFound,       // missing or unparseable snapshot
    UnsupportedVersion,     // unsupported snapshot format version
    Corruption,             // snapshot integrity check failed
    DeletionFailed          // collection deletion aborted
};

// Returns a stable, human-readable name for an ErrorCategory, used to build the
// "category: cause" diagnostic messages. The names match the enumerator
// spellings.
const char* to_string(ErrorCategory category);

// Success, or an error carrying a category and a cause string. Failures are
// reported by value rather than thrown across the C++ ABI boundary. A
// default-constructed Status is "ok".
class Status {
public:
    // Constructs a success status.
    Status() = default;

    // Constructs a success status (explicit, equivalent to the default).
    static Status ok() { return Status{}; }

    // Constructs an error status with the given category and cause string.
    static Status error(ErrorCategory category, std::string cause) {
        return Status{category, std::move(cause)};
    }

    // True when the status represents success.
    bool is_ok() const { return !error_.has_value(); }

    // True when the status represents a failure.
    bool is_error() const { return error_.has_value(); }

    // Allows `if (status)` to read as "if ok".
    explicit operator bool() const { return is_ok(); }

    // The failing category. Only meaningful when is_error() is true.
    ErrorCategory category() const { return error_->category; }

    // The human-readable cause. Empty when the status is ok.
    const std::string& cause() const {
        static const std::string kEmpty;
        return error_ ? error_->cause : kEmpty;
    }

    // A "category: cause" diagnostic message. Empty when the status is ok.
    std::string message() const {
        if (!error_) {
            return std::string{};
        }
        return std::string{to_string(error_->category)} + ": " + error_->cause;
    }

private:
    struct ErrorState {
        ErrorCategory category;
        std::string   cause;
    };

    Status(ErrorCategory category, std::string cause)
        : error_(ErrorState{category, std::move(cause)}) {}

    std::optional<ErrorState> error_;
};

// A value on success, or an error Status on failure. Operations that produce a
// value return Result<T>; operations that produce none return Status. A Result
// is never constructed from a successful Status: success always carries T.
template <typename T>
class Result {
public:
    // Constructs a successful Result holding `value`.
    Result(T value) : storage_(std::move(value)) {}

    // Constructs a failed Result from an error Status.
    Result(Status status) : storage_(std::move(status)) {}

    // Convenience factory mirroring Status::error for failed results.
    static Result error(ErrorCategory category, std::string cause) {
        return Result{Status::error(category, std::move(cause))};
    }

    // True when the Result holds a value.
    bool is_ok() const { return std::holds_alternative<T>(storage_); }

    // True when the Result holds an error.
    bool is_error() const { return !is_ok(); }

    // Allows `if (result)` to read as "if ok".
    explicit operator bool() const { return is_ok(); }

    // Access the contained value. Precondition: is_ok().
    T& value() { return std::get<T>(storage_); }
    const T& value() const { return std::get<T>(storage_); }

    // The error category. Precondition: is_error().
    ErrorCategory category() const { return std::get<Status>(storage_).category(); }

    // The error cause string. Precondition: is_error().
    const std::string& cause() const { return std::get<Status>(storage_).cause(); }

    // The underlying error Status. Precondition: is_error().
    const Status& status() const { return std::get<Status>(storage_); }

private:
    std::variant<T, Status> storage_;
};

}  // namespace vectorvault

#endif  // VECTORVAULT_ERROR_HPP
