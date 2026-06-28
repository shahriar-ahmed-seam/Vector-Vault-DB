#ifndef VECTORVAULT_SPAN_HPP
#define VECTORVAULT_SPAN_HPP

// vectorvault::span - a non-owning view over a contiguous sequence.
//
// The engine's vector parameters are expressed as span<const float>. std::span
// is C++20, but the Core_Engine currently compiles as C++17, so this header
// provides a compatibility alias:
//   * When <span> is available (C++20+), span is an alias for std::span, so the
//     codebase transparently uses the real type once the project moves to C++20.
//   * Otherwise it is a minimal, dynamic-extent, read-and-iterate span covering
//     exactly what the engine needs: construction from a pointer and length,
//     from a C array, and from any contiguous container exposing data()/size()
//     (e.g. std::vector, std::array), plus size/empty, indexing, and iteration.

#include <cstddef>

#if defined(__has_include)
#  if __has_include(<version>)
#    include <version>
#  endif
#endif

#if defined(__cpp_lib_span) && __cpp_lib_span >= 202002L

#include <span>

namespace vectorvault {
template <typename T>
using span = std::span<T>;
}  // namespace vectorvault

#else  // ----------------------------- C++17 fallback ----------------------

#include <type_traits>

namespace vectorvault {

template <typename T>
class span {
public:
    using element_type = T;
    using value_type   = std::remove_cv_t<T>;
    using size_type    = std::size_t;
    using pointer      = T*;
    using reference    = T&;
    using iterator     = T*;

    // Empty span.
    constexpr span() noexcept : data_(nullptr), size_(0) {}

    // View of `count` elements starting at `ptr`.
    constexpr span(pointer ptr, size_type count) noexcept
        : data_(ptr), size_(count) {}

    // View over a C array.
    template <std::size_t N>
    constexpr span(element_type (&arr)[N]) noexcept : data_(arr), size_(N) {}

    // View over any contiguous container exposing data()/size() whose element
    // pointer is qualification-convertible to T* (so a std::vector<float> can
    // construct a span<const float>). The array-of-pointer convertibility check
    // mirrors the std::span constructor constraint.
    template <
        typename Container,
        typename = std::enable_if_t<
            !std::is_same<std::remove_cv_t<Container>, span>::value &&
            std::is_convertible<
                std::remove_pointer_t<
                    decltype(std::declval<Container&>().data())> (*)[],
                T (*)[]>::value>>
    constexpr span(Container& c) : data_(c.data()), size_(c.size()) {}

    constexpr pointer   data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return size_; }
    constexpr bool      empty() const noexcept { return size_ == 0; }

    constexpr reference operator[](size_type i) const { return data_[i]; }

    constexpr iterator begin() const noexcept { return data_; }
    constexpr iterator end() const noexcept { return data_ + size_; }

private:
    pointer   data_;
    size_type size_;
};

}  // namespace vectorvault

#endif  // __cpp_lib_span

#endif  // VECTORVAULT_SPAN_HPP
