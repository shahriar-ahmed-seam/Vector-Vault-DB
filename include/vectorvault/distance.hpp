#ifndef VECTORVAULT_DISTANCE_HPP
#define VECTORVAULT_DISTANCE_HPP

#include <cstddef>

#include "vectorvault/error.hpp"
#include "vectorvault/span.hpp"
#include "vectorvault/types.hpp"

namespace vectorvault {

// Portable scalar implementations of the two primitive reductions the distance
// metrics are built from. They double as the reference oracle the SIMD path is
// checked against.
namespace kernels {

// Sum of squared component differences: sum_i (a_i - b_i)^2. This is the
// squared L2 distance; the Euclidean metric takes its sqrt.
float scalar_l2_squared(const float* a, const float* b, std::size_t n);

// Dot product: sum_i (a_i * b_i). Also used to derive squared L2 norms for the
// cosine metric (||v||^2 == dot(v, v)).
float scalar_dot(const float* a, const float* b, std::size_t n);

}  // namespace kernels

// Computes a DistanceMetric between two equal-length float32 vectors. Stateless
// apart from the one-time dispatch decision made at create(), which detects
// host AVX-512 support exactly once for the process lifetime and binds the
// kernel function pointers accordingly.
//
// Kernel semantics:
//   * Euclidean : sqrt(sum_i (a_i - b_i)^2)              -> non-negative
//   * DotProduct: sum_i (a_i * b_i)
//   * Cosine    : 1 - dot(a,b) / (||a|| * ||b||), clamped to [0, 2]
//
// Errors:
//   * DimensionMismatch when the two vectors differ in length
//   * UndefinedDistance for cosine when either vector has a zero L2 norm
class DistanceCalculator {
public:
    // Detects AVX-512 support once and selects the implementation path for the
    // process lifetime.
    static DistanceCalculator create();

    // Computes the distance between `a` and `b` under `metric`. Returns
    // DimensionMismatch if the lengths differ and UndefinedDistance for cosine
    // on a zero-norm vector.
    Result<float> distance(DistanceMetric metric,
                           span<const float> a,
                           span<const float> b) const;

    // True when the active path is AVX-512; false when the scalar fallback is
    // in use.
    bool uses_simd() const { return uses_simd_; }

private:
    // A kernel reduces two equal-length vectors to a single scalar.
    using Kernel = float (*)(const float*, const float*, std::size_t);

    DistanceCalculator(Kernel l2_squared, Kernel dot, bool uses_simd)
        : l2_squared_(l2_squared), dot_(dot), uses_simd_(uses_simd) {}

    Kernel l2_squared_;  // returns sum of squared differences (squared L2)
    Kernel dot_;         // returns the dot product
    bool   uses_simd_;   // whether AVX-512 was detected at create()
};

}  // namespace vectorvault

#endif  // VECTORVAULT_DISTANCE_HPP
