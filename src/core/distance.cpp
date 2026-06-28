#include "vectorvault/distance.hpp"

#include <cmath>
#include <string>

// AVX-512 intrinsics are only pulled in on x86 GCC/Clang builds. The kernels
// below are compiled with a per-function target attribute (see avx512f_target)
// rather than a global -mavx512f flag, so the translation unit still produces
// baseline code that loads and runs on non-AVX-512 CPUs; the SIMD functions are
// only ever entered after runtime detection confirms support.
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#  include <immintrin.h>
#  define VECTORVAULT_HAVE_AVX512_KERNELS 1
#  define VECTORVAULT_AVX512F_TARGET __attribute__((target("avx512f")))
#else
#  define VECTORVAULT_HAVE_AVX512_KERNELS 0
#  define VECTORVAULT_AVX512F_TARGET
#endif

namespace vectorvault {

// Scalar reference kernels.

namespace kernels {

// The reductions accumulate in double and return float. Widening the
// accumulator preserves accuracy for large vectors (dimensionality up to a few
// thousand) and keeps the scalar path numerically consistent with the SIMD
// kernels, which also widen to double.
float scalar_l2_squared(const float* a, const float* b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return static_cast<float>(sum);
}

float scalar_dot(const float* a, const float* b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(sum);
}

}  // namespace kernels

// AVX-512 SIMD kernels.
//
// These mirror the scalar reductions above but process 16 float32 lanes per
// __m512 register, then horizontally reduce the 16 partial sums to a single
// scalar and finish any remainder (n not a multiple of 16) with a scalar tail.
// They are bound by create() only when the host reports AVX-512 support; the
// scalar kernels remain both the fallback and the accuracy oracle.
//
// Each function carries an avx512f target attribute so the compiler may emit
// AVX-512 instructions for it without enabling those instructions for the rest
// of the translation unit. This keeps the binary runnable on CPUs without
// AVX-512: such CPUs never reach these functions because detection routes them
// to the scalar path.
#if VECTORVAULT_HAVE_AVX512_KERNELS
namespace {

// Widen the upper 8 float32 lanes of a __m512 to a __m256 using only AVX512F
// facilities. The natural intrinsic for this (_mm512_extractf32x8_ps) is part
// of AVX512DQ, which the runtime detection does not require; reinterpreting as
// doubles and extracting the upper 256-bit half (_mm512_extractf64x4_pd) is an
// AVX512F operation, so the kernels stay within the detected feature set.
VECTORVAULT_AVX512F_TARGET
inline __m256 upper_ps256(__m512 v) {
    return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
}

// Sum of squared component differences over 16-lane registers with a scalar
// remainder tail. Equivalent to kernels::scalar_l2_squared.
//
// The per-lane squared differences are widened to double and accumulated in
// double precision. For vectors up to 4096 components this keeps the result
// within the 1e-4 tolerance of the scalar reference even for inputs that would
// otherwise lose precision through float32 accumulation.
VECTORVAULT_AVX512F_TARGET
float avx512_l2_squared(const float* a, const float* b, std::size_t n) {
    __m512d acc_lo = _mm512_setzero_pd();  // lanes 0..7 (widened to double)
    __m512d acc_hi = _mm512_setzero_pd();  // lanes 8..15 (widened to double)

    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 va   = _mm512_loadu_ps(a + i);
        const __m512 vb   = _mm512_loadu_ps(b + i);
        const __m512 diff = _mm512_sub_ps(va, vb);
        // Widen the two 256-bit halves of the float32 differences to double.
        const __m512d dlo = _mm512_cvtps_pd(_mm512_castps512_ps256(diff));
        const __m512d dhi = _mm512_cvtps_pd(upper_ps256(diff));
        // acc += diff * diff in double precision (fused multiply-add).
        acc_lo = _mm512_fmadd_pd(dlo, dlo, acc_lo);
        acc_hi = _mm512_fmadd_pd(dhi, dhi, acc_hi);
    }

    // Horizontal reduction of the lane-wise partial sums.
    double sum = _mm512_reduce_add_pd(acc_lo) + _mm512_reduce_add_pd(acc_hi);

    // Scalar tail for the remaining (n % 16) components.
    for (; i < n; ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return static_cast<float>(sum);
}

// Dot product over 16-lane registers with a scalar remainder tail. Equivalent
// to kernels::scalar_dot. The cosine metric reuses this kernel to derive both
// norms and the cross term (||v||^2 == dot(v, v)).
//
// Products are widened to double and accumulated in double precision so the
// reduction stays within the 1e-4 tolerance of the scalar reference even when
// the result is small relative to the magnitude of the intermediate terms.
VECTORVAULT_AVX512F_TARGET
float avx512_dot(const float* a, const float* b, std::size_t n) {
    __m512d acc_lo = _mm512_setzero_pd();
    __m512d acc_hi = _mm512_setzero_pd();

    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 va = _mm512_loadu_ps(a + i);
        const __m512 vb = _mm512_loadu_ps(b + i);
        const __m512d alo = _mm512_cvtps_pd(_mm512_castps512_ps256(va));
        const __m512d ahi = _mm512_cvtps_pd(upper_ps256(va));
        const __m512d blo = _mm512_cvtps_pd(_mm512_castps512_ps256(vb));
        const __m512d bhi = _mm512_cvtps_pd(upper_ps256(vb));
        // acc += a * b in double precision (fused multiply-add).
        acc_lo = _mm512_fmadd_pd(alo, blo, acc_lo);
        acc_hi = _mm512_fmadd_pd(ahi, bhi, acc_hi);
    }

    double sum = _mm512_reduce_add_pd(acc_lo) + _mm512_reduce_add_pd(acc_hi);

    for (; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(sum);
}

}  // namespace
#endif  // VECTORVAULT_HAVE_AVX512_KERNELS

namespace {

// One-time AVX-512 detection. Resolved on first use via a function-local static,
// which the C++ runtime initializes exactly once and in a thread-safe manner,
// so the answer is fixed for the process lifetime.
bool host_supports_avx512() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    // avx512f is the foundation feature the SIMD kernels require for 16-lane
    // float32 operations.
    return __builtin_cpu_supports("avx512f") != 0;
#else
    // Non-x86 targets and compilers without the builtin have no AVX-512 path.
    return false;
#endif
}

bool detect_avx512_once() {
    static const bool supported = host_supports_avx512();
    return supported;
}

}  // namespace

// DistanceCalculator.

DistanceCalculator DistanceCalculator::create() {
    const bool avx512 = detect_avx512_once();

    // Dispatch is bound here, once, for the lifetime of the calculator. The
    // scalar kernels are the default and remain the fallback when the host has
    // no AVX-512 support, as well as the accuracy oracle for the SIMD path.
    Kernel l2_squared = &kernels::scalar_l2_squared;
    Kernel dot        = &kernels::scalar_dot;

    // When the host reports AVX-512 support, bind the 16-lane SIMD kernels in
    // place of the scalar ones.
#if VECTORVAULT_HAVE_AVX512_KERNELS
    if (avx512) {
        l2_squared = &avx512_l2_squared;
        dot        = &avx512_dot;
    }
#endif

    return DistanceCalculator(l2_squared, dot, avx512);
}

Result<float> DistanceCalculator::distance(DistanceMetric metric,
                                           span<const float> a,
                                           span<const float> b) const {
    if (a.size() != b.size()) {
        return Result<float>::error(
            ErrorCategory::DimensionMismatch,
            "distance requires vectors of equal length (got " +
                std::to_string(a.size()) + " and " + std::to_string(b.size()) +
                ")");
    }

    const std::size_t n = a.size();

    switch (metric) {
        case DistanceMetric::Euclidean: {
            // Non-negative square root of the sum of squared diffs. For
            // identical vectors every difference is exactly 0, so the sum is 0
            // and sqrt(0) == 0.
            const float sum_sq = l2_squared_(a.data(), b.data(), n);
            return std::sqrt(sum_sq);
        }

        case DistanceMetric::DotProduct: {
            return dot_(a.data(), b.data(), n);
        }

        case DistanceMetric::Cosine: {
            // 1 - dot(a,b) / (||a|| * ||b||), clamped to [0, 2]. Norms are
            // derived from the dot kernel: ||v||^2 == dot(v, v).
            const float norm_a_sq = dot_(a.data(), a.data(), n);
            const float norm_b_sq = dot_(b.data(), b.data(), n);

            // Cosine is undefined when either vector has zero norm.
            if (norm_a_sq == 0.0f || norm_b_sq == 0.0f) {
                return Result<float>::error(
                    ErrorCategory::UndefinedDistance,
                    "cosine distance is undefined for a zero-norm vector");
            }

            const float dot_ab = dot_(a.data(), b.data(), n);
            const float denom  = std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq);
            float cosine_distance = 1.0f - (dot_ab / denom);

            // Cosine similarity is mathematically in [-1, 1] so the distance is
            // in [0, 2]; clamp to absorb floating-point overshoot at the
            // endpoints and guarantee the documented range.
            if (cosine_distance < 0.0f) {
                cosine_distance = 0.0f;
            } else if (cosine_distance > 2.0f) {
                cosine_distance = 2.0f;
            }
            return cosine_distance;
        }
    }

    // Defensive: a value outside the defined enumerators. The shared model
    // only produces valid metrics, but keep the dispatch total.
    return Result<float>::error(ErrorCategory::InvalidMetric,
                                "unknown distance metric");
}

}  // namespace vectorvault
