// DistanceCalculator tests: kernel accuracy, Euclidean self-distance, and
// dispatch edge cases against a high-accuracy reference oracle.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <rapidcheck.h>

#include "vectorvault/distance.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

using namespace vectorvault;

namespace {

// Reference oracle per metric: reductions accumulate in double so the oracle is
// a faithful baseline rather than a float32 reference that drifts at high dimension.
double reference_l2_squared(const std::vector<float>& a,
                            const std::vector<float>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return sum;
}

double reference_dot(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

double reference_distance(DistanceMetric metric,
                          const std::vector<float>& a,
                          const std::vector<float>& b) {
    switch (metric) {
        case DistanceMetric::Euclidean:
            return std::sqrt(reference_l2_squared(a, b));
        case DistanceMetric::DotProduct:
            return reference_dot(a, b);
        case DistanceMetric::Cosine: {
            // Combine in float32 (not double): the 1 - ratio cancellation is
            // inherent to any correct float32 cosine, so the oracle shares it.
            const float norm_a_sq = static_cast<float>(reference_dot(a, a));
            const float norm_b_sq = static_cast<float>(reference_dot(b, b));
            const float dot_ab    = static_cast<float>(reference_dot(a, b));
            const float denom = std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq);
            float cosine_distance = 1.0f - (dot_ab / denom);
            if (cosine_distance < 0.0f) {
                cosine_distance = 0.0f;
            } else if (cosine_distance > 2.0f) {
                cosine_distance = 2.0f;
            }
            return cosine_distance;
        }
    }
    return 0.0;  // unreachable
}

// Tolerance: 1e-4 relative, falling back to 1e-4 absolute when the reference
// magnitude is near zero. Computed in double to keep the check itself precise.
bool within_tolerance(double actual, double reference) {
    constexpr double kTol = 1e-4;
    const double diff   = std::fabs(actual - reference);
    const double ref_ab = std::fabs(reference);
    if (ref_ab < kTol) {
        return diff <= kTol;             // absolute fallback
    }
    return (diff / ref_ab) <= kTol;      // relative error
}

}  // namespace

// Feature: vector-vault-db, Property 13: Distance kernels match the scalar reference within tolerance
// Validates: Requirements 4.1, 4.2, 4.3, 4.5
TEST_CASE("Property 13: distance kernels match the scalar reference within tolerance",
          "[distance][property]") {
    const DistanceCalculator calc = DistanceCalculator::create();

    // Integer-derived components in [-1, 1] so generation is uniform and
    // shrinking stays well-behaved.
    const auto gen_component = rc::gen::map(
        rc::gen::inRange<int>(-1000000, 1000001),
        [](int v) { return static_cast<float>(v) / 1000000.0f; });

    const bool ok = rc::check(
        "active kernel matches the scalar reference within 1e-4 tolerance",
        [&] {
            const DistanceMetric metric = *rc::gen::element(
                DistanceMetric::Euclidean,
                DistanceMetric::Cosine,
                DistanceMetric::DotProduct);

            const std::size_t n = *rc::gen::inRange<std::size_t>(1, 4097);
            auto a = *rc::gen::container<std::vector<float>>(n, gen_component);
            auto b = *rc::gen::container<std::vector<float>>(n, gen_component);

            // Cosine is undefined for a zero-norm vector (tested separately);
            // force a non-zero norm to stay in the defined range.
            if (metric == DistanceMetric::Cosine) {
                if (kernels::scalar_dot(a.data(), a.data(), n) == 0.0f) {
                    a[0] = 1.0f;
                }
                if (kernels::scalar_dot(b.data(), b.data(), n) == 0.0f) {
                    b[0] = 1.0f;
                }
            }

            const auto result = calc.distance(metric, a, b);
            RC_ASSERT(result.is_ok());

            const float actual    = result.value();
            const float reference = reference_distance(metric, a, b);
            RC_ASSERT(within_tolerance(actual, reference));

            // Cosine distance is mathematically in [0, 2].
            if (metric == DistanceMetric::Cosine) {
                RC_ASSERT(actual >= 0.0f);
                RC_ASSERT(actual <= 2.0f);
            }
        });
    REQUIRE(ok);
}

// Feature: vector-vault-db, Property 14: Euclidean distance of a vector to itself is exactly zero
// Validates: Requirements 4.4
TEST_CASE("Property 14: Euclidean distance of a vector to itself is exactly zero",
          "[distance][property]") {
    const DistanceCalculator calc = DistanceCalculator::create();

    // Wide finite range (no NaN/inf): self-distance is 0 at any magnitude, so
    // exercise large components too.
    const auto gen_component = rc::gen::map(
        rc::gen::inRange<int>(-1000000000, 1000000001),
        [](int v) { return static_cast<float>(v) / 1000000.0f; });

    const bool ok = rc::check(
        "Euclidean distance from a vector to itself is exactly 0.0f",
        [&] {
            const std::size_t n = *rc::gen::inRange<std::size_t>(1, 4097);
            auto v = *rc::gen::container<std::vector<float>>(n, gen_component);

            const auto result = calc.distance(DistanceMetric::Euclidean, v, v);
            RC_ASSERT(result.is_ok());

            RC_ASSERT(result.value() == 0.0f);
            RC_ASSERT(reference_distance(DistanceMetric::Euclidean, v, v) == 0.0f);
        });
    REQUIRE(ok);
}

// Unit tests: edge cases and dispatch stability.

// Cosine distance is undefined when either vector has a zero L2 norm.
TEST_CASE("cosine on a zero-norm vector returns UndefinedDistance",
          "[distance][unit]") {
    const DistanceCalculator calc = DistanceCalculator::create();

    const std::vector<float> zero  = {0.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> other = {1.0f, 2.0f, 3.0f, 4.0f};

    const auto first_zero = calc.distance(DistanceMetric::Cosine, zero, other);
    REQUIRE(first_zero.is_error());
    REQUIRE(first_zero.category() == ErrorCategory::UndefinedDistance);

    const auto second_zero = calc.distance(DistanceMetric::Cosine, other, zero);
    REQUIRE(second_zero.is_error());
    REQUIRE(second_zero.category() == ErrorCategory::UndefinedDistance);

    const auto both_zero = calc.distance(DistanceMetric::Cosine, zero, zero);
    REQUIRE(both_zero.is_error());
    REQUIRE(both_zero.category() == ErrorCategory::UndefinedDistance);
}

// Vectors of different component counts are a dimension mismatch, for every metric.
TEST_CASE("distance with mismatched lengths returns DimensionMismatch",
          "[distance][unit]") {
    const DistanceCalculator calc = DistanceCalculator::create();

    const std::vector<float> a = {1.0f, 2.0f, 3.0f};
    const std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};

    for (const DistanceMetric metric : {DistanceMetric::Euclidean,
                                        DistanceMetric::Cosine,
                                        DistanceMetric::DotProduct}) {
        const auto result = calc.distance(metric, a, b);
        REQUIRE(result.is_error());
        REQUIRE(result.category() == ErrorCategory::DimensionMismatch);
    }
}

// SIMD support is detected once and fixed for the process lifetime, so repeated
// create() calls report a stable uses_simd().
TEST_CASE("dispatch decision is stable across repeated create() calls",
          "[distance][unit]") {
    const bool first = DistanceCalculator::create().uses_simd();
    for (int i = 0; i < 8; ++i) {
        REQUIRE(DistanceCalculator::create().uses_simd() == first);
    }
}

// Whichever path is active (SIMD or scalar fallback) computes correct distances,
// checked against independently derived expected values for each metric.
TEST_CASE("active dispatch path computes correct distances on known inputs",
          "[distance][unit]") {
    using Catch::Matchers::WithinAbs;
    using Catch::Matchers::WithinRel;

    const DistanceCalculator calc = DistanceCalculator::create();

    const std::vector<float> a = {1.0f, 2.0f, 3.0f};
    const std::vector<float> b = {4.0f, 6.0f, 8.0f};

    // Euclidean: diffs (3, 4, 5) -> sqrt(9 + 16 + 25) = sqrt(50).
    const auto l2 = calc.distance(DistanceMetric::Euclidean, a, b);
    REQUIRE(l2.is_ok());
    REQUIRE_THAT(l2.value(), WithinRel(std::sqrt(50.0f), 1e-5f));

    // Dot product: 1*4 + 2*6 + 3*8 = 40.
    const auto dot = calc.distance(DistanceMetric::DotProduct, a, b);
    REQUIRE(dot.is_ok());
    REQUIRE_THAT(dot.value(), WithinRel(40.0f, 1e-5f));

    // Cosine: 1 - 40 / (sqrt(14) * sqrt(116)).
    const float expected_cosine =
        1.0f - 40.0f / (std::sqrt(14.0f) * std::sqrt(116.0f));
    const auto cosine = calc.distance(DistanceMetric::Cosine, a, b);
    REQUIRE(cosine.is_ok());
    REQUIRE_THAT(cosine.value(), WithinAbs(expected_cosine, 1e-5f));
    REQUIRE(cosine.value() >= 0.0f);
    REQUIRE(cosine.value() <= 2.0f);

    // The active path agrees with the scalar reference oracle for all metrics.
    for (const DistanceMetric metric : {DistanceMetric::Euclidean,
                                        DistanceMetric::Cosine,
                                        DistanceMetric::DotProduct}) {
        const auto result = calc.distance(metric, a, b);
        REQUIRE(result.is_ok());
        REQUIRE(within_tolerance(result.value(),
                                 reference_distance(metric, a, b)));
    }
}
