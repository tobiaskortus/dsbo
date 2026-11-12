#pragma once

#include "types.hpp"
#include "internal/arrays.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace dsbo::io {
namespace internal {

struct CompensatedSum {
    double sum = 0;
    double correction = 0;

    void add(double value) {
        const double next = sum + value;
        if (!std::isfinite(next)) {
            // Do not let compensation turn infinities into NaNs through inf - inf.
            sum = next;
            correction = 0;
            return;
        }
        // Neumaier summation also recovers small terms when a later value
        // cancels the running sum (unlike ordinary Kahan summation).
        correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
        sum = next;
    }

    double value() const { return sum + correction; }

    double divided_by(double divisor) const {
        const double quotient = sum / divisor;
        if (!std::isfinite(sum)) return quotient;
        // Preserve the low part through division instead of rounding the total
        // first, which can shift even a constant-data mean by one ULP.
        return quotient + (std::fma(-quotient, divisor, sum) + correction) / divisor;
    }
};

inline void validate_statistics(const Statistics& value) {
    if (value.mean.shape != value.squared_deviation.shape)
        throw std::invalid_argument("HDF5 statistics mean and squared deviation shapes must match.");
}
} // namespace internal

// Summarizes real numeric 1-D/2-D arrays along axis 0 using float64 values.
// Returns nullopt for other ranks, complex input, or an empty sample axis.
inline std::optional<Statistics> statistics(ArrayView value) {
    if ((value.shape.size() != 1 && value.shape.size() != 2) || value.shape[0] == 0 || value.type.kind == Kind::complex)
        return std::nullopt;
    if (value.size() && !value.data) throw std::invalid_argument("Non-empty array has no data buffer.");
    std::optional<Array> converted;
    if (value.type != dtype<double>()) {
        converted = internal::cast(value, dtype<double>());
        value = converted->view();
    }
    const auto read = [&](std::size_t index) {
        double result;
        std::memcpy(&result, static_cast<const unsigned char*>(value.data) + index * sizeof(double), sizeof(double));
        return result;
    };
    const std::size_t width = value.shape.size() == 1 ? 1 : value.shape[1];
    Shape dimensions = value.shape.size() == 1 ? Shape{} : Shape{value.shape[1]};
    std::vector<double> means(width, 0), deviations(width, 0);
    std::vector<internal::CompensatedSum> sums(width), residual_sums(width);
    for (std::size_t row = 0; row < value.shape[0]; ++row)
        for (std::size_t column = 0; column < width; ++column) sums[column].add(read(row * width + column));
    for (std::size_t column = 0; column < width; ++column) {
        means[column] = sums[column].divided_by(static_cast<double>(value.shape[0]));
        sums[column] = {};
    }
    // Keep a centered second pass: E[x*x] - E[x]*E[x] loses small variances
    // for large-offset data, even when its reductions are compensated.
    for (std::size_t row = 0; row < value.shape[0]; ++row)
        for (std::size_t column = 0; column < width; ++column) {
            const double delta = read(row * width + column) - means[column];
            residual_sums[column].add(delta);
            sums[column].add(delta * delta);
        }
    for (std::size_t column = 0; column < width; ++column) {
        deviations[column] = sums[column].value();
        const double residual = residual_sums[column].value();
        // Correct for rounding the mean, especially when the spread is near
        // one ULP of a large offset. Clamp only finite roundoff, not NaNs/infs.
        if (std::isfinite(deviations[column]) && std::isfinite(residual))
            deviations[column] = std::max(0.0, deviations[column] - residual * (residual / value.shape[0]));
    }
    return Statistics{value.shape[0], Array::from(means, dimensions), Array::from(deviations, dimensions)};
}

// Combines float64 summaries with matching mean/M2 shapes. Empty input or any
// missing summary yields nullopt; incompatible shapes raise invalid_argument.
inline std::optional<Statistics> combine_statistics(const std::vector<std::optional<Statistics>>& values) {
    if (values.empty() || std::any_of(values.begin(), values.end(), [](const auto& value) { return !value; })) return std::nullopt;
    for (const auto& value : values) internal::validate_statistics(*value);
    auto result = *values.front();
    auto mean = result.mean.values<double>(), deviation = result.squared_deviation.values<double>();
    for (std::size_t i = 1; i < values.size(); ++i) {
        const auto& batch = *values[i];
        if (batch.mean.shape != result.mean.shape) throw std::invalid_argument("Incompatible HDF5 statistics shapes.");
        const auto batch_mean = batch.mean.values<double>(), batch_deviation = batch.squared_deviation.values<double>();
        const auto total = result.count + batch.count;
        if (!total) return std::nullopt;
        for (std::size_t j = 0; j < mean.size(); ++j) {
            const double delta = batch_mean[j] - mean[j];
            mean[j] += delta * static_cast<double>(batch.count) / total;
            deviation[j] += batch_deviation[j] + delta * delta * static_cast<double>(result.count) * batch.count / total;
        }
        result.count = total;
    }
    result.mean = Array::from(mean, result.mean.shape);
    result.squared_deviation = Array::from(deviation, result.squared_deviation.shape);
    return result;
}

} // namespace dsbo::io
