#pragma once

#include "types.hpp"
#include "internal/arrays.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
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

struct ScaledSum {
    CompensatedSum parts;
    int exponent = 0;

    void add_scaled(double value, int value_exponent) {
        if (value_exponent > exponent) {
            parts.sum = std::ldexp(parts.sum, exponent - value_exponent);
            parts.correction = std::ldexp(parts.correction, exponent - value_exponent);
            exponent = value_exponent;
        }
        if (value_exponent != exponent) value = std::ldexp(value, value_exponent - exponent);
        const auto previous = parts;
        parts.add(value);
        if ((!std::isfinite(parts.sum) || !std::isfinite(parts.correction)) &&
            std::isfinite(previous.sum) && std::isfinite(previous.correction) && std::isfinite(value)) {
            // A partition may overflow before another partition cancels it.
            // Power-of-two scaling retains its finite sum for the later merge.
            parts = {previous.sum * 0.5, previous.correction * 0.5};
            ++exponent;
            parts.add(value * 0.5);
        }
    }

    void add(double value) { add_scaled(value, 0); }
    void merge(const ScaledSum& other) {
        add_scaled(other.parts.sum, other.exponent);
        add_scaled(other.parts.correction, other.exponent);
    }
    double divided_by(double divisor) const {
        return parts.divided_by(exponent ? std::ldexp(divisor, -exponent) : divisor);
    }
};

struct WelfordAccumulator {
    ScaledSum sum;
    CompensatedSum center, moment;

    void add(double sample, double origin, double weight) {
        sum.add(sample);
        if (!std::isfinite(sample)) {
            moment.add(std::numeric_limits<double>::quiet_NaN());
            return;
        }
        if (!std::isfinite(moment.sum)) return;
        const double centered = sample - origin;
        const double delta = centered - center.value();
        if (!std::isfinite(delta)) {
            moment.add(std::numeric_limits<double>::infinity());
            return;
        }
        center.add(delta * weight);
        moment.add(delta * (centered - center.value()));
    }

    void merge(const WelfordAccumulator& other, std::size_t count, std::size_t other_count) {
        sum.merge(other.sum);
        moment.add(other.moment.sum);
        moment.add(other.moment.correction);
        if (!std::isfinite(moment.value())) return;
        // All row partitions use the same origin; merge the small centered
        // means instead of subtracting rounded, large-offset absolute means.
        const double delta = other.center.value() - center.value();
        const double fraction = static_cast<double>(other_count) / static_cast<double>(count + other_count);
        center.add(delta * fraction);
        moment.add(delta * (delta * (static_cast<double>(count) * fraction)));
    }
};

template<class T> inline double load_statistic(const unsigned char* data) {
    T result;
    std::memcpy(&result, data, sizeof(T));
    return static_cast<double>(result);
}

inline void validate_statistics(const Statistics& value) {
    if (value.mean.shape != value.squared_deviation.shape)
        throw std::invalid_argument("HDF5 statistics mean and squared deviation shapes must match.");
}
} // namespace internal

struct StatisticsOptions {
    std::size_t workers = 0; // Automatic; DSBO_STATISTICS_THREADS overrides the default.
};

// Summarizes real numeric 1-D/2-D arrays along axis 0 with single-pass Welford updates.
// Rows are partitioned across workers; small inputs stay serial.
// Returns nullopt for other ranks or an empty sample axis.
inline std::optional<Statistics> statistics(ArrayView value, StatisticsOptions options = {}) {
    if ((value.shape.size() != 1 && value.shape.size() != 2) || value.shape[0] == 0)
        return std::nullopt;
    if (value.size() && !value.data) throw std::invalid_argument("Non-empty array has no data buffer.");
    const std::size_t rows = value.shape[0], width = value.shape.size() == 1 ? 1 : value.shape[1];
    const Shape dimensions = value.shape.size() == 1 ? Shape{} : Shape{width};
    std::vector<double> means(width), deviations(width);
    if (!width) return Statistics{rows, Array::from(means, dimensions), Array::from(deviations, dimensions)};

    std::size_t workers = options.workers;
    if (!workers) {
        if (const char* setting = std::getenv("DSBO_STATISTICS_THREADS")) {
            const auto end = setting + std::strlen(setting);
            const auto parsed = std::from_chars(setting, end, workers);
            if (parsed.ec != std::errc{} || parsed.ptr != end || !workers)
                throw std::invalid_argument("DSBO_STATISTICS_THREADS must be a positive integer.");
        } else {
            // Narrow rows have higher per-element cost (weight division per row);
            // wide rows pay a larger workers x width merge. Keep at least ~8K
            // elements per worker and amortize the merge over 32 elements/column.
            constexpr std::size_t base_elements = 1 << 13;
            const std::size_t merge_allowance = 32 * std::min<std::size_t>(width, 1 << 20);
            const std::size_t min_elements_per_worker = std::max(base_elements, merge_allowance);
            const std::size_t cores = std::max(1u, std::thread::hardware_concurrency());
            workers = std::min({cores, rows, std::max<std::size_t>(1, value.size() / min_elements_per_worker)});
        }
    }
    workers = std::max<std::size_t>(1, std::min(workers, rows));

    const bool native_float = value.type == dtype<float>(), native_double = value.type == dtype<double>();
    const std::size_t conversion_elements = std::max<std::size_t>(1, (1024 * 1024) / std::max(value.type.size, sizeof(double)));
    const auto* bytes = static_cast<const unsigned char*>(value.data);

    // Row 0 is the shared centering origin for every partition, so merging
    // combines small centered means instead of large rounded absolute means.
    std::vector<double> origins(width);
    for (std::size_t column = 0; column < width;) {
        const auto count = std::min(conversion_elements, width - column);
        if (native_float || native_double) {
            for (std::size_t offset = 0; offset < count; ++offset) {
                const auto* data = bytes + (column + offset) * value.type.size;
                origins[column + offset] = native_float ? internal::load_statistic<float>(data) : internal::load_statistic<double>(data);
            }
        } else {
            const auto converted = internal::cast({value.type, {count}, bytes + column * value.type.size}, dtype<double>());
            std::memcpy(origins.data() + column, converted.storage.data(), count * sizeof(double));
        }
        column += count;
    }

    struct PartitionAccumulator {
        std::size_t count = 0;
        std::vector<internal::WelfordAccumulator> columns;
    };
    std::vector<PartitionAccumulator> partition_accumulators(workers);
    std::vector<std::size_t> partition_starts(workers + 1);
    for (std::size_t index = 0; index <= workers; ++index)
        partition_starts[index] = (rows / workers) * index + std::min(rows % workers, index);

    const auto accumulate_partition = [&](PartitionAccumulator& partition,
                                          std::size_t row_begin, std::size_t row_end) {
        partition.count = row_end - row_begin;
        partition.columns.resize(width);
        if (native_float || native_double) {
            for (std::size_t row = row_begin; row < row_end; ++row) {
                const double weight = 1.0 / static_cast<double>(row - row_begin + 1);
                for (std::size_t column = 0; column < width; ++column) {
                    const auto* data = bytes + (row * width + column) * value.type.size;
                    partition.columns[column].add(native_float ? internal::load_statistic<float>(data)
                                                             : internal::load_statistic<double>(data),
                                                origins[column], weight);
                }
            }
        } else if (width <= conversion_elements) {
            // Convert contiguous row blocks; the buffer stays near 1 MiB.
            const std::size_t row_step = std::max<std::size_t>(1, conversion_elements / width);
            for (std::size_t first = row_begin; first < row_end;) {
                const std::size_t count = std::min(row_step, row_end - first);
                const auto converted = internal::cast(
                    {value.type, {count, width}, bytes + first * width * value.type.size}, dtype<double>());
                for (std::size_t row = 0; row < count; ++row) {
                    const double weight = 1.0 / static_cast<double>(first - row_begin + row + 1);
                    for (std::size_t column = 0; column < width; ++column)
                        partition.columns[column].add(
                            internal::load_statistic<double>(converted.storage.data() + (row * width + column) * sizeof(double)),
                            origins[column], weight);
                }
                first += count;
            }
        } else {
            // Extremely wide rows: convert bounded column chunks per row.
            for (std::size_t row = row_begin; row < row_end; ++row) {
                const double weight = 1.0 / static_cast<double>(row - row_begin + 1);
                for (std::size_t column = 0; column < width; column += conversion_elements) {
                    const std::size_t count = std::min(conversion_elements, width - column);
                    const auto converted = internal::cast(
                        {value.type, {count}, bytes + (row * width + column) * value.type.size}, dtype<double>());
                    for (std::size_t offset = 0; offset < count; ++offset)
                        partition.columns[column + offset].add(
                            internal::load_statistic<double>(converted.storage.data() + offset * sizeof(double)),
                            origins[column + offset], weight);
                }
            }
        }
    };

    std::exception_ptr failure;
    std::mutex failure_mutex;
    const auto partition_worker = [&](std::size_t partition_index) {
        try {
            accumulate_partition(partition_accumulators[partition_index],
                                 partition_starts[partition_index],
                                 partition_starts[partition_index + 1]);
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    try {
        for (std::size_t index = 1; index < workers; ++index) threads.emplace_back(partition_worker, index);
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    partition_worker(0);
    for (auto& thread : threads) thread.join();
    if (failure) std::rethrow_exception(failure);

    // Fixed partition order makes results independent of thread timing.
    auto& merged = partition_accumulators.front();
    for (std::size_t index = 1; index < workers; ++index) {
        auto& next_partition = partition_accumulators[index];
        for (std::size_t column = 0; column < width; ++column)
            merged.columns[column].merge(next_partition.columns[column],
                                         merged.count, next_partition.count);
        merged.count += next_partition.count;
    }
    for (std::size_t column = 0; column < width; ++column) {
        deviations[column] = merged.columns[column].moment.value();
        if (std::isfinite(deviations[column])) deviations[column] = std::max(0.0, deviations[column]);
        const double mean = merged.columns[column].sum.divided_by(static_cast<double>(rows));
        means[column] = !std::isfinite(mean) && std::isfinite(deviations[column])
            ? origins[column] + merged.columns[column].center.value() : mean;
    }
    return Statistics{rows, Array::from(means, dimensions), Array::from(deviations, dimensions)};
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
