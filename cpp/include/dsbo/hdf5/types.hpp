#pragma once

#include <hdf5.h>
#if H5_VERS_MAJOR < 2
#error "DSBO requires HDF5 >= 2.0"
#endif

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace dsbo::io {

struct IOError : std::runtime_error { using std::runtime_error::runtime_error; };
struct FileNotFoundError : IOError { using IOError::IOError; };
struct KeyError : std::runtime_error { using std::runtime_error::runtime_error; };
struct TypeError : std::runtime_error { using std::runtime_error::runtime_error; };

using Shape = std::vector<hsize_t>;
// nullopt means an unnamed write or the default stored selection on read.
// An explicit list supplies ordered names; an empty list is not a default.
using Columns = std::optional<std::vector<std::string>>;

inline std::size_t elements(const Shape& shape) {
    std::size_t count = 1;
    for (auto dimension : shape) {
        if (dimension && count > std::numeric_limits<std::size_t>::max() / dimension)
            throw std::overflow_error("Array shape exceeds addressable memory.");
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

enum class Kind { boolean, signed_integer, unsigned_integer, floating, complex };

inline bool native_little_endian() {
    const std::uint16_t value = 1;
    return *reinterpret_cast<const unsigned char*>(&value) == 1;
}

struct DType {
    Kind kind = Kind::floating;
    std::size_t size = sizeof(double);
    bool little_endian = native_little_endian();
    bool operator==(const DType& other) const {
        return kind == other.kind && size == other.size && (size == 1 || little_endian == other.little_endian);
    }
    bool operator!=(const DType& other) const { return !(*this == other); }
};

template<class T> inline DType dtype() {
    if constexpr (std::is_same_v<T, bool>) return {Kind::boolean, 1};
    else if constexpr (std::is_integral_v<T>)
        return {std::is_signed_v<T> ? Kind::signed_integer : Kind::unsigned_integer, sizeof(T)};
    else if constexpr (std::is_floating_point_v<T>) return {Kind::floating, sizeof(T)};
    else if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>> ||
                       std::is_same_v<T, std::complex<long double>>) return {Kind::complex, sizeof(T)};
    else static_assert(!sizeof(T), "Unsupported native array element type");
}

// Borrows contiguous, row-major storage for a synchronous call. The caller
// keeps the buffer alive and must not mutate it concurrently with that call.
struct ArrayView {
    DType type;
    Shape shape;
    const void* data = nullptr;
    std::size_t size() const { return elements(shape); }
    std::size_t nbytes() const {
        const auto count = size();
        if (!type.size || count > std::numeric_limits<std::size_t>::max() / type.size)
            throw std::overflow_error("Array byte size exceeds addressable memory.");
        return count * type.size;
    }
    // Selects axis 0, as used by named-column I/O; the result still borrows data.
    ArrayView column(std::size_t index) const {
        if (shape.empty() || index >= shape[0]) throw std::out_of_range("Array column is out of range.");
        Shape result(shape.begin() + 1, shape.end());
        const auto bytes = elements(result) * type.size;
        const auto* pointer = static_cast<const unsigned char*>(data);
        return {type, std::move(result), bytes ? pointer + index * bytes : pointer};
    }
};

// Owns contiguous bytes. view() borrows them; values<T>() returns a typed copy
// and requires an exact dtype match. Use convert() for numeric conversion.
struct Array {
    DType type;
    Shape shape;
    std::vector<unsigned char> storage;
    Array() = default;
    Array(DType type_, Shape shape_) : type(type_), shape(std::move(shape_)) {
        storage.resize(ArrayView{type, shape, nullptr}.nbytes());
    }
    ArrayView view() const { return {type, shape, storage.data()}; }
    std::size_t size() const { return elements(shape); }
    template<class T> static Array from(const std::vector<T>& values, Shape shape) {
        Array result(dtype<T>(), std::move(shape));
        if (values.size() != result.size()) throw std::invalid_argument("Array shape does not match its values.");
        if constexpr (std::is_same_v<T, bool>) {
            for (std::size_t i = 0; i < values.size(); ++i) result.storage[i] = values[i];
        } else if (!values.empty()) std::memcpy(result.storage.data(), values.data(), result.storage.size());
        return result;
    }
    template<class T> std::vector<T> values() const {
        if (type != dtype<T>()) throw TypeError("Requested values do not match the array dtype.");
        std::vector<T> result(size());
        if constexpr (std::is_same_v<T, bool>) {
            for (std::size_t i = 0; i < result.size(); ++i) result[i] = storage[i] != 0;
        } else if (!result.empty()) std::memcpy(result.data(), storage.data(), storage.size());
        return result;
    }
};

using NamedArrays = std::vector<std::pair<std::string, ArrayView>>;

// Axis-0 sample count and float64 summaries. For 1-D input the summaries are
// scalars; for 2-D input they have shape {input.shape[1]}. squared_deviation is
// M2 (the sum of squared deviations from the mean), not variance.
struct Statistics {
    std::uint64_t count = 0;
    Array mean;
    Array squared_deviation;
};

// A normalized selection along the sample axis. start is nonnegative for a
// nonempty selection; step may be negative but never zero. squeeze represents
// an integer index and requires count == 1. Python normalization is in bindings.
struct Selection {
    std::int64_t start = 0;
    std::int64_t step = 1;
    std::size_t count = 0;
    bool squeeze = false;
};

struct GradientContribution {
    std::string primal_path;
    // A gradient-file path for a physical input, or owned gradient values
    // after splitting a virtual input. columns gives their named-column order.
    std::variant<std::string, Array> source;
    Columns columns;
};

} // namespace dsbo::io
