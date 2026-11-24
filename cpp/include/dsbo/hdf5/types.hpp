#pragma once

#if defined(_WIN32)
#error "DSBO supports Linux and macOS only. On Windows, use WSL2 or a Linux container."
#endif

#include <hdf5.h>
#if H5_VERS_MAJOR < 2
#error "DSBO requires HDF5 >= 2.0"
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
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

// nullopt means an unnamed write or the default stored selection on read.
// An explicit list supplies ordered names; an empty list is not a default.
using Columns = std::optional<std::vector<std::string>>;

struct Shape : std::vector<hsize_t> {
private:
    struct Dim {
        hsize_t value;

        template<class Integer>
        using EnableSizeInteger = std::enable_if_t<
            std::is_integral_v<Integer> &&
            std::numeric_limits<Integer>::digits <= std::numeric_limits<hsize_t>::digits,
            int>;

        template<class Integer, EnableSizeInteger<Integer> = 0>
        Dim(Integer size) : value(static_cast<hsize_t>(size)) {
            if constexpr (std::is_signed_v<Integer>)
                if (size < 0) throw std::invalid_argument("Shape axis sizes must be non-negative.");
        }
        operator hsize_t() const { return value; }
    };

public:
    Shape() = default;
    Shape(std::initializer_list<Dim> sizes) : Shape(sizes.begin(), sizes.end()) {}
    // Preserve typed hsize_t lists without forcing braced ints to narrow.
    template<class Integer, std::enable_if_t<std::is_same_v<Integer, hsize_t>, int> = 0>
    Shape(std::initializer_list<Integer> sizes) : Shape(sizes.begin(), sizes.end()) {}
    Shape(const std::vector<hsize_t>& values) : std::vector<hsize_t>(values) {}
    Shape(std::vector<hsize_t>&& values) : std::vector<hsize_t>(std::move(values)) {}
    template<class Iterator, std::enable_if_t<!std::is_integral_v<Iterator>, int> = 0>
    Shape(Iterator first, Iterator last) : std::vector<hsize_t>(first, last) {}
};

inline std::size_t elements(const Shape& shape) {
    std::size_t count = 1;
    for (auto dimension : shape) {
        if (dimension && count > std::numeric_limits<std::size_t>::max() / dimension)
            throw std::overflow_error("Array shape exceeds addressable memory.");
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

enum class Kind { boolean, signed_integer, unsigned_integer, floating };

struct DType {
    Kind kind = Kind::floating;
    std::size_t size = sizeof(double);
    bool operator==(const DType& other) const { return kind == other.kind && size == other.size; }
    bool operator!=(const DType& other) const { return !(*this == other); }
};

template<class T> inline DType dtype() {
    if constexpr (std::is_same_v<T, bool>) return {Kind::boolean, 1};
    else if constexpr (std::is_integral_v<T>) {
        static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                      "DSBO supports only 8-, 16-, 32-, and 64-bit integers.");
        return {std::is_signed_v<T> ? Kind::signed_integer : Kind::unsigned_integer, sizeof(T)};
    } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
        return {Kind::floating, sizeof(T)};
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

namespace internal {
// Element types the typed API accepts without conversion.
template<class T> inline constexpr bool supported_element =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    (std::is_integral_v<T> && (sizeof(T) == 1 || sizeof(T) == 2 ||
                               sizeof(T) == 4 || sizeof(T) == 8));

template<class T, class Count>
inline constexpr bool valid_view_args =
    supported_element<std::remove_cv_t<T>> && !std::is_volatile_v<T> &&
    std::is_integral_v<Count> && !std::is_same_v<Count, bool> &&
    sizeof(Count) <= sizeof(hsize_t);

enum class Initialization { zero, uninitialized };

using ArrayStorage = std::variant<
    std::unique_ptr<std::uint8_t[]>, std::unique_ptr<std::int8_t[]>,
    std::unique_ptr<std::int16_t[]>, std::unique_ptr<std::uint16_t[]>,
    std::unique_ptr<std::int32_t[]>, std::unique_ptr<std::uint32_t[]>,
    std::unique_ptr<std::int64_t[]>, std::unique_ptr<std::uint64_t[]>,
    std::unique_ptr<float[]>, std::unique_ptr<double[]>>;
} // namespace internal

struct Array {
    DType type;
    Shape shape;
    Array() = default;

    Array(DType type_, Shape shape_, internal::Initialization initialization = internal::Initialization::zero)
        : type(type_), shape(std::move(shape_)) {
        const auto count = elements(shape);
        if (!type.size || count > std::numeric_limits<std::size_t>::max() / type.size)
            throw std::overflow_error("Array byte size exceeds addressable memory.");
        byte_size_ = count * type.size;
        if (type == dtype<std::int8_t>()) allocate<std::int8_t>(count);
        else if (type == dtype<std::uint8_t>()) allocate<std::uint8_t>(count);
        else if (type == dtype<std::int16_t>()) allocate<std::int16_t>(count);
        else if (type == dtype<std::uint16_t>()) allocate<std::uint16_t>(count);
        else if (type == dtype<std::int32_t>()) allocate<std::int32_t>(count);
        else if (type == dtype<std::uint32_t>()) allocate<std::uint32_t>(count);
        else if (type == dtype<std::int64_t>()) allocate<std::int64_t>(count);
        else if (type == dtype<std::uint64_t>()) allocate<std::uint64_t>(count);
        else if (type == dtype<float>()) allocate<float>(count);
        else if (type == dtype<double>()) allocate<double>(count);
        else allocate<std::uint8_t>(byte_size_);
        if (byte_size_ && initialization == internal::Initialization::zero)
            std::memset(data(), 0, byte_size_);
    }
    Array(const Array& other) : type(other.type), shape(other.shape), byte_size_(other.byte_size_) {
        std::visit([&](const auto& buffer) {
            using T = typename std::decay_t<decltype(buffer)>::element_type;
            allocate<T>(byte_size_ / sizeof(T));
        }, other.storage_);
        if (byte_size_) std::memcpy(data(), other.data(), byte_size_);
    }
    Array(Array&& other) noexcept
        : type(other.type), shape(std::move(other.shape)), storage_(std::move(other.storage_)),
          byte_size_(std::exchange(other.byte_size_, 0)) {}

    Array& operator=(Array other) noexcept {
        type = other.type;
        shape = std::move(other.shape);
        storage_ = std::move(other.storage_);
        byte_size_ = std::exchange(other.byte_size_, 0);
        return *this;
    }
    ArrayView view() const { return {type, shape, data()}; }

    std::size_t size() const { return elements(shape); }

    std::size_t nbytes() const { return byte_size_; }

    const unsigned char* data() const {
        return std::visit([](const auto& buffer) {
            return reinterpret_cast<const unsigned char*>(buffer.get());
        }, storage_);
    }

    unsigned char* data() { return const_cast<unsigned char*>(std::as_const(*this).data()); }

    template<class T> const T* data() const {
        static_assert(internal::supported_element<T> && std::is_same_v<T, std::remove_cv_t<T>>,
                      "Array::data<T>() requires an unqualified native element type.");

        if (type != dtype<T>())
            throw TypeError("Requested data do not match the array dtype.");

        const auto count = size();
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T) || byte_size_ != count * sizeof(T))
            throw TypeError("Array shape does not match its storage byte size.");

        return std::visit([](const auto& buffer) -> const T* {
            if constexpr (std::is_same_v<std::decay_t<decltype(buffer)>, std::unique_ptr<T[]>>)
                return buffer.get();
            else throw TypeError("Requested C++ type does not match the array's canonical storage type.");
        }, storage_);
    }
    template<class T> T* data() { return const_cast<T*>(std::as_const(*this).template data<T>()); }

    template<class T> static Array from(const std::vector<T>& values, Shape shape) {
        if (values.size() != elements(shape)) throw std::invalid_argument("Array shape does not match its values.");
        Array result(dtype<T>(), std::move(shape), internal::Initialization::uninitialized);
        if constexpr (std::is_same_v<T, bool>) {
            auto* bytes = result.data();
            for (std::size_t i = 0; i < values.size(); ++i) bytes[i] = values[i];
        } else if (!values.empty()) std::memcpy(result.data(), values.data(), result.nbytes());
        return result;
    }
private:
    internal::ArrayStorage storage_;
    std::size_t byte_size_ = 0;
    template<class T> void allocate(std::size_t count) {
        storage_ = std::unique_ptr<T[]>(count ? new T[count] : nullptr);
    }
};

template<class T, class Count, std::enable_if_t<internal::valid_view_args<T, Count>, int> = 0>
ArrayView view_of(const T* data, Count count, Shape shape) {
    const auto size = static_cast<std::size_t>(static_cast<hsize_t>(count));
    if (elements(shape) != size)
        throw std::invalid_argument("Array shape does not match its buffer size.");
    return {dtype<std::remove_cv_t<T>>(), std::move(shape), data};
}

// Deduce Count so a braced shape ({3} or {}) cannot bind as a pointer count
// when the source is a C array.
template<class T, class Count, std::enable_if_t<internal::valid_view_args<T, Count>, int> = 0>
ArrayView view_of(const T* data, Count count) {
    return view_of(data, count, Shape{static_cast<hsize_t>(count)});
}

template<class Buffer>
auto view_of(const Buffer& values, Shape shape)
    -> decltype(view_of(std::data(values), std::size(values), std::move(shape))) {
    return view_of(std::data(values), std::size(values), std::move(shape));
}

template<class Buffer>
auto view_of(const Buffer& values) -> decltype(view_of(std::data(values), std::size(values))) {
    return view_of(std::data(values), std::size(values));
}

inline ArrayView view_of(ArrayView value) { return value; }
inline ArrayView view_of(const Array& value) { return value.view(); }
ArrayView view_of(const Array&, Shape) = delete;

using NamedArrays = std::vector<std::pair<std::string, ArrayView>>;
using NamedArrayValues = std::vector<std::pair<std::string, Array>>;

struct NamedArrayViews {
    std::unique_ptr<unsigned char[]> storage;
    std::vector<std::pair<std::string, ArrayView>> views;
};

inline NamedArrays::value_type field(std::string name, ArrayView value) {
    return {std::move(name), std::move(value)};
}

template<class Buffer>
auto field(std::string name, const Buffer& values)
    -> decltype(view_of(values), NamedArrays::value_type{}) {
    return field(std::move(name), view_of(values));
}

template<class Buffer>
auto field(std::string name, const Buffer& values, Shape shape)
    -> decltype(view_of(values, std::move(shape)), NamedArrays::value_type{}) {
    return field(std::move(name), view_of(values, std::move(shape)));
}

// Looks up an owning named array while preserving the stored or requested order.
inline Array& at(NamedArrayValues& values, const std::string& key) {
    for (auto& [name, value] : values)
        if (name == key) return value;
    throw KeyError("No named array for key '" + key + "'.");
}
inline const Array& at(const NamedArrayValues& values, const std::string& key) {
    for (const auto& [name, value] : values)
        if (name == key) return value;
    throw KeyError("No named array for key '" + key + "'.");
}

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
    // after splitting a virtual input. keys gives their TensorDict key order.
    std::variant<std::string, Array, NamedArrayValues> source;
    std::vector<std::string> keys;
};

} // namespace dsbo::io
