#pragma once

#include "../types.hpp"
#include "core.hpp"
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace dsbo::io {
namespace internal {

inline Array copy(ArrayView value) {
    Array result(value.type, value.shape);
    if (!result.storage.empty()) {
        if (!value.data) throw std::invalid_argument("Non-empty array has no data buffer.");
        std::memcpy(result.storage.data(), value.data, result.storage.size());
    }
    return result;
}

// Numeric promotion follows NumPy's array/array promotion, not C++ scalar rules.
inline DType promote(DType left, DType right) {
    left.little_endian = right.little_endian = native_little_endian();
    if (left.kind == Kind::boolean) return right;
    if (right.kind == Kind::boolean) return left;
    const bool complex = left.kind == Kind::complex || right.kind == Kind::complex;
    if (left.kind == Kind::complex) { left.kind = Kind::floating; left.size /= 2; }
    if (right.kind == Kind::complex) { right.kind = Kind::floating; right.size /= 2; }
    DType result;
    if (left.kind == Kind::floating || right.kind == Kind::floating) {
        const auto float_size = [](DType type) {
            if (type.kind == Kind::floating) return type.size;
            return type.size <= 1 ? std::size_t(2) : type.size <= 2 ? std::size_t(4) : std::size_t(8);
        };
        result = {Kind::floating, std::max(float_size(left), float_size(right))};
    } else if (left.kind == right.kind) result = {left.kind, std::max(left.size, right.size)};
    else {
        const auto sign = left.kind == Kind::signed_integer ? left.size : right.size;
        const auto unsign = left.kind == Kind::unsigned_integer ? left.size : right.size;
        if (sign > unsign) result = {Kind::signed_integer, sign};
        else if (unsign < 8) result = {Kind::signed_integer, unsign * 2};
        else result = {Kind::floating, 8};
    }
    if (complex) { result.kind = Kind::complex; result.size *= 2; }
    return result;
}

inline Array cast(ArrayView value, DType target) {
    if (value.type == target) return copy(value);
    if (value.type.kind == Kind::boolean) {
        auto normalized = copy(value);
        normalized.type = dtype<std::uint8_t>();
        for (auto& byte : normalized.storage) byte = byte != 0;
        return cast(normalized.view(), target);
    }
    if (target.kind == Kind::boolean) {
        const bool complex = value.type.kind == Kind::complex;
        const auto components = complex ? ArrayView{{Kind::floating, value.type.size / 2, value.type.little_endian},
                                                    {static_cast<hsize_t>(value.size() * 2)}, value.data} : value;
        const auto numbers = cast(components, dtype<long double>()).values<long double>();
        Array result(target, value.shape);
        for (std::size_t i = 0; i < value.size(); ++i)
            result.storage[i] = complex ? numbers[2 * i] != 0 || numbers[2 * i + 1] != 0 : numbers[i] != 0;
        return result;
    }
    if (value.type.kind == Kind::complex || target.kind == Kind::complex) {
        if (target.kind != Kind::complex) throw TypeError("Cannot discard an array's imaginary component.");
        const DType source_component{value.type.kind == Kind::complex ? Kind::floating : value.type.kind,
                                      value.type.kind == Kind::complex ? value.type.size / 2 : value.type.size,
                                      value.type.little_endian};
        ArrayView components{source_component, {static_cast<hsize_t>(value.size() * (value.type.kind == Kind::complex ? 2 : 1))}, value.data};
        auto converted = cast(components, {Kind::floating, target.size / 2, target.little_endian});
        Array result(target, value.shape);
        if (value.type.kind == Kind::complex) result.storage = std::move(converted.storage);
        else for (std::size_t i = 0; i < value.size(); ++i)
            std::memcpy(result.storage.data() + i * target.size, converted.storage.data() + i * target.size / 2, target.size / 2);
        return result;
    }
    auto source_type = datatype(value.type, false);
    auto target_type = datatype(target, false);
    std::vector<unsigned char> buffer(value.size() * std::max(value.type.size, target.size));
    if (!buffer.empty()) {
        std::memcpy(buffer.data(), value.data, value.nbytes());
        check(H5Tconvert(source_type, target_type, value.size(), buffer.data(), nullptr, H5P_DEFAULT), "Cannot convert numeric array dtype.");
    }
    Array result(target, value.shape);
    if (!result.storage.empty()) std::memcpy(result.storage.data(), buffer.data(), result.storage.size());
    return result;
}

inline Array stack(const std::vector<ArrayView>& arrays) {
    if (arrays.empty()) throw std::invalid_argument("need at least one array to stack");
    auto type = arrays.front().type;
    type.little_endian = native_little_endian();
    for (const auto& value : arrays) {
        if (value.shape != arrays.front().shape) throw std::invalid_argument("all input arrays must have the same shape");
        type = promote(type, value.type);
    }
    Shape shape{static_cast<hsize_t>(arrays.size())};
    shape.insert(shape.end(), arrays.front().shape.begin(), arrays.front().shape.end());
    Array result(type, std::move(shape));
    std::size_t offset = 0;
    for (const auto& value : arrays) {
        auto converted = cast(value, type);
        if (!converted.storage.empty()) std::memcpy(result.storage.data() + offset, converted.storage.data(), converted.storage.size());
        offset += converted.storage.size();
    }
    return result;
}

template<class T> inline void add_values(Array& left, const Array& right) {
    for (std::size_t i = 0; i < left.size(); ++i) {
        T a, b;
        std::memcpy(&a, left.storage.data() + i * sizeof(T), sizeof(T));
        std::memcpy(&b, right.storage.data() + i * sizeof(T), sizeof(T));
        if constexpr (std::is_integral_v<T>) {
            using Unsigned = std::make_unsigned_t<T>;
            const auto sum = static_cast<Unsigned>(a) + static_cast<Unsigned>(b);
            a = static_cast<T>(sum);
        } else a += b;
        std::memcpy(left.storage.data() + i * sizeof(T), &a, sizeof(T));
    }
}

inline Shape broadcast_shape(const Shape& left, const Shape& right) {
    Shape result(std::max(left.size(), right.size()), 1);
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto a = i < left.size() ? left[left.size() - i - 1] : 1;
        const auto b = i < right.size() ? right[right.size() - i - 1] : 1;
        if (a != b && a != 1 && b != 1) throw std::invalid_argument("Gradient arrays could not be broadcast together.");
        result[result.size() - i - 1] = a == 1 ? b : a;
    }
    return result;
}

inline Array broadcast(ArrayView value, const Shape& shape) {
    if (value.shape == shape) return copy(value);
    Array output(value.type, shape);
    const auto* bytes = static_cast<const unsigned char*>(value.data);
    for (std::size_t i = 0; i < output.size(); ++i) {
        std::size_t remaining = i, source = 0, stride = 1;
        for (std::size_t j = 0; j < shape.size(); ++j) {
            const auto coordinate = remaining % shape[shape.size() - j - 1];
            remaining /= shape[shape.size() - j - 1];
            if (j < value.shape.size()) {
                const auto width = value.shape[value.shape.size() - j - 1];
                if (width != 1) source += coordinate * stride;
                stride *= width;
            }
        }
        std::memcpy(output.storage.data() + i * value.type.size, bytes + source * value.type.size, value.type.size);
    }
    return output;
}

inline Array add(ArrayView left, ArrayView right) {
    const auto shape = broadcast_shape(left.shape, right.shape);
    const auto type = promote(left.type, right.type);
    auto result = cast(left, type);
    auto other = cast(right, type);
    if (result.shape != shape) result = broadcast(result.view(), shape);
    if (other.shape != shape) other = broadcast(other.view(), shape);
    if (type.kind == Kind::boolean) {
        for (std::size_t i = 0; i < result.size(); ++i) result.storage[i] = result.storage[i] || other.storage[i];
    } else if (type.kind == Kind::signed_integer || type.kind == Kind::unsigned_integer) {
        switch (type.size) {
            case 1: add_values<std::uint8_t>(result, other); break;
            case 2: add_values<std::uint16_t>(result, other); break;
            case 4: add_values<std::uint32_t>(result, other); break;
            case 8: add_values<std::uint64_t>(result, other); break;
            default: throw TypeError("Unsupported integer width.");
        }
    } else if (type.kind == Kind::complex) {
        if (type.size == 8) add_values<std::complex<float>>(result, other);
        else if (type.size == 16) add_values<std::complex<double>>(result, other);
        else add_values<std::complex<long double>>(result, other);
    } else if (type.size == 2) {
        auto a = cast(result.view(), dtype<float>()), b = cast(other.view(), dtype<float>());
        add_values<float>(a, b);
        return cast(a.view(), type);
    } else if (type.size == 4) add_values<float>(result, other);
    else if (type.size == 8) add_values<double>(result, other);
    else add_values<long double>(result, other);
    return result;
}
} // namespace internal

// Returns an owning array with the requested dtype, preserving the shape.
inline Array convert(ArrayView value, DType type) { return internal::cast(value, type); }

} // namespace dsbo::io
