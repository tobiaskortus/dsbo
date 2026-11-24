#pragma once

#include "../types.hpp"
#include "core.hpp"
#include <algorithm>
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
    if (result.nbytes()) {
        if (!value.data) throw std::invalid_argument("Non-empty array has no data buffer.");
        std::memcpy(result.data(), value.data, result.nbytes());
    }
    return result;
}

// Numeric promotion follows NumPy's array/array promotion, not C++ scalar rules.
inline DType promote(DType left, DType right) {
    if (left.kind == Kind::boolean) return right;
    if (right.kind == Kind::boolean) return left;
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
    return result;
}

inline Array cast(ArrayView value, DType target) {
    if (value.type == target) return copy(value);
    if (value.type.kind == Kind::boolean) {
        auto normalized = copy(value);
        normalized.type = dtype<std::uint8_t>();
        auto* bytes = normalized.data();
        for (std::size_t i = 0; i < normalized.nbytes(); ++i) bytes[i] = bytes[i] != 0;
        return cast(normalized.view(), target);
    }
    if (target.kind == Kind::boolean) {
        const auto numbers = cast(value, dtype<double>());
        const auto* data = numbers.data<double>();
        Array result(target, value.shape);
        auto* bytes = result.data();
        const auto count = result.size();
        for (std::size_t i = 0; i < count; ++i) bytes[i] = data[i] != 0;
        return result;
    }
    auto source_type = datatype(value.type);
    auto target_type = datatype(target);
    std::vector<unsigned char> buffer(value.size() * std::max(value.type.size, target.size));
    if (!buffer.empty()) {
        std::memcpy(buffer.data(), value.data, value.nbytes());
        check(H5Tconvert(source_type, target_type, value.size(), buffer.data(), nullptr, H5P_DEFAULT), "Cannot convert numeric array dtype.");
    }
    Array result(target, value.shape);
    if (result.nbytes()) std::memcpy(result.data(), buffer.data(), result.nbytes());
    return result;
}

inline Array stack(const std::vector<ArrayView>& arrays) {
    if (arrays.empty()) throw std::invalid_argument("need at least one array to stack");
    auto type = arrays.front().type;
    for (const auto& value : arrays) {
        if (value.shape != arrays.front().shape) throw std::invalid_argument("all input arrays must have the same shape");
        type = promote(type, value.type);
    }
    Shape shape{static_cast<hsize_t>(arrays.size())};
    shape.insert(shape.end(), arrays.front().shape.begin(), arrays.front().shape.end());
    Array result(type, std::move(shape));
    auto* output = result.data();
    std::size_t offset = 0;
    for (const auto& value : arrays) {
        auto source = value;
        std::optional<Array> converted;
        if (source.type != type) {
            converted = cast(source, type);
            source = converted->view();
        }
        const auto bytes = source.nbytes();
        if (bytes) {
            if (!source.data) throw std::invalid_argument("Non-empty array has no data buffer.");
            std::memcpy(output + offset, source.data, bytes);
        }
        offset += bytes;
    }
    return result;
}

template<class T> inline void add_values(Array& left, const Array& right) {
    auto* output = left.data();
    const auto* input = right.data();
    const auto count = left.size();
    for (std::size_t i = 0; i < count; ++i) {
        T a, b;
        std::memcpy(&a, output + i * sizeof(T), sizeof(T));
        std::memcpy(&b, input + i * sizeof(T), sizeof(T));
        if constexpr (std::is_integral_v<T>) {
            using Unsigned = std::make_unsigned_t<T>;
            const auto sum = static_cast<Unsigned>(a) + static_cast<Unsigned>(b);
            a = static_cast<T>(sum);
        } else a += b;
        std::memcpy(output + i * sizeof(T), &a, sizeof(T));
    }
}

inline Shape broadcast_shape(const Shape& left, const Shape& right) {
    Shape result(std::vector<hsize_t>(std::max(left.size(), right.size()), 1));
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
    auto* destination = output.data();
    const auto count = output.size();
    for (std::size_t i = 0; i < count; ++i) {
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
        std::memcpy(destination + i * value.type.size, bytes + source * value.type.size, value.type.size);
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
        auto* output = result.data();
        const auto* input = other.data();
        const auto count = result.size();
        for (std::size_t i = 0; i < count; ++i) output[i] = output[i] || input[i];
    } else if (type.kind == Kind::signed_integer || type.kind == Kind::unsigned_integer) {
        switch (type.size) {
            case 1: add_values<std::uint8_t>(result, other); break;
            case 2: add_values<std::uint16_t>(result, other); break;
            case 4: add_values<std::uint32_t>(result, other); break;
            case 8: add_values<std::uint64_t>(result, other); break;
            default: throw TypeError("Unsupported integer width.");
        }
    } else if (type.size == 2) {
        auto a = cast(result.view(), dtype<float>()), b = cast(other.view(), dtype<float>());
        add_values<float>(a, b);
        return cast(a.view(), type);
    } else if (type.size == 4) add_values<float>(result, other);
    else if (type.size == 8) add_values<double>(result, other);
    else throw TypeError("Unsupported floating-point width.");
    return result;
}
} // namespace internal

// Returns an owning array with the requested dtype, preserving the shape.
inline Array convert(ArrayView value, DType type) { return internal::cast(value, type); }

} // namespace dsbo::io
