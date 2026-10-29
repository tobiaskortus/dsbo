#pragma once

// DSBO's native storage SDK. Requires C++17 and thread-safe HDF5 >= 2.0.
// No Python, NumPy, or HDF5 C++ wrapper dependency.
#include <hdf5.h>
#if H5_VERS_MAJOR < 2
#error "DSBO requires HDF5 >= 2.0"
#endif

#include "third_party/json.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <complex>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace dsbo::io {

using Shape = std::vector<hsize_t>;
using Columns = std::optional<std::vector<std::string>>;

inline constexpr char store_attribute[] = "dsbo_store";
inline constexpr char columns_attribute[] = "dsbo_columns";
inline constexpr char row_store_dataset[] = "__dsbo_row_store__";
inline constexpr char keys_attribute[] = "dsbo_keys";
inline constexpr char sources_attribute[] = "dsbo_vds_sources";
inline constexpr char partitions_attribute[] = "dsbo_partition_sizes";
inline constexpr char count_attribute[] = "dsbo_stats_count";
inline constexpr char mean_attribute[] = "dsbo_stats_mean";
inline constexpr char deviation_attribute[] = "dsbo_stats_squared_deviation";

struct IOError : std::runtime_error { using std::runtime_error::runtime_error; };
struct FileNotFoundError : IOError { using IOError::IOError; };
struct KeyError : std::runtime_error { using std::runtime_error::runtime_error; };
struct TypeError : std::runtime_error { using std::runtime_error::runtime_error; };

// Validates the actual linked library, not merely the headers used to compile.
inline void check_environment() {
    static const bool checked = [] {
        unsigned major = 0, minor = 0, patch = 0;
        bool threadsafe = false;
        if (H5get_libversion(&major, &minor, &patch) < 0 || major < 2)
            throw IOError("DSBO requires HDF5 >= 2.0 at runtime.");
        if (H5is_library_threadsafe(&threadsafe) < 0 || !threadsafe)
            throw IOError("DSBO requires a thread-safe HDF5 installation. Rebuild HDF5 with HDF5_ENABLE_THREADSAFE=ON.");
        return true;
    }();
    (void)checked;
}

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
    ArrayView column(std::size_t index) const {
        if (shape.empty() || index >= shape[0]) throw std::out_of_range("Array column is out of range.");
        Shape result(shape.begin() + 1, shape.end());
        const auto bytes = elements(result) * type.size;
        const auto* pointer = static_cast<const unsigned char*>(data);
        return {type, std::move(result), bytes ? pointer + index * bytes : pointer};
    }
};

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

namespace detail {
namespace fs = std::filesystem;
using Json = nlohmann::json;

inline void check(herr_t status, const std::string& message) {
    if (status < 0) throw IOError(message);
}

class Handle {
    hid_t id_ = -1;
    herr_t (*close_)(hid_t) = nullptr;
public:
    Handle() = default;
    Handle(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {
        if (id < 0) throw IOError("HDF5 operation failed.");
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& other) noexcept : id_(std::exchange(other.id_, -1)), close_(other.close_) {}
    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) { reset(); id_ = std::exchange(other.id_, -1); close_ = other.close_; }
        return *this;
    }
    ~Handle() { reset(); }
    void close_checked() {
        if (id_ >= 0 && close_) {
            check(close_(id_), "Cannot close HDF5 resource.");
            id_ = -1;
        }
    }
    void reset() noexcept {
        if (id_ >= 0 && close_) {
            H5E_BEGIN_TRY { close_(id_); } H5E_END_TRY;
        }
        id_ = -1;
    }
    operator hid_t() const { return id_; }
    explicit operator bool() const { return id_ >= 0; }
};

inline bool exists(hid_t parent, const std::string& name) {
    htri_t result;
    H5E_BEGIN_TRY { result = H5Lexists(parent, name.c_str(), H5P_DEFAULT); } H5E_END_TRY;
    return result > 0;
}

inline H5O_type_t object_type(hid_t parent, const std::string& name) {
    if (!exists(parent, name)) throw KeyError("HDF5 object '" + name + "' does not exist.");
    H5O_info2_t info{};
    check(H5Oget_info_by_name3(parent, name.c_str(), &info, H5O_INFO_BASIC, H5P_DEFAULT), "Cannot inspect HDF5 object '" + name + "'.");
    return info.type;
}

inline Handle dataset(hid_t parent, const std::string& name) {
    if (object_type(parent, name) != H5O_TYPE_DATASET) throw std::invalid_argument("HDF5 object '" + name + "' is not a dataset.");
    return {H5Dopen2(parent, name.c_str(), H5P_DEFAULT), H5Dclose};
}

inline Handle group(hid_t parent, const std::string& name) {
    if (name != "/" && object_type(parent, name) != H5O_TYPE_GROUP)
        throw std::invalid_argument("HDF5 object '" + name + "' is not a group.");
    return {H5Gopen2(parent, name.c_str(), H5P_DEFAULT), H5Gclose};
}

inline Handle require_group(hid_t parent, const std::string& name) {
    if (name.empty() || name == "/") return group(parent, "/");
    if (exists(parent, name)) return group(parent, name);
    Handle links(H5Pcreate(H5P_LINK_CREATE), H5Pclose);
    check(H5Pset_create_intermediate_group(links, 1), "Cannot enable intermediate groups.");
    check(H5Pset_char_encoding(links, H5T_CSET_UTF8), "Cannot enable UTF-8 group names.");
    Handle properties(H5Pcreate(H5P_GROUP_CREATE), H5Pclose);
    check(H5Pset_link_creation_order(properties, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED), "Cannot track column order.");
    return {H5Gcreate2(parent, name.c_str(), links, properties, H5P_DEFAULT), H5Gclose};
}

inline Handle open_file(const std::string& path, bool write = false, bool append = false) {
    check_environment();
    if (!write && !fs::is_regular_file(path)) throw FileNotFoundError("Missing HDF5 file: " + path);
    if (write) {
        const auto directory = fs::path(path).parent_path();
        if (!directory.empty()) fs::create_directories(directory);
    }
    Handle access(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
    check(H5Pset_libver_bounds(access, H5F_LIBVER_V200, H5F_LIBVER_V200), "Cannot set HDF5 format bounds.");
    hid_t id;
    H5E_BEGIN_TRY {
        id = write && !(append && fs::exists(path))
            ? H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, access)
            : H5Fopen(path.c_str(), write ? H5F_ACC_RDWR : H5F_ACC_RDONLY, access);
    } H5E_END_TRY;
    if (id < 0) throw IOError("Cannot open HDF5 file '" + path + "'.");
    return {id, H5Fclose};
}

inline void finish_file(Handle& file) {
    check(H5Fflush(file, H5F_SCOPE_GLOBAL), "Cannot flush HDF5 output.");
    file.close_checked();
}

inline Shape shape(hid_t space) {
    const int rank = H5Sget_simple_extent_ndims(space);
    if (rank < 0) throw IOError("Cannot inspect HDF5 shape.");
    Shape result(static_cast<std::size_t>(rank));
    check(H5Sget_simple_extent_dims(space, result.data(), nullptr), "Cannot inspect HDF5 dimensions.");
    return result;
}

inline Shape dataset_shape(hid_t value) {
    Handle space(H5Dget_space(value), H5Sclose);
    return shape(space);
}

inline Handle dataspace(const Shape& dimensions) {
    return {dimensions.empty() ? H5Screate(H5S_SCALAR)
                              : H5Screate_simple(static_cast<int>(dimensions.size()), dimensions.data(), nullptr), H5Sclose};
}

inline Handle datatype(DType type, bool boolean_enum = true) {
    check_environment();
    hid_t base = -1;
    const bool little = type.little_endian;
    if (type.kind == Kind::complex) {
        if (type.size != 8 && type.size != 16 && type.size != 2 * sizeof(long double))
            throw TypeError("Unsupported complex dtype.");
        auto component = datatype({Kind::floating, type.size / 2, little});
        Handle compound(H5Tcreate(H5T_COMPOUND, type.size), H5Tclose);
        check(H5Tinsert(compound, "r", 0, component), "Cannot create complex real field.");
        check(H5Tinsert(compound, "i", type.size / 2, component), "Cannot create complex imaginary field.");
        return compound;
    }
    if (type.kind == Kind::floating) {
        if (type.size == 2) base = little ? H5T_IEEE_F16LE : H5T_IEEE_F16BE;
        else if (type.size == 4) base = little ? H5T_IEEE_F32LE : H5T_IEEE_F32BE;
        else if (type.size == 8) base = little ? H5T_IEEE_F64LE : H5T_IEEE_F64BE;
        else if (type.size == sizeof(long double)) base = H5T_NATIVE_LDOUBLE;
    } else {
        const bool sign = type.kind == Kind::signed_integer;
        if (type.size == 1) base = sign ? H5T_STD_I8LE : H5T_STD_U8LE;
        else if (type.size == 2) base = sign ? (little ? H5T_STD_I16LE : H5T_STD_I16BE) : (little ? H5T_STD_U16LE : H5T_STD_U16BE);
        else if (type.size == 4) base = sign ? (little ? H5T_STD_I32LE : H5T_STD_I32BE) : (little ? H5T_STD_U32LE : H5T_STD_U32BE);
        else if (type.size == 8) base = sign ? (little ? H5T_STD_I64LE : H5T_STD_I64BE) : (little ? H5T_STD_U64LE : H5T_STD_U64BE);
    }
    if (base < 0) throw TypeError("Unsupported numeric dtype.");
    Handle result(H5Tcopy(base), H5Tclose);
    if (type.kind == Kind::floating && type.size > 8)
        check(H5Tset_order(result, little ? H5T_ORDER_LE : H5T_ORDER_BE), "Cannot set extended float byte order.");
    if (type.kind == Kind::boolean && boolean_enum) {
        Handle enumeration(H5Tenum_create(result), H5Tclose);
        const unsigned char no = 0, yes = 1;
        check(H5Tenum_insert(enumeration, "FALSE", &no), "Cannot create boolean dtype.");
        check(H5Tenum_insert(enumeration, "TRUE", &yes), "Cannot create boolean dtype.");
        return enumeration;
    }
    return result;
}

inline DType dtype_from_hdf5(hid_t type) {
    const auto size = H5Tget_size(type);
    const auto kind = H5Tget_class(type);
    const bool little = H5Tget_order(type) != H5T_ORDER_BE;
    if (kind == H5T_INTEGER) return {H5Tget_sign(type) == H5T_SGN_NONE ? Kind::unsigned_integer : Kind::signed_integer, size, little};
    if (kind == H5T_FLOAT) return {Kind::floating, size, little};
    if (kind == H5T_ENUM) {
        if (size == 1 && H5Tget_nmembers(type) == 2) {
            unsigned char no = 2, yes = 2;
            if (H5Tenum_valueof(type, "FALSE", &no) >= 0 && H5Tenum_valueof(type, "TRUE", &yes) >= 0 && no == 0 && yes == 1)
                return {Kind::boolean, 1};
        }
        Handle super(H5Tget_super(type), H5Tclose);
        return dtype_from_hdf5(super);
    }
    if (kind == H5T_COMPOUND && H5Tget_nmembers(type) == 2) {
        char* real = H5Tget_member_name(type, 0);
        char* imag = H5Tget_member_name(type, 1);
        const bool names = real && imag && std::string(real) == "r" && std::string(imag) == "i";
        if (real) H5free_memory(real);
        if (imag) H5free_memory(imag);
        Handle component(H5Tget_member_type(type, 0), H5Tclose);
        Handle second(H5Tget_member_type(type, 1), H5Tclose);
        if (names && H5Tequal(component, second) > 0 && H5Tget_class(component) == H5T_FLOAT &&
            H5Tget_member_offset(type, 0) == 0 && H5Tget_member_offset(type, 1) == size / 2 && H5Tget_size(component) * 2 == size)
            return {Kind::complex, size, H5Tget_order(component) != H5T_ORDER_BE};
    }
    if (kind == H5T_COMPLEX) {
        Handle component(H5Tget_super(type), H5Tclose);
        return {Kind::complex, size, H5Tget_order(component) != H5T_ORDER_BE};
    }
    throw TypeError("DSBO arrays require boolean, integer, floating-point, or complex numeric values.");
}

inline DType dataset_dtype(hid_t value) {
    Handle type(H5Dget_type(value), H5Tclose);
    return dtype_from_hdf5(type);
}

inline std::optional<std::string> string_attribute(hid_t parent, const char* name) {
    if (H5Aexists(parent, name) <= 0) return std::nullopt;
    Handle attribute(H5Aopen(parent, name, H5P_DEFAULT), H5Aclose);
    Handle type(H5Aget_type(attribute), H5Tclose);
    if (H5Tget_class(type) != H5T_STRING) throw std::invalid_argument(std::string("Invalid HDF5 string attribute: ") + name);
    if (H5Tis_variable_str(type) > 0) {
        char* value = nullptr;
        check(H5Aread(attribute, type, &value), "Cannot read HDF5 string attribute.");
        std::string result = value ? value : "";
        if (value) H5free_memory(value);
        return result;
    }
    std::vector<char> value(H5Tget_size(type) + 1, '\0');
    check(H5Aread(attribute, type, value.data()), "Cannot read HDF5 string attribute.");
    return std::string(value.data());
}

inline void set_string(hid_t parent, const char* name, const std::string& value) {
    if (H5Aexists(parent, name) > 0) check(H5Adelete(parent, name), "Cannot replace HDF5 attribute.");
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    check(H5Tset_size(type, H5T_VARIABLE), "Cannot set string size.");
    check(H5Tset_cset(type, H5T_CSET_UTF8), "Cannot enable UTF-8 metadata.");
    auto space = dataspace({});
    Handle attribute(H5Acreate2(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    const char* pointer = value.c_str();
    check(H5Awrite(attribute, type, &pointer), "Cannot write HDF5 string attribute.");
}

template<class T> inline std::optional<T> json_attribute(hid_t parent, const char* name) {
    const auto value = string_attribute(parent, name);
    if (!value) return std::nullopt;
    try { return Json::parse(*value).template get<T>(); }
    catch (const Json::exception& error) { throw std::invalid_argument(std::string("Invalid HDF5 metadata '") + name + "': " + error.what()); }
}

inline std::vector<std::string> children(hid_t parent, bool stored_order = false) {
    auto root = group(parent, "/");
    // For a group identifier, "." refers to the group itself, not the file root.
    if (H5Iget_type(parent) == H5I_GROUP) root = Handle(H5Gopen2(parent, ".", H5P_DEFAULT), H5Gclose);
    H5_index_t index = H5_INDEX_NAME;
    if (stored_order) {
        Handle properties(H5Gget_create_plist(root), H5Pclose);
        unsigned flags = 0;
        check(H5Pget_link_creation_order(properties, &flags), "Cannot inspect HDF5 group order.");
        if (flags & H5P_CRT_ORDER_TRACKED) index = H5_INDEX_CRT_ORDER;
    }
    struct State { std::vector<std::string> names; std::exception_ptr error; } state;
    const auto callback = [](hid_t, const char* name, const H5L_info2_t*, void* pointer) -> herr_t {
        auto& state = *static_cast<State*>(pointer);
        try { state.names.emplace_back(name); return 0; }
        catch (...) { state.error = std::current_exception(); return -1; }
    };
    const auto status = H5Literate2(root, index, H5_ITER_INC, nullptr, callback, &state);
    if (state.error) std::rethrow_exception(state.error);
    check(status, "Cannot enumerate HDF5 group.");
    return state.names;
}

inline void validate_columns(const std::vector<std::string>& columns, const ArrayView* value = nullptr) {
    std::set<std::string> unique;
    for (const auto& column : columns) {
        if (column.empty() || column == "." || column.find('/') != std::string::npos || column.find('\0') != std::string::npos)
            throw std::invalid_argument("Column names must be valid, non-empty HDF5 names without '/'.");
        if (!unique.insert(column).second) throw std::invalid_argument("Column names must be unique.");
    }
    if (value && (value->shape.empty() || value->shape[0] != columns.size()))
        throw std::invalid_argument("Received " + std::to_string(columns.size()) + " columns for an array with size " +
                                    std::to_string(value->shape.empty() ? 0 : value->shape[0]) + " along dimension 0.");
}

inline std::vector<std::string> stored_columns(hid_t parent) {
    return json_attribute<std::vector<std::string>>(parent, columns_attribute).value_or(children(parent, true));
}

inline std::vector<std::string> sources(hid_t file, const std::string& path) {
    const auto manifest = json_attribute<std::vector<std::string>>(file, sources_attribute);
    std::vector<std::string> result;
    if (manifest) for (const auto& source : *manifest) {
        const auto resolved = (fs::absolute(path).parent_path() / source).lexically_normal().string();
        if (!fs::is_regular_file(resolved))
            throw FileNotFoundError("HDF5 virtual dataset '" + path + "' is missing source '" + resolved + "'.");
        result.push_back(resolved);
    }
    return result;
}

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
} // namespace detail

inline Array convert(ArrayView value, DType type) { return detail::cast(value, type); }

struct Statistics {
    std::uint64_t count = 0;
    Array mean;
    Array squared_deviation;
};

inline std::optional<Statistics> statistics(ArrayView value) {
    if ((value.shape.size() != 1 && value.shape.size() != 2) || value.shape[0] == 0 || value.type.kind == Kind::complex)
        return std::nullopt;
    if (value.size() && !value.data) throw std::invalid_argument("Non-empty array has no data buffer.");
    std::optional<Array> converted;
    if (value.type != dtype<double>()) {
        converted = detail::cast(value, dtype<double>());
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
    for (std::size_t row = 0; row < value.shape[0]; ++row)
        for (std::size_t column = 0; column < width; ++column) means[column] += read(row * width + column);
    for (auto& mean : means) mean /= value.shape[0];
    for (std::size_t row = 0; row < value.shape[0]; ++row)
        for (std::size_t column = 0; column < width; ++column) {
            const double delta = read(row * width + column) - means[column];
            deviations[column] += delta * delta;
        }
    return Statistics{value.shape[0], Array::from(means, dimensions), Array::from(deviations, dimensions)};
}

inline std::optional<Statistics> combine_statistics(const std::vector<std::optional<Statistics>>& values) {
    if (values.empty() || std::any_of(values.begin(), values.end(), [](const auto& value) { return !value; })) return std::nullopt;
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

namespace detail {
inline void set_array_attribute(hid_t parent, const char* name, ArrayView value) {
    if (H5Aexists(parent, name) > 0) check(H5Adelete(parent, name), "Cannot replace HDF5 attribute.");
    auto type = datatype(value.type), space = dataspace(value.shape);
    Handle attribute(H5Acreate2(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    if (value.size()) check(H5Awrite(attribute, type, value.data), "Cannot write HDF5 numeric attribute.");
}

inline Array read_array_attribute(hid_t parent, const char* name, DType type) {
    Handle attribute(H5Aopen(parent, name, H5P_DEFAULT), H5Aclose);
    Handle space(H5Aget_space(attribute), H5Sclose);
    Array result(type, shape(space));
    auto memory_type = datatype(type);
    if (result.size()) check(H5Aread(attribute, memory_type, result.storage.data()), "Cannot read HDF5 numeric attribute.");
    return result;
}

inline void write_statistics(hid_t value, const Statistics& stats) {
    const auto count = Array::from<std::int64_t>({static_cast<std::int64_t>(stats.count)}, {});
    set_array_attribute(value, count_attribute, count.view());
    set_array_attribute(value, mean_attribute, stats.mean.view());
    set_array_attribute(value, deviation_attribute, stats.squared_deviation.view());
}

inline std::optional<Statistics> read_statistics(hid_t value) {
    if (H5Aexists(value, count_attribute) <= 0 || H5Aexists(value, mean_attribute) <= 0 || H5Aexists(value, deviation_attribute) <= 0)
        return std::nullopt;
    const auto count = read_array_attribute(value, count_attribute, dtype<std::uint64_t>()).values<std::uint64_t>();
    if (count.size() != 1) throw std::invalid_argument("Invalid HDF5 statistics count.");
    return Statistics{count[0], read_array_attribute(value, mean_attribute, dtype<double>()),
                     read_array_attribute(value, deviation_attribute, dtype<double>())};
}

inline void create_dataset(hid_t parent, const std::string& name, ArrayView value) {
    if (value.size() && !value.data) throw std::invalid_argument("Non-empty array has no data buffer.");
    auto type = datatype(value.type), space = dataspace(value.shape);
    Handle links(H5Pcreate(H5P_LINK_CREATE), H5Pclose);
    check(H5Pset_create_intermediate_group(links, 1), "Cannot enable intermediate dataset groups.");
    check(H5Pset_char_encoding(links, H5T_CSET_UTF8), "Cannot enable UTF-8 dataset names.");
    Handle output(H5Dcreate2(parent, name.c_str(), type, space, links, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
    if (value.size()) check(H5Dwrite(output, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.data), "Cannot write HDF5 dataset '" + name + "'.");
    const auto stats = statistics(value);
    if (stats) write_statistics(output, *stats);
    output.close_checked();
}

inline Handle read_datatype(hid_t value, DType type) {
    Handle stored(H5Dget_type(value), H5Tclose);
    // HDF5 2.0 native complex datasets already have interleaved components.
    // Read their own type rather than requesting an unsupported conversion to
    // the compound representation we retain for DSBO's output schema.
    if (H5Tget_class(stored) == H5T_COMPLEX) return stored;
    return datatype(type);
}

inline Array read_dataset(hid_t value) {
    Array result(dataset_dtype(value), dataset_shape(value));
    auto type = read_datatype(value, result.type);
    if (result.size()) check(H5Dread(value, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.storage.data()), "Cannot read HDF5 dataset.");
    return result;
}

inline Array read_named(hid_t file, const std::string& name, Columns columns = std::nullopt) {
    if (object_type(file, name) == H5O_TYPE_DATASET) {
        if (columns) throw std::invalid_argument("HDF5 dataset '" + name + "' was not written with named columns.");
        auto value = dataset(file, name);
        return read_dataset(value);
    }
    auto parent = group(file, name);
    const auto selected = columns.value_or(stored_columns(parent));
    validate_columns(selected);
    std::vector<Array> values;
    for (const auto& column : selected) {
        if (!exists(parent, column)) throw KeyError("HDF5 dataset '" + name + "' has no column '" + column + "'.");
        auto value = dataset(parent, column);
        values.push_back(read_dataset(value));
    }
    std::vector<ArrayView> views;
    for (const auto& value : values) views.push_back(value.view());
    return stack(views);
}

inline std::vector<hsize_t> partition_sizes(hid_t value) {
    if (const auto stored = json_attribute<std::vector<hsize_t>>(value, partitions_attribute)) return *stored;
    Handle properties(H5Dget_create_plist(value), H5Pclose);
    if (H5Pget_layout(properties) == H5D_VIRTUAL) {
        std::size_t count = 0;
        check(H5Pget_virtual_count(properties, &count), "Cannot inspect virtual partitions.");
        std::vector<hsize_t> sizes;
        for (std::size_t i = 0; i < count; ++i) {
            Handle space(H5Pget_virtual_vspace(properties, i), H5Sclose);
            const auto rank = H5Sget_simple_extent_ndims(space);
            Shape start(rank), end(rank);
            if (H5Sget_select_npoints(space) == 0) sizes.push_back(0);
            else {
                check(H5Sget_select_bounds(space, start.data(), end.data()), "Cannot inspect virtual selection.");
                sizes.push_back(rank ? end[0] - start[0] + 1 : 1);
            }
        }
        return sizes;
    }
    const auto dimensions = dataset_shape(value);
    return {dimensions.empty() ? 1 : dimensions[0]};
}

inline std::vector<hsize_t> partition_offsets(hid_t value, std::size_t count) {
    const auto sizes = json_attribute<std::vector<hsize_t>>(value, partitions_attribute);
    const auto dimensions = dataset_shape(value);
    if (!sizes || sizes->size() != count || dimensions.empty() ||
        std::accumulate(sizes->begin(), sizes->end(), hsize_t(0)) != dimensions[0])
        throw std::invalid_argument("Invalid HDF5 concatenation partitions.");
    std::vector<hsize_t> offsets{0};
    for (auto size : *sizes) offsets.push_back(offsets.back() + size);
    return offsets;
}
} // namespace detail

// A selection along the sample axis; a squeezed selection represents an integer
// index. Negative Python indices/slices are normalized by the bindings.
struct Selection {
    std::int64_t start = 0;
    std::int64_t step = 1;
    std::size_t count = 0;
    bool squeeze = false;
};

// Reader handles must not cross fork/process boundaries. Calls on one reader,
// including close(), are serialized; unrelated readers use HDF5's own locking.
class Reader {
    std::string path_;
    detail::Handle file_;
    mutable std::mutex mutex_;
    void require_open() const { if (!file_) throw std::runtime_error("HDF5 reader has been closed."); }

    Array read_locked(const std::string& name, Selection selection) const {
        auto value = detail::dataset(file_, name);
        const auto physical = detail::dataset_shape(value);
        const Shape series = physical.empty() ? Shape{1} : physical;
        if (!selection.step) throw std::invalid_argument("slice step cannot be zero");
        if (selection.squeeze && selection.count != 1) throw std::invalid_argument("An integer selection requires exactly one row.");
        if (selection.count) {
            if (selection.start < 0 || static_cast<hsize_t>(selection.start) >= series[0] ||
                selection.step == std::numeric_limits<std::int64_t>::min())
                throw std::out_of_range("HDF5 dataset index is out of range.");
            const auto available = selection.step > 0 ? (series[0] - 1 - selection.start) / selection.step
                                                     : static_cast<hsize_t>(selection.start / -selection.step);
            if (selection.count - 1 > available)
                throw std::out_of_range("HDF5 dataset index is out of range.");
        }
        Shape output = series;
        output[0] = selection.count;
        if (selection.squeeze) output.erase(output.begin());
        Array result(detail::dataset_dtype(value), output);
        if (!result.size()) return result;
        if (physical.empty()) {
            auto scalar = detail::read_dataset(value);
            for (std::size_t i = 0; i < selection.count; ++i)
                std::memcpy(result.storage.data() + i * result.type.size, scalar.storage.data(), result.type.size);
            return result;
        }
        detail::Handle source(H5Dget_space(value), H5Sclose);
        Shape start(series.size(), 0), stride(series.size(), 1), count = series;
        start[0] = selection.step > 0 ? selection.start : selection.start + static_cast<std::int64_t>(selection.count - 1) * selection.step;
        stride[0] = static_cast<hsize_t>(selection.step > 0 ? selection.step : -selection.step);
        count[0] = selection.count;
        detail::check(H5Sselect_hyperslab(source, H5S_SELECT_SET, start.data(), stride.data(), count.data(), nullptr), "Cannot select HDF5 rows.");
        auto memory = detail::dataspace(output), type = detail::read_datatype(value, result.type);
        detail::check(H5Dread(value, type, memory, source, H5P_DEFAULT, result.storage.data()), "Cannot read selected HDF5 rows.");
        if (selection.step < 0) {
            const auto row_bytes = result.storage.size() / selection.count;
            for (std::size_t i = 0; i < selection.count / 2; ++i)
                for (std::size_t j = 0; j < row_bytes; ++j)
                    std::swap(result.storage[i * row_bytes + j], result.storage[(selection.count - i - 1) * row_bytes + j]);
        }
        return result;
    }
public:
    explicit Reader(const std::string& path) : path_(path), file_(detail::open_file(path)) { detail::sources(file_, path_); }
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    std::string store() const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        return detail::string_attribute(file_, store_attribute).value_or("");
    }
    std::vector<std::string> keys(const std::string& root = "") const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        std::vector<std::string> result;
        const auto visit = [&](const auto& self, const std::string& prefix) -> void {
            auto parent = prefix.empty() ? detail::group(file_, "/") : detail::group(file_, prefix);
            for (const auto& name : detail::children(parent)) {
                const auto path = prefix.empty() ? name : prefix + "/" + name;
                if (detail::object_type(file_, path) == H5O_TYPE_DATASET) result.push_back(path);
                else if (detail::object_type(file_, path) == H5O_TYPE_GROUP) self(self, path);
            }
        };
        visit(visit, root);
        std::sort(result.begin(), result.end());
        return result;
    }
    std::vector<std::string> columns(const std::string& name = "primal") const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        if (detail::object_type(file_, name) == H5O_TYPE_DATASET) return {};
        auto parent = detail::group(file_, name);
        return detail::stored_columns(parent);
    }
    std::vector<std::string> row_keys() const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        auto value = detail::dataset(file_, row_store_dataset);
        const auto keys = detail::json_attribute<std::vector<std::string>>(value, keys_attribute);
        if (!keys) throw std::invalid_argument("Row store has no dsbo_keys attribute.");
        return *keys;
    }
    Shape shape(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        auto value = detail::dataset(file_, name);
        return detail::dataset_shape(value);
    }
    DType type(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        auto value = detail::dataset(file_, name);
        return detail::dataset_dtype(value);
    }
    std::vector<hsize_t> partition_sizes(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        auto value = detail::dataset(file_, name);
        return detail::partition_sizes(value);
    }
    std::optional<Statistics> statistics(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        auto value = detail::dataset(file_, name);
        return detail::read_statistics(value);
    }
    Array read(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        auto value = detail::dataset(file_, name);
        return detail::read_dataset(value);
    }
    Array read(const std::string& name, Selection selection) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        return read_locked(name, selection);
    }
    Array read_indices(const std::string& name, std::vector<std::int64_t> indices) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        auto value = detail::dataset(file_, name);
        auto dimensions = detail::dataset_shape(value);
        if (dimensions.empty()) dimensions = {1};
        for (auto& index : indices) {
            if (index < 0) index += dimensions[0];
            if (index < 0 || static_cast<hsize_t>(index) >= dimensions[0]) throw std::out_of_range("HDF5 dataset index is out of range.");
        }
        auto output = dimensions;
        output[0] = indices.size();
        Array result(detail::dataset_dtype(value), output);
        if (!result.size()) return result;
        auto unique = indices;
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        auto selected_shape = dimensions;
        selected_shape[0] = unique.size();
        Array selected(result.type, selected_shape);
        if (detail::dataset_shape(value).empty()) {
            selected = detail::read_dataset(value);
        } else {
            detail::Handle source(H5Dget_space(value), H5Sclose);
            detail::check(H5Sselect_none(source), "Cannot initialize indexed selection.");
            Shape start(dimensions.size(), 0), count = dimensions;
            count[0] = 1;
            for (auto index : unique) {
                start[0] = index;
                detail::check(H5Sselect_hyperslab(source, H5S_SELECT_OR, start.data(), nullptr, count.data(), nullptr), "Cannot select indexed rows.");
            }
            auto memory = detail::dataspace(selected_shape), type = detail::read_datatype(value, selected.type);
            detail::check(H5Dread(value, type, memory, source, H5P_DEFAULT, selected.storage.data()), "Cannot read indexed HDF5 rows.");
        }
        const auto row_bytes = selected.storage.size() / unique.size();
        for (std::size_t i = 0; i < indices.size(); ++i) {
            const auto offset = std::lower_bound(unique.begin(), unique.end(), indices[i]) - unique.begin();
            std::memcpy(result.storage.data() + i * row_bytes, selected.storage.data() + offset * row_bytes, row_bytes);
        }
        return result;
    }
    void close() { std::lock_guard<std::mutex> lock(mutex_); file_.close_checked(); }
};

struct GradientContribution {
    std::string primal_path;
    std::variant<std::string, Array> source;
    Columns columns;
};

// The writer is stateless. Callers coordinate writes to the same destination;
// HDF5's thread safety does not make multi-call updates transactional.
class HDF5Writer {
    static void write_datasets(const std::string& path, const NamedArrays& values, Columns columns) {
        if (columns) {
            detail::validate_columns(*columns);
            for (const auto& item : values) detail::validate_columns(*columns, &item.second);
        }
        auto file = detail::open_file(path, true);
        detail::set_string(file, store_attribute, columns ? "column" : "flat");
        for (const auto& [name, value] : values) {
            if (!columns) detail::create_dataset(file, name, value);
            else {
                auto parent = detail::require_group(file, name);
                detail::set_string(parent, columns_attribute, detail::Json(*columns).dump());
                for (std::size_t i = 0; i < columns->size(); ++i) detail::create_dataset(parent, (*columns)[i], value.column(i));
            }
        }
        detail::finish_file(file);
    }

    static Array read_gradient(const std::string& path, Columns columns, Columns primal_columns = std::nullopt,
                               const std::map<std::string, Shape>* primal_shapes = nullptr) {
        auto file = detail::open_file(path);
        detail::sources(file, path);
        if (detail::object_type(file, "gradient") == H5O_TYPE_DATASET) return detail::read_named(file, "gradient", columns);
        if (!columns) {
            columns = primal_columns;
            const auto original = detail::fs::path(path);
            const auto stem = original.stem().string();
            const std::string suffix = "_gradient";
            if (!columns && stem.size() >= suffix.size() && stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0) {
                const auto primal_path = original.parent_path() / (stem.substr(0, stem.size() - suffix.size()) + original.extension().string());
                auto primal = detail::open_file(primal_path.string());
                if (detail::object_type(primal, "primal") == H5O_TYPE_GROUP) {
                    auto group = detail::group(primal, "primal");
                    columns = detail::stored_columns(group);
                }
            }
        }
        auto parent = detail::group(file, "gradient");
        const auto stored = detail::stored_columns(parent);
        const auto selected = columns.value_or(stored);
        detail::validate_columns(selected);
        if (stored.empty()) throw std::invalid_argument("HDF5 gradient contains no named columns.");
        auto first = detail::dataset(parent, stored[0]);
        const auto type = detail::dataset_dtype(first);
        const auto template_shape = detail::dataset_shape(first);
        std::vector<Array> values;
        for (const auto& column : selected) {
            if (detail::exists(parent, column)) {
                auto value = detail::dataset(parent, column);
                values.push_back(detail::read_dataset(value));
            } else values.emplace_back(type, primal_shapes ? primal_shapes->at(column) : template_shape);
        }
        std::vector<ArrayView> views;
        for (const auto& value : values) views.push_back(value.view());
        return detail::stack(views);
    }

    static void gradient_contributions_impl(const std::string& path, ArrayView gradient, Columns columns,
                                            std::vector<GradientContribution>& output, std::set<std::string>& ancestors) {
        const auto identity = detail::fs::absolute(path).lexically_normal().string();
        if (!ancestors.insert(identity).second) throw std::invalid_argument("Cyclic HDF5 virtual concatenation.");
        auto file = detail::open_file(path);
        const auto inputs = detail::sources(file, path);
        if (inputs.empty()) {
            output.push_back({path, detail::copy(gradient), columns});
            ancestors.erase(identity);
            return;
        }
        const bool named = detail::object_type(file, "primal") == H5O_TYPE_GROUP;
        if (named) {
            auto parent = detail::group(file, "primal");
            if (!columns) columns = detail::stored_columns(parent);
            detail::validate_columns(*columns, &gradient);
        } else if (columns) throw std::invalid_argument("HDF5 dataset 'primal' was not written with named columns.");
        std::vector<std::string> names;
        if (named) for (const auto& column : *columns) names.push_back("primal/" + column);
        else names.push_back("primal");
        std::vector<std::vector<hsize_t>> offsets;
        for (std::size_t j = 0; j < names.size(); ++j) {
            auto value = detail::dataset(file, names[j]);
            const auto piece = named ? gradient.column(j) : gradient;
            if (piece.shape != detail::dataset_shape(value)) throw std::invalid_argument("Gradient shape does not match HDF5 dataset '" + names[j] + "' shape.");
            offsets.push_back(detail::partition_offsets(value, inputs.size()));
        }
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            auto source = detail::open_file(inputs[i]);
            std::vector<Array> pieces;
            for (std::size_t j = 0; j < names.size(); ++j) {
                auto dataset = detail::dataset(source, names[j]);
                auto view = named ? gradient.column(j) : gradient;
                const auto row_bytes = elements(Shape(view.shape.begin() + 1, view.shape.end())) * view.type.size;
                const auto first = offsets[j][i], count = offsets[j][i + 1] - first;
                auto dimensions = detail::dataset_shape(dataset);
                if (elements(dimensions) * view.type.size != count * row_bytes) throw std::invalid_argument("Invalid HDF5 concatenation partitions.");
                const auto* data = static_cast<const unsigned char*>(view.data);
                pieces.push_back(detail::copy({view.type, dimensions, row_bytes && data ? data + first * row_bytes : data}));
            }
            Array value;
            if (named) {
                std::vector<ArrayView> views;
                for (const auto& piece : pieces) views.push_back(piece.view());
                value = detail::stack(views);
            } else value = std::move(pieces[0]);
            source.reset();
            gradient_contributions_impl(inputs[i], value.view(), columns, output, ancestors);
        }
        ancestors.erase(identity);
    }
public:
    static Array open_primal(const std::string& path, Columns columns = std::nullopt) {
        auto file = detail::open_file(path);
        detail::sources(file, path);
        return detail::read_named(file, "primal", columns);
    }
    static std::pair<Array, std::optional<Array>> open_primal_tangent(const std::string& path, Columns columns = std::nullopt) {
        auto file = detail::open_file(path);
        detail::sources(file, path);
        if (!detail::exists(file, "primal")) throw std::invalid_argument("Saved HDF5 file must contain a 'primal' dataset.");
        auto primal = detail::read_named(file, "primal", columns);
        std::optional<Array> tangent;
        if (detail::exists(file, "tangent")) tangent = detail::read_named(file, "tangent", columns);
        return {std::move(primal), std::move(tangent)};
    }
    static void write_primal(const std::string& path, ArrayView primal, Columns columns = std::nullopt) {
        write_datasets(path, {{"primal", primal}}, columns);
    }
    static void write_primal_tangent(const std::string& path, ArrayView primal, ArrayView tangent, Columns columns = std::nullopt) {
        write_datasets(path, {{"primal", primal}, {"tangent", tangent}}, columns);
    }
    static void write_named_datasets(const std::string& path, const NamedArrays& values) {
        if (values.empty()) return;
        auto file = detail::open_file(path, true, true);
        detail::set_string(file, store_attribute, "flat");
        for (const auto& [name, value] : values) {
            if (name.empty() || name.back() == '/') throw std::invalid_argument("Invalid HDF5 dataset path: '" + name + "'.");
            if (detail::exists(file, name)) detail::check(H5Ldelete(file, name.c_str(), H5P_DEFAULT), "Cannot replace HDF5 dataset.");
            detail::create_dataset(file, name, value);
        }
        detail::finish_file(file);
    }
    static void write_row_store(const std::string& path, const NamedArrays& features) {
        if (features.empty()) throw std::invalid_argument("Row-store storage requires at least one feature.");
        std::vector<std::string> keys;
        std::size_t rows = features.front().second.size();
        auto type = features.front().second.type;
        type.little_endian = native_little_endian();
        for (const auto& [key, value] : features) {
            if (key.empty() || key.find('\0') != std::string::npos) throw std::invalid_argument("Feature keys must be non-empty strings without NUL characters.");
            if (value.shape.size() > 1) throw std::invalid_argument("Row-store features must be scalars or one-dimensional arrays.");
            if (value.size() != rows) throw std::invalid_argument("All row-store features must contain the same number of rows.");
            if (std::find(keys.begin(), keys.end(), key) != keys.end()) throw std::invalid_argument("Feature keys must be unique.");
            keys.push_back(key);
            type = detail::promote(type, value.type);
        }
        Array combined(type, {static_cast<hsize_t>(rows), static_cast<hsize_t>(features.size())});
        for (std::size_t column = 0; column < features.size(); ++column) {
            const auto converted = detail::cast(features[column].second, type);
            for (std::size_t row = 0; row < rows; ++row)
                std::memcpy(combined.storage.data() + (row * features.size() + column) * type.size, converted.storage.data() + row * type.size, type.size);
        }
        auto file = detail::open_file(path, true);
        detail::set_string(file, store_attribute, "row");
        detail::create_dataset(file, row_store_dataset, combined.view());
        auto value = detail::dataset(file, row_store_dataset);
        detail::set_string(value, keys_attribute, detail::Json(keys).dump());
        value.close_checked();
        detail::finish_file(file);
    }
    static std::vector<std::string> columns(const std::string& path) {
        auto file = detail::open_file(path);
        if (detail::object_type(file, "primal") == H5O_TYPE_DATASET) return {};
        auto parent = detail::group(file, "primal");
        return detail::stored_columns(parent);
    }
    static Shape shape_from_path(const std::string& path) {
        auto file = detail::open_file(path);
        if (detail::object_type(file, "primal") == H5O_TYPE_DATASET) {
            auto value = detail::dataset(file, "primal");
            return detail::dataset_shape(value);
        }
        auto parent = detail::group(file, "primal");
        const auto columns = detail::stored_columns(parent);
        if (columns.empty()) throw std::invalid_argument("HDF5 dataset '" + path + "' contains no columns under 'primal'.");
        auto first = detail::dataset(parent, columns[0]);
        const auto dimensions = detail::dataset_shape(first);
        return {dimensions.empty() ? 1 : dimensions[0], static_cast<hsize_t>(columns.size())};
    }
    static Array load_gradient(const std::string& path, Columns columns = std::nullopt) {
        return read_gradient(path, columns);
    }
    static Array load_backward_gradient(const std::string& primal_path, const std::string& gradient_path) {
        auto file = detail::open_file(primal_path);
        detail::sources(file, primal_path);
        if (detail::object_type(file, "primal") == H5O_TYPE_DATASET) return read_gradient(gradient_path, std::nullopt);
        auto parent = detail::group(file, "primal");
        const auto columns = detail::stored_columns(parent);
        std::map<std::string, Shape> dimensions;
        for (const auto& column : columns) {
            auto value = detail::dataset(parent, column);
            dimensions[column] = detail::dataset_shape(value);
        }
        return read_gradient(gradient_path, std::nullopt, columns, &dimensions);
    }
    static void write_gradient(const std::string& path, ArrayView gradient, Columns columns = std::nullopt, bool merge = false) {
        if (merge && columns) {
            detail::validate_columns(*columns, &gradient);
            auto file = detail::open_file(path, true, true);
            if (detail::object_type(file, "gradient") != H5O_TYPE_GROUP)
                throw std::invalid_argument("Cannot accumulate named gradients into an unnamed gradient.");
            auto parent = detail::group(file, "gradient");
            auto stored = detail::stored_columns(parent);
            for (std::size_t i = 0; i < columns->size(); ++i) {
                const auto& column = (*columns)[i];
                if (detail::exists(parent, column)) detail::check(H5Ldelete(parent, column.c_str(), H5P_DEFAULT), "Cannot replace gradient column.");
                detail::create_dataset(parent, column, gradient.column(i));
                if (std::find(stored.begin(), stored.end(), column) == stored.end()) stored.push_back(column);
            }
            detail::set_string(parent, columns_attribute, detail::Json(stored).dump());
            parent.close_checked();
            detail::finish_file(file);
            return;
        }
        static std::atomic<std::uint64_t> sequence{0};
        const auto temporary = path + "." + std::to_string(std::random_device{}()) + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(sequence++) + ".partial";
        try {
            write_datasets(temporary, {{"gradient", gradient}}, columns);
            detail::fs::rename(temporary, path);
        } catch (...) {
            std::error_code ignored;
            detail::fs::remove(temporary, ignored);
            throw;
        }
    }
    static void aggregate_gradients(const std::vector<std::string>& paths, const std::string& destination) {
        if (paths.empty()) throw std::invalid_argument("Gradient aggregation requires at least one source file.");
        std::vector<std::string> selected;
        for (const auto& path : paths) {
            auto file = detail::open_file(path);
            if (detail::object_type(file, "gradient") == H5O_TYPE_GROUP) {
                auto parent = detail::group(file, "gradient");
                for (const auto& column : detail::stored_columns(parent))
                    if (std::find(selected.begin(), selected.end(), column) == selected.end()) selected.push_back(column);
            }
        }
        const Columns columns = selected.empty() ? std::nullopt : Columns(selected);
        auto total = load_gradient(paths[0], columns);
        for (std::size_t i = 1; i < paths.size(); ++i) {
            const auto value = load_gradient(paths[i], columns);
            total = detail::add(total.view(), value.view());
        }
        write_gradient(destination, total.view(), columns);
    }
    static std::vector<GradientContribution> gradient_contributions(const std::string& path, ArrayView gradient, Columns columns = std::nullopt) {
        std::vector<GradientContribution> output;
        std::set<std::string> ancestors;
        gradient_contributions_impl(path, gradient, columns, output, ancestors);
        return output;
    }
    static std::vector<GradientContribution> gradient_file_contributions(const std::string& primal_path, const std::string& gradient_path) {
        Columns columns;
        bool virtual_input;
        {
            auto primal_file = detail::open_file(primal_path), gradient_file = detail::open_file(gradient_path);
            detail::sources(primal_file, primal_path);
            std::vector<std::pair<std::string, std::string>> pairs;
            if (!detail::exists(primal_file, "primal") || !detail::exists(gradient_file, "gradient"))
                throw std::invalid_argument("Native HDF5 '/gradient' layout must match the primal dataset or named-column group.");
            const auto primal_kind = detail::object_type(primal_file, "primal"), gradient_kind = detail::object_type(gradient_file, "gradient");
            if (primal_kind == H5O_TYPE_DATASET && gradient_kind == H5O_TYPE_DATASET) pairs.emplace_back("primal", "gradient");
            else if (primal_kind == H5O_TYPE_GROUP && gradient_kind == H5O_TYPE_GROUP) {
                auto primal = detail::group(primal_file, "primal"), gradient = detail::group(gradient_file, "gradient");
                columns = detail::stored_columns(gradient);
                detail::validate_columns(*columns);
                const auto actual = detail::children(gradient), primal_columns = detail::stored_columns(primal);
                if (columns->empty() || std::set<std::string>(columns->begin(), columns->end()) != std::set<std::string>(actual.begin(), actual.end()))
                    throw std::invalid_argument("Native HDF5 gradient columns must name every stored gradient column.");
                for (const auto& column : *columns) {
                    if (std::find(primal_columns.begin(), primal_columns.end(), column) == primal_columns.end())
                        throw std::invalid_argument("HDF5 primal has no gradient column '" + column + "'.");
                    pairs.emplace_back("primal/" + column, "gradient/" + column);
                }
            } else throw std::invalid_argument("Native HDF5 '/gradient' layout must match the primal dataset or named-column group.");
            for (const auto& [primal_name, gradient_name] : pairs) {
                if (detail::object_type(primal_file, primal_name) != H5O_TYPE_DATASET || detail::object_type(gradient_file, gradient_name) != H5O_TYPE_DATASET)
                    throw std::invalid_argument("Native HDF5 gradient columns must be datasets.");
                auto primal = detail::dataset(primal_file, primal_name), gradient = detail::dataset(gradient_file, gradient_name);
                if (detail::dataset_shape(primal) != detail::dataset_shape(gradient))
                    throw std::invalid_argument("Gradient shape does not match HDF5 dataset '" + primal_name + "' shape.");
                try { detail::dataset_dtype(gradient); }
                catch (const TypeError&) { throw std::invalid_argument("Native HDF5 gradient datasets must have numeric values."); }
            }
            virtual_input = H5Aexists(primal_file, sources_attribute) > 0;
        }
        if (!virtual_input) return {{primal_path, gradient_path, columns}};
        const auto gradient = load_gradient(gradient_path, columns);
        return gradient_contributions(primal_path, gradient.view(), columns);
    }

    static void reduce(const std::vector<std::string>& paths, const std::string& destination) {
        if (paths.empty()) throw std::invalid_argument("HDF5 reduction requires at least one source file.");
        struct Metadata { Shape shape; DType type; std::optional<Statistics> stats; std::optional<std::vector<std::string>> keys; };
        std::vector<std::map<std::string, Metadata>> metadata;
        std::vector<std::string> stores, relative;
        const auto destination_directory = detail::fs::absolute(destination).parent_path();
        for (const auto& path : paths) {
            if (!detail::fs::is_regular_file(path)) throw FileNotFoundError("Missing registered batch input: " + path);
            auto extension = detail::fs::path(path).extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension != ".h5" && extension != ".hdf5")
                throw TypeError("HDF5 reduction does not support '" + path + "'. Only .h5, .hdf5 files can be reduced.");
            detail::Handle file;
            try { file = detail::open_file(path); }
            catch (const IOError&) { throw TypeError("HDF5 reduction source '" + path + "' is not a valid HDF5 file. The file may be a different format (e.g. PyTorch .pt, JSON, etc.)."); }
            const auto store = detail::string_attribute(file, store_attribute).value_or("");
            if (store != "flat" && store != "column" && store != "row")
                throw std::invalid_argument("HDF5 source '" + path + "' has an unknown or missing store-type tag.");
            stores.push_back(store);
            Reader reader(path);
            std::map<std::string, Metadata> items;
            for (const auto& name : reader.keys()) {
                auto value = detail::dataset(file, name);
                items.emplace(name, Metadata{detail::dataset_shape(value), detail::dataset_dtype(value), detail::read_statistics(value),
                              detail::json_attribute<std::vector<std::string>>(value, keys_attribute)});
            }
            if (items.empty()) throw std::invalid_argument("HDF5 source '" + path + "' contains no datasets.");
            metadata.push_back(std::move(items));
            relative.push_back(detail::fs::absolute(path).lexically_normal().lexically_relative(destination_directory.lexically_normal()).generic_string());
        }
        for (std::size_t i = 1; i < metadata.size(); ++i) {
            if (metadata[i].size() != metadata[0].size()) throw std::invalid_argument("All HDF5 reduction sources must expose identical datasets.");
            for (const auto& item : metadata[0]) if (!metadata[i].count(item.first))
                throw std::invalid_argument("All HDF5 reduction sources must expose identical datasets.");
            if (stores[i] != stores[0]) throw std::invalid_argument("HDF5 reduction sources must share the same store type.");
        }
        // Validate before truncating the destination or opening any VDS mappings.
        for (const auto& [name, first] : metadata[0]) for (const auto& batch : metadata) {
            const auto& current = batch.at(name);
            if (current.keys != first.keys) throw std::invalid_argument("Cannot reduce HDF5 dataset '" + name + "' with incompatible keys.");
            if (current.shape.size() != first.shape.size() || (!first.shape.empty() && !std::equal(first.shape.begin() + 1, first.shape.end(), current.shape.begin() + 1)))
                throw std::invalid_argument("Cannot reduce HDF5 dataset '" + name + "' with incompatible shapes.");
        }
        const auto destination_identity = detail::fs::absolute(destination).lexically_normal();
        for (const auto& path : paths) {
            if (detail::fs::absolute(path).lexically_normal() == destination_identity ||
                (detail::fs::exists(destination) && detail::fs::equivalent(path, destination)))
                throw std::invalid_argument("HDF5 reduction destination must not overwrite a source.");
        }
        auto output = detail::open_file(destination, true);
        detail::set_string(output, sources_attribute, detail::Json(relative).dump());
        detail::set_string(output, store_attribute, stores[0]);
        {
            auto first = detail::open_file(paths[0]);
            const auto copy_groups = [&](const auto& self, hid_t source, hid_t target) -> void {
                for (const auto& name : detail::children(source)) if (detail::object_type(source, name) == H5O_TYPE_GROUP) {
                    auto source_group = detail::group(source, name), target_group = detail::require_group(target, name);
                    if (const auto columns = detail::string_attribute(source_group, columns_attribute)) detail::set_string(target_group, columns_attribute, *columns);
                    self(self, source_group, target_group);
                }
            };
            copy_groups(copy_groups, first, output);
        }
        for (const auto& [name, first] : metadata[0]) {
            Shape dimensions = first.shape.empty() ? Shape{static_cast<hsize_t>(paths.size())} : first.shape;
            if (!first.shape.empty()) {
                dimensions[0] = 0;
                for (const auto& batch : metadata) dimensions[0] += batch.at(name).shape[0];
            }
            auto space = detail::dataspace(dimensions), type = detail::datatype(first.type);
            detail::Handle properties(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
            // Keep even an all-empty concatenation virtual.
            detail::check(H5Pset_layout(properties, H5D_VIRTUAL), "Cannot enable virtual dataset layout.");
            std::vector<hsize_t> sizes;
            std::vector<std::optional<Statistics>> stats;
            hsize_t offset = 0;
            for (std::size_t i = 0; i < paths.size(); ++i) {
                const auto& batch = metadata[i].at(name);
                const auto count = batch.shape.empty() ? 1 : batch.shape[0];
                sizes.push_back(count);
                stats.push_back(batch.stats);
                if (elements(batch.shape)) {
                    auto source_space = detail::dataspace(batch.shape);
                    Shape start(dimensions.size(), 0), selection = batch.shape.empty() ? Shape{1} : batch.shape;
                    start[0] = offset;
                    detail::check(H5Sselect_hyperslab(space, H5S_SELECT_SET, start.data(), nullptr, selection.data(), nullptr), "Cannot select virtual partition.");
                    detail::check(H5Pset_virtual(properties, space, relative[i].c_str(), name.c_str(), source_space), "Cannot map virtual HDF5 source.");
                }
                offset += count;
            }
            auto full_space = detail::dataspace(dimensions);
            detail::Handle links(H5Pcreate(H5P_LINK_CREATE), H5Pclose);
            detail::check(H5Pset_create_intermediate_group(links, 1), "Cannot create virtual parent groups.");
            detail::check(H5Pset_char_encoding(links, H5T_CSET_UTF8), "Cannot enable UTF-8 virtual dataset names.");
            detail::Handle value(H5Dcreate2(output, name.c_str(), type, full_space, links, properties, H5P_DEFAULT), H5Dclose);
            detail::set_string(value, partitions_attribute, detail::Json(sizes).dump());
            if (const auto combined = combine_statistics(stats)) detail::write_statistics(value, *combined);
            if (first.keys) detail::set_string(value, keys_attribute, detail::Json(*first.keys).dump());
        }
        detail::finish_file(output);
    }
};
} // namespace dsbo::io
