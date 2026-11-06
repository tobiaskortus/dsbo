#pragma once

#include "../types.hpp"
#include <algorithm>
#include <exception>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dsbo::io {

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

namespace internal {
namespace fs = std::filesystem;

// Resource ownership and file/group access.
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

// Dataspaces and numeric datatypes.
inline Shape shape(hid_t space) {
    switch (H5Sget_simple_extent_type(space)) {
        case H5S_SCALAR: return {};
        case H5S_SIMPLE: break;
        case H5S_NULL: throw std::invalid_argument("HDF5 NULL dataspaces (H5S_NULL) are not supported.");
        default: throw IOError("Cannot inspect HDF5 dataspace class.");
    }
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

// Generic string and numeric attributes.
inline std::optional<std::string> string_attribute(hid_t parent, const char* name) {
    if (H5Aexists(parent, name) <= 0) return std::nullopt;
    Handle attribute(H5Aopen(parent, name, H5P_DEFAULT), H5Aclose);
    Handle type(H5Aget_type(attribute), H5Tclose);
    if (H5Tget_class(type) != H5T_STRING) throw std::invalid_argument(std::string("Invalid HDF5 string attribute: ") + name);
    Handle space(H5Aget_space(attribute), H5Sclose);
    const auto count = H5Sget_simple_extent_npoints(space);
    if (count < 0) throw IOError(std::string("Cannot inspect HDF5 string attribute: ") + name);
    // H5Aread reads the entire attribute into storage for a single string.
    if (count != 1) throw std::invalid_argument(std::string("Invalid HDF5 string attribute '") + name + "': expected exactly one element.");
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

// Group traversal.
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

// Sorted dataset paths under root. Visit each hard-linked object once,
// including in cyclic graphs, and do not follow soft or external links.
inline std::vector<std::string> dataset_paths(hid_t file, const std::string& root = "") {
    auto parent = group(file, root.empty() ? "/" : root);
    std::vector<std::string> result;
    struct State { const std::string& prefix; std::vector<std::string>& names; std::exception_ptr error; } state{root, result, {}};
    const auto callback = [](hid_t, const char* name, const H5O_info2_t* info, void* pointer) -> herr_t {
        auto& state = *static_cast<State*>(pointer);
        try {
            if (info->type == H5O_TYPE_DATASET)
                state.names.push_back(state.prefix.empty() ? name : state.prefix + "/" + name);
            return 0;
        } catch (...) { state.error = std::current_exception(); return -1; }
    };
    const auto status = H5Ovisit3(parent, H5_INDEX_NAME, H5_ITER_INC, callback, &state, H5O_INFO_BASIC);
    if (state.error) std::rethrow_exception(state.error);
    check(status, "Cannot enumerate HDF5 datasets.");
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace internal
} // namespace dsbo::io
