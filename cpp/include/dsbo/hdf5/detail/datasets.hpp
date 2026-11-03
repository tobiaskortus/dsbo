#pragma once

#include "../types.hpp"
#include "../statistics.hpp"
#include "core.hpp"
#include "arrays.hpp"
#include "../../third_party/json.hpp"
#include <cstdint>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsbo::io {

// DSBO's on-disk schema.
inline constexpr char store_attribute[] = "dsbo_store";
inline constexpr char columns_attribute[] = "dsbo_columns";
inline constexpr char row_store_dataset[] = "__dsbo_row_store__";
inline constexpr char keys_attribute[] = "dsbo_keys";
inline constexpr char sources_attribute[] = "dsbo_vds_sources";
inline constexpr char partitions_attribute[] = "dsbo_partition_sizes";
inline constexpr char count_attribute[] = "dsbo_stats_count";
inline constexpr char mean_attribute[] = "dsbo_stats_mean";
inline constexpr char deviation_attribute[] = "dsbo_stats_squared_deviation";

namespace detail {
using Json = nlohmann::json;

template<class T> inline std::optional<T> json_attribute(hid_t parent, const char* name) {
    const auto value = string_attribute(parent, name);
    if (!value) return std::nullopt;
    try { return Json::parse(*value).template get<T>(); }
    catch (const Json::exception& error) { throw std::invalid_argument(std::string("Invalid HDF5 metadata '") + name + "': " + error.what()); }
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

// Statistics persistence.
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
    Statistics result{count[0], read_array_attribute(value, mean_attribute, dtype<double>()),
                      read_array_attribute(value, deviation_attribute, dtype<double>())};
    validate_statistics(result);
    return result;
}

// Creates/writes a numeric dataset and persists DSBO statistics when supported.
inline void create_dsbo_dataset(hid_t parent, const std::string& name, ArrayView value) {
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

// Virtual-concatenation partitions.
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
} // namespace dsbo::io
