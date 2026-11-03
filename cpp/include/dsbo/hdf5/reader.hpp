#pragma once

#include "types.hpp"
#include "detail/core.hpp"
#include "detail/datasets.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dsbo::io {

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
        return detail::dataset_paths(file_, root);
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

} // namespace dsbo::io
