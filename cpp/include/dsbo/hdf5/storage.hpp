#pragma once

#include "types.hpp"
#include "statistics.hpp"
#include "internal/core.hpp"
#include "internal/arrays.hpp"
#include "internal/datasets.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace dsbo::io {

// Stateless storage facade. Callers coordinate writes to the same destination;
// HDF5's thread safety does not make multi-call updates transactional.
class HDF5Storage {
public:
    // Primal/tangent I/O. Named columns occupy axis 0 of the supplied arrays.
    static Array open_primal(const std::string& path, Columns columns = std::nullopt) {
        auto file = internal::open_file(path);
        internal::sources(file, path);
        return internal::read_named(file, "primal", columns);
    }
    static std::pair<Array, std::optional<Array>> open_primal_tangent(const std::string& path, Columns columns = std::nullopt) {
        auto file = internal::open_file(path);
        internal::sources(file, path);
        if (!internal::exists(file, "primal")) throw std::invalid_argument("Saved HDF5 file must contain a 'primal' dataset.");
        auto primal = internal::read_named(file, "primal", columns);
        std::optional<Array> tangent;
        if (internal::exists(file, "tangent")) tangent = internal::read_named(file, "tangent", columns);
        return {std::move(primal), std::move(tangent)};
    }
    static void write_primal(const std::string& path, ArrayView primal, Columns columns = std::nullopt) {
        write_datasets(path, {{"primal", primal}}, columns);
    }
    static void write_primal_tangent(const std::string& path, ArrayView primal, ArrayView tangent, Columns columns = std::nullopt) {
        write_datasets(path, {{"primal", primal}, {"tangent", tangent}}, columns);
    }

    // Named dataset updates and row-store construction.
    static void write_named_datasets(const std::string& path, const NamedArrays& values) {
        if (values.empty()) return;
        auto file = internal::open_file(path, true, true);
        internal::set_string(file, store_attribute, "flat");
        for (const auto& [name, value] : values) {
            if (name.empty() || name.back() == '/') throw std::invalid_argument("Invalid HDF5 dataset path: '" + name + "'.");
            if (internal::exists(file, name)) internal::check(H5Ldelete(file, name.c_str(), H5P_DEFAULT), "Cannot replace HDF5 dataset.");
            internal::create_dsbo_dataset(file, name, value);
        }
        internal::finish_file(file);
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
            type = internal::promote(type, value.type);
        }
        Array combined(type, {static_cast<hsize_t>(rows), static_cast<hsize_t>(features.size())});
        for (std::size_t column = 0; column < features.size(); ++column) {
            const auto converted = internal::cast(features[column].second, type);
            for (std::size_t row = 0; row < rows; ++row)
                std::memcpy(combined.storage.data() + (row * features.size() + column) * type.size, converted.storage.data() + row * type.size, type.size);
        }
        auto file = internal::open_file(path, true);
        internal::set_string(file, store_attribute, "row");
        internal::create_dsbo_dataset(file, row_store_dataset, combined.view());
        auto value = internal::dataset(file, row_store_dataset);
        internal::set_string(value, keys_attribute, internal::Json(keys).dump());
        value.close_checked();
        internal::finish_file(file);
    }

    // Primal metadata inspection.
    static std::vector<std::string> columns(const std::string& path) {
        auto file = internal::open_file(path);
        if (internal::object_type(file, "primal") == H5O_TYPE_DATASET) return {};
        auto parent = internal::group(file, "primal");
        return internal::stored_columns(parent);
    }
    static Shape shape_from_path(const std::string& path) {
        auto file = internal::open_file(path);
        if (internal::object_type(file, "primal") == H5O_TYPE_DATASET) {
            auto value = internal::dataset(file, "primal");
            return internal::dataset_shape(value);
        }
        auto parent = internal::group(file, "primal");
        const auto columns = internal::stored_columns(parent);
        if (columns.empty()) throw std::invalid_argument("HDF5 dataset '" + path + "' contains no columns under 'primal'.");
        auto first = internal::dataset(parent, columns[0]);
        const auto dimensions = internal::dataset_shape(first);
        return {dimensions.empty() ? 1 : dimensions[0], static_cast<hsize_t>(columns.size())};
    }

    // Gradient loading and writing.
    static Array load_gradient(const std::string& path, Columns columns = std::nullopt) {
        return read_gradient(path, columns);
    }
    static Array load_backward_gradient(const std::string& primal_path, const std::string& gradient_path) {
        auto file = internal::open_file(primal_path);
        internal::sources(file, primal_path);
        if (internal::object_type(file, "primal") == H5O_TYPE_DATASET) {
            auto primal = internal::dataset(file, "primal");
            internal::dataset_shape(primal);
            return read_gradient(gradient_path, std::nullopt);
        }
        auto parent = internal::group(file, "primal");
        const auto columns = internal::stored_columns(parent);
        std::map<std::string, Shape> dimensions;
        for (const auto& column : columns) {
            auto value = internal::dataset(parent, column);
            dimensions[column] = internal::dataset_shape(value);
        }
        return read_gradient(gradient_path, std::nullopt, columns, &dimensions);
    }
    // merge updates/replaces the selected named columns; it does not add values.
    static void write_gradient(const std::string& path, ArrayView gradient, Columns columns = std::nullopt, bool merge = false) {
        if (merge && columns) update_gradient_columns(path, gradient, *columns);
        else replace_gradient_file(path, gradient, columns);
    }

    // Gradient aggregation and virtual-concatenation adjoint routing.
    static void aggregate_gradients(const std::vector<std::string>& paths, const std::string& destination) {
        if (paths.empty()) throw std::invalid_argument("Gradient aggregation requires at least one source file.");
        std::vector<std::string> selected;
        for (const auto& path : paths) {
            auto file = internal::open_file(path);
            if (internal::object_type(file, "gradient") == H5O_TYPE_GROUP) {
                auto parent = internal::group(file, "gradient");
                for (const auto& column : internal::stored_columns(parent))
                    if (std::find(selected.begin(), selected.end(), column) == selected.end()) selected.push_back(column);
            }
        }
        const Columns columns = selected.empty() ? std::nullopt : Columns(selected);
        auto total = load_gradient(paths[0], columns);
        for (std::size_t i = 1; i < paths.size(); ++i) {
            const auto value = load_gradient(paths[i], columns);
            total = internal::add(total.view(), value.view());
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
        const auto layout = validate_gradient_file(primal_path, gradient_path);
        if (!layout.virtual_input) return {{primal_path, gradient_path, layout.columns}};
        const auto gradient = load_gradient(gradient_path, layout.columns);
        return gradient_contributions(primal_path, gradient.view(), layout.columns);
    }

    // Virtual reduction: validate sources before opening/truncating the output.
    static void reduce(const std::vector<std::string>& paths, const std::string& destination) {
        const auto inputs = collect_reduction_sources(paths, destination);
        validate_reduction_sources(inputs);
        validate_reduction_destination(inputs, destination);
        std::vector<std::string> relative;
        for (const auto& input : inputs) relative.push_back(input.relative_path);
        auto output = internal::open_file(destination, true);
        internal::set_string(output, sources_attribute, internal::Json(relative).dump());
        internal::set_string(output, store_attribute, inputs.front().store);
        {
            auto first = internal::open_file(inputs.front().path);
            copy_reduction_groups(first, output);
        }
        for (const auto& item : inputs.front().datasets) create_virtual_dataset(output, item.first, inputs);
        internal::finish_file(output);
    }

private:
    // Physical dataset writes shared by primal/tangent and gradient operations.
    static void write_datasets(const std::string& path, const NamedArrays& values, Columns columns) {
        if (columns) {
            internal::validate_columns(*columns);
            for (const auto& item : values) internal::validate_columns(*columns, &item.second);
        }
        auto file = internal::open_file(path, true);
        internal::set_string(file, store_attribute, columns ? "column" : "flat");
        for (const auto& [name, value] : values) {
            if (!columns) internal::create_dsbo_dataset(file, name, value);
            else {
                auto parent = internal::require_group(file, name);
                internal::set_string(parent, columns_attribute, internal::Json(*columns).dump());
                for (std::size_t i = 0; i < columns->size(); ++i) internal::create_dsbo_dataset(parent, (*columns)[i], value.column(i));
            }
        }
        internal::finish_file(file);
    }

    // Gradient layout, loading, and write modes.
    struct GradientLayout {
        Columns columns;
        bool virtual_input;
    };

    static GradientLayout validate_gradient_file(const std::string& primal_path, const std::string& gradient_path) {
        Columns columns;
        auto primal_file = internal::open_file(primal_path), gradient_file = internal::open_file(gradient_path);
        internal::sources(primal_file, primal_path);
        std::vector<std::pair<std::string, std::string>> pairs;
        if (!internal::exists(primal_file, "primal") || !internal::exists(gradient_file, "gradient"))
            throw std::invalid_argument("Native HDF5 '/gradient' layout must match the primal dataset or named-column group.");
        const auto primal_kind = internal::object_type(primal_file, "primal"), gradient_kind = internal::object_type(gradient_file, "gradient");
        if (primal_kind == H5O_TYPE_DATASET && gradient_kind == H5O_TYPE_DATASET) pairs.emplace_back("primal", "gradient");
        else if (primal_kind == H5O_TYPE_GROUP && gradient_kind == H5O_TYPE_GROUP) {
            auto primal = internal::group(primal_file, "primal"), gradient = internal::group(gradient_file, "gradient");
            columns = internal::stored_columns(gradient);
            internal::validate_columns(*columns);
            const auto actual = internal::children(gradient), primal_columns = internal::stored_columns(primal);
            if (columns->empty() || std::set<std::string>(columns->begin(), columns->end()) != std::set<std::string>(actual.begin(), actual.end()))
                throw std::invalid_argument("Native HDF5 gradient columns must name every stored gradient column.");
            for (const auto& column : *columns) {
                if (std::find(primal_columns.begin(), primal_columns.end(), column) == primal_columns.end())
                    throw std::invalid_argument("HDF5 primal has no gradient column '" + column + "'.");
                pairs.emplace_back("primal/" + column, "gradient/" + column);
            }
        } else throw std::invalid_argument("Native HDF5 '/gradient' layout must match the primal dataset or named-column group.");
        for (const auto& [primal_name, gradient_name] : pairs) {
            if (internal::object_type(primal_file, primal_name) != H5O_TYPE_DATASET || internal::object_type(gradient_file, gradient_name) != H5O_TYPE_DATASET)
                throw std::invalid_argument("Native HDF5 gradient columns must be datasets.");
            auto primal = internal::dataset(primal_file, primal_name), gradient = internal::dataset(gradient_file, gradient_name);
            if (internal::dataset_shape(primal) != internal::dataset_shape(gradient))
                throw std::invalid_argument("Gradient shape does not match HDF5 dataset '" + primal_name + "' shape.");
            try { internal::dataset_dtype(gradient); }
            catch (const TypeError&) { throw std::invalid_argument("Native HDF5 gradient datasets must have numeric values."); }
        }
        return {columns, H5Aexists(primal_file, sources_attribute) > 0};
    }

    static Array read_gradient(const std::string& path, Columns columns, Columns primal_columns = std::nullopt,
                               const std::map<std::string, Shape>* primal_shapes = nullptr) {
        auto file = internal::open_file(path);
        internal::sources(file, path);
        if (internal::object_type(file, "gradient") == H5O_TYPE_DATASET) return internal::read_named(file, "gradient", columns);
        if (!columns) {
            columns = primal_columns;
            const auto original = internal::fs::path(path);
            const auto stem = original.stem().string();
            const std::string suffix = "_gradient";
            if (!columns && stem.size() >= suffix.size() && stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0) {
                const auto primal_path = original.parent_path() / (stem.substr(0, stem.size() - suffix.size()) + original.extension().string());
                auto primal = internal::open_file(primal_path.string());
                if (internal::object_type(primal, "primal") == H5O_TYPE_GROUP) {
                    auto group = internal::group(primal, "primal");
                    columns = internal::stored_columns(group);
                }
            }
        }
        auto parent = internal::group(file, "gradient");
        const auto stored = internal::stored_columns(parent);
        const auto selected = columns.value_or(stored);
        internal::validate_columns(selected);
        if (stored.empty()) throw std::invalid_argument("HDF5 gradient contains no named columns.");
        auto first = internal::dataset(parent, stored[0]);
        const auto type = internal::dataset_dtype(first);
        const auto template_shape = internal::dataset_shape(first);
        std::vector<Array> values;
        for (const auto& column : selected) {
            if (internal::exists(parent, column)) {
                auto value = internal::dataset(parent, column);
                values.push_back(internal::read_dataset(value));
            } else values.emplace_back(type, primal_shapes ? primal_shapes->at(column) : template_shape);
        }
        std::vector<ArrayView> views;
        for (const auto& value : values) views.push_back(value.view());
        return internal::stack(views);
    }

    static void update_gradient_columns(const std::string& path, ArrayView gradient, const std::vector<std::string>& columns) {
        internal::validate_columns(columns, &gradient);
        auto file = internal::open_file(path, true, true);
        if (internal::object_type(file, "gradient") != H5O_TYPE_GROUP)
            throw std::invalid_argument("Cannot accumulate named gradients into an unnamed gradient.");
        auto parent = internal::group(file, "gradient");
        auto stored = internal::stored_columns(parent);
        for (std::size_t i = 0; i < columns.size(); ++i) {
            const auto& column = columns[i];
            if (internal::exists(parent, column)) internal::check(H5Ldelete(parent, column.c_str(), H5P_DEFAULT), "Cannot replace gradient column.");
            internal::create_dsbo_dataset(parent, column, gradient.column(i));
            if (std::find(stored.begin(), stored.end(), column) == stored.end()) stored.push_back(column);
        }
        internal::set_string(parent, columns_attribute, internal::Json(stored).dump());
        parent.close_checked();
        internal::finish_file(file);
    }

    static void replace_gradient_file(const std::string& path, ArrayView gradient, Columns columns) {
        static std::atomic<std::uint64_t> sequence{0};
        const auto temporary = path + "." + std::to_string(std::random_device{}()) + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(sequence++) + ".partial";
        try {
            write_datasets(temporary, {{"gradient", gradient}}, columns);
            internal::fs::rename(temporary, path);
        } catch (...) {
            std::error_code ignored;
            internal::fs::remove(temporary, ignored);
            throw;
        }
    }

    // Recursively split concatenated gradients, retaining the active ancestry
    // until every child is processed. Close each source before descending.
    static void gradient_contributions_impl(const std::string& path, ArrayView gradient, Columns columns,
                                            std::vector<GradientContribution>& output, std::set<std::string>& ancestors) {
        const auto identity = internal::fs::absolute(path).lexically_normal().string();
        if (!ancestors.insert(identity).second) throw std::invalid_argument("Cyclic HDF5 virtual concatenation.");
        auto file = internal::open_file(path);
        const auto inputs = internal::sources(file, path);
        if (inputs.empty()) {
            output.push_back({path, internal::copy(gradient), columns});
            ancestors.erase(identity);
            return;
        }
        const bool named = internal::object_type(file, "primal") == H5O_TYPE_GROUP;
        if (named) {
            auto parent = internal::group(file, "primal");
            if (!columns) columns = internal::stored_columns(parent);
            internal::validate_columns(*columns, &gradient);
        } else if (columns) throw std::invalid_argument("HDF5 dataset 'primal' was not written with named columns.");
        std::vector<std::string> names;
        if (named) for (const auto& column : *columns) names.push_back("primal/" + column);
        else names.push_back("primal");
        std::vector<std::vector<hsize_t>> offsets;
        for (std::size_t j = 0; j < names.size(); ++j) {
            auto value = internal::dataset(file, names[j]);
            const auto piece = named ? gradient.column(j) : gradient;
            if (piece.shape != internal::dataset_shape(value)) throw std::invalid_argument("Gradient shape does not match HDF5 dataset '" + names[j] + "' shape.");
            offsets.push_back(internal::partition_offsets(value, inputs.size()));
        }
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            auto source = internal::open_file(inputs[i]);
            std::vector<Array> pieces;
            for (std::size_t j = 0; j < names.size(); ++j) {
                auto dataset = internal::dataset(source, names[j]);
                auto view = named ? gradient.column(j) : gradient;
                const auto row_bytes = elements(Shape(view.shape.begin() + 1, view.shape.end())) * view.type.size;
                const auto first = offsets[j][i], count = offsets[j][i + 1] - first;
                auto dimensions = internal::dataset_shape(dataset);
                if (elements(dimensions) * view.type.size != count * row_bytes) throw std::invalid_argument("Invalid HDF5 concatenation partitions.");
                const auto* data = static_cast<const unsigned char*>(view.data);
                pieces.push_back(internal::copy({view.type, dimensions, row_bytes && data ? data + first * row_bytes : data}));
            }
            Array value;
            if (named) {
                std::vector<ArrayView> views;
                for (const auto& piece : pieces) views.push_back(piece.view());
                value = internal::stack(views);
            } else value = std::move(pieces[0]);
            source.reset();
            gradient_contributions_impl(inputs[i], value.view(), columns, output, ancestors);
        }
        ancestors.erase(identity);
    }

    // Virtual reduction metadata and phases.
    struct DatasetMetadata {
        Shape shape;
        DType type;
        std::optional<Statistics> stats;
        std::optional<std::vector<std::string>> keys;
    };
    struct ReductionSource {
        std::string path;
        std::string relative_path;
        std::string store;
        std::map<std::string, DatasetMetadata> datasets;
    };

    static std::vector<ReductionSource> collect_reduction_sources(const std::vector<std::string>& paths,
                                                                 const std::string& destination) {
        if (paths.empty()) throw std::invalid_argument("HDF5 reduction requires at least one source file.");
        std::vector<ReductionSource> inputs;
        const auto destination_directory = internal::fs::absolute(destination).parent_path();
        for (const auto& path : paths) {
            if (!internal::fs::is_regular_file(path)) throw FileNotFoundError("Missing registered batch input: " + path);
            auto extension = internal::fs::path(path).extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension != ".h5" && extension != ".hdf5")
                throw TypeError("HDF5 reduction does not support '" + path + "'. Only .h5, .hdf5 files can be reduced.");
            internal::Handle file;
            try { file = internal::open_file(path); }
            catch (const IOError&) { throw TypeError("HDF5 reduction source '" + path + "' is not a valid HDF5 file. The file may be a different format (e.g. PyTorch .pt, JSON, etc.)."); }
            const auto store = internal::string_attribute(file, store_attribute).value_or("");
            if (store != "flat" && store != "column" && store != "row")
                throw std::invalid_argument("HDF5 source '" + path + "' has an unknown or missing store-type tag.");
            internal::sources(file, path);
            std::map<std::string, DatasetMetadata> items;
            for (const auto& name : internal::dataset_paths(file)) {
                auto value = internal::dataset(file, name);
                items.emplace(name, DatasetMetadata{internal::dataset_shape(value), internal::dataset_dtype(value), internal::read_statistics(value),
                              internal::json_attribute<std::vector<std::string>>(value, keys_attribute)});
            }
            if (items.empty()) throw std::invalid_argument("HDF5 source '" + path + "' contains no datasets.");
            const auto relative = internal::fs::absolute(path).lexically_normal().lexically_relative(destination_directory.lexically_normal()).generic_string();
            inputs.push_back({path, relative, store, std::move(items)});
        }
        return inputs;
    }

    static void validate_reduction_sources(const std::vector<ReductionSource>& inputs) {
        const auto& first_source = inputs.front();
        for (std::size_t i = 1; i < inputs.size(); ++i) {
            if (inputs[i].datasets.size() != first_source.datasets.size()) throw std::invalid_argument("All HDF5 reduction sources must expose identical datasets.");
            for (const auto& item : first_source.datasets) if (!inputs[i].datasets.count(item.first))
                throw std::invalid_argument("All HDF5 reduction sources must expose identical datasets.");
            if (inputs[i].store != first_source.store) throw std::invalid_argument("HDF5 reduction sources must share the same store type.");
        }
        for (const auto& [name, first] : first_source.datasets) {
            for (const auto& input : inputs) {
                const auto& current = input.datasets.at(name);
                if (current.keys != first.keys) throw std::invalid_argument("Cannot reduce HDF5 dataset '" + name + "' with incompatible keys.");
                if (current.shape.size() != first.shape.size() || (!first.shape.empty() && !std::equal(first.shape.begin() + 1, first.shape.end(), current.shape.begin() + 1)))
                    throw std::invalid_argument("Cannot reduce HDF5 dataset '" + name + "' with incompatible shapes.");
            }
        }
    }

    static void validate_reduction_destination(const std::vector<ReductionSource>& inputs, const std::string& destination) {
        const auto identity = internal::fs::absolute(destination).lexically_normal();
        for (const auto& input : inputs) {
            if (internal::fs::absolute(input.path).lexically_normal() == identity ||
                (internal::fs::exists(destination) && internal::fs::equivalent(input.path, destination)))
                throw std::invalid_argument("HDF5 reduction destination must not overwrite a source.");
        }
    }

    static void copy_reduction_groups(hid_t source, hid_t target) {
        for (const auto& name : internal::children(source)) {
            if (internal::object_type(source, name) != H5O_TYPE_GROUP) continue;
            auto source_group = internal::group(source, name), target_group = internal::require_group(target, name);
            if (const auto columns = internal::string_attribute(source_group, columns_attribute))
                internal::set_string(target_group, columns_attribute, *columns);
            copy_reduction_groups(source_group, target_group);
        }
    }

    static void create_virtual_dataset(hid_t output, const std::string& name, const std::vector<ReductionSource>& inputs) {
        const auto& first = inputs.front().datasets.at(name);
        Shape dimensions = first.shape.empty() ? Shape{static_cast<hsize_t>(inputs.size())} : first.shape;
        if (!first.shape.empty()) {
            dimensions[0] = 0;
            for (const auto& input : inputs) dimensions[0] += input.datasets.at(name).shape[0];
        }
        auto space = internal::dataspace(dimensions), type = internal::datatype(first.type);
        internal::Handle properties(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
        // Keep even an all-empty concatenation virtual.
        internal::check(H5Pset_layout(properties, H5D_VIRTUAL), "Cannot enable virtual dataset layout.");
        std::vector<hsize_t> sizes;
        std::vector<std::optional<Statistics>> stats;
        hsize_t offset = 0;
        for (const auto& input : inputs) {
            const auto& batch = input.datasets.at(name);
            const auto count = batch.shape.empty() ? 1 : batch.shape[0];
            sizes.push_back(count);
            stats.push_back(batch.stats);
            if (elements(batch.shape)) {
                auto source_space = internal::dataspace(batch.shape);
                Shape start(dimensions.size(), 0), selection = batch.shape.empty() ? Shape{1} : batch.shape;
                start[0] = offset;
                internal::check(H5Sselect_hyperslab(space, H5S_SELECT_SET, start.data(), nullptr, selection.data(), nullptr), "Cannot select virtual partition.");
                internal::check(H5Pset_virtual(properties, space, input.relative_path.c_str(), name.c_str(), source_space), "Cannot map virtual HDF5 source.");
            }
            offset += count;
        }
        auto full_space = internal::dataspace(dimensions);
        internal::Handle links(H5Pcreate(H5P_LINK_CREATE), H5Pclose);
        internal::check(H5Pset_create_intermediate_group(links, 1), "Cannot create virtual parent groups.");
        internal::check(H5Pset_char_encoding(links, H5T_CSET_UTF8), "Cannot enable UTF-8 virtual dataset names.");
        internal::Handle value(H5Dcreate2(output, name.c_str(), type, full_space, links, properties, H5P_DEFAULT), H5Dclose);
        internal::set_string(value, partitions_attribute, internal::Json(sizes).dump());
        if (const auto combined = combine_statistics(stats)) internal::write_statistics(value, *combined);
        if (first.keys) internal::set_string(value, keys_attribute, internal::Json(*first.keys).dump());
    }
};

} // namespace dsbo::io
