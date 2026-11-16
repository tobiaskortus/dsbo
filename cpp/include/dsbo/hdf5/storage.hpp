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
#include <initializer_list>
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
    // Primal/tangent I/O. Array methods are flat-only; dict methods expose per-key values.
    static Array open_primal(const std::string& path) {
        auto file = internal::open_file(path);
        internal::sources(file, path);
        if (internal::object_type(file, "primal") == H5O_TYPE_GROUP)
            throw std::invalid_argument("HDF5 store is a dict; use open_dict.");
        return internal::read_dataset(internal::dataset(file, "primal"));
    }

    static NamedArrayValues open_dict(const std::string& path, Columns columns = std::nullopt) {
        auto file = internal::open_file(path);
        internal::sources(file, path);
        return read_key_group(file, "primal", columns);
    }

    static std::pair<Array, std::optional<Array>> open_primal_tangent(const std::string& path) {
        auto file = internal::open_file(path);
        internal::sources(file, path);
        if (!internal::exists(file, "primal")) throw std::invalid_argument("Saved HDF5 file must contain a 'primal' dataset.");
        if (internal::object_type(file, "primal") == H5O_TYPE_GROUP)
            throw std::invalid_argument("HDF5 store is a dict; use open_dict_tangent.");
        auto primal = internal::read_dataset(internal::dataset(file, "primal"));
        std::optional<Array> tangent;
        if (internal::exists(file, "tangent")) {
            if (internal::object_type(file, "tangent") == H5O_TYPE_GROUP)
                throw std::invalid_argument("HDF5 tangent is a dict; use open_dict_tangent.");
            tangent = internal::read_dataset(internal::dataset(file, "tangent"));
        }
        return {std::move(primal), std::move(tangent)};
    }

    static std::pair<NamedArrayValues, std::optional<NamedArrayValues>> open_dict_tangent(const std::string& path, Columns columns = std::nullopt) {
        auto file = internal::open_file(path);
        internal::sources(file, path);
        auto primal = read_key_group(file, "primal", columns);
        std::optional<NamedArrayValues> tangent;
        if (internal::exists(file, "tangent")) {
            auto values = read_key_group(file, "tangent", columns, true);
            if (!values.empty()) tangent = std::move(values);
        }
        return {std::move(primal), std::move(tangent)};
    }

    static void write_primal(const std::string& path, ArrayView primal) {
        write_datasets(path, {{"primal", primal}});
    }

    static void write_primal(const std::string& path, const NamedArrays& primal) {
        write_dict_file(path, {{"primal", &primal}});
    }

    static void write_primal_tangent(const std::string& path, ArrayView primal, ArrayView tangent) {
        write_datasets(path, {{"primal", primal}, {"tangent", tangent}});
    }

    static void write_primal_tangent(const std::string& path, const NamedArrays& primal, const NamedArrays& tangent) {
        const auto primal_keys = keys(primal);
        internal::validate_keys(primal_keys);
        internal::validate_keys(keys(tangent));
        for (const auto& [key, value] : tangent) {
            auto found = primal.end();
            for (auto item = primal.begin(); item != primal.end(); ++item) {
                if (item->first == key) { found = item; break; }
            }
            if (found == primal.end()) throw std::invalid_argument("Tangent key '" + key + "' is not a primal key.");
            if (found->second.shape != value.shape || found->second.type != value.type)
                throw std::invalid_argument("Tangent '" + key + "' does not match the primal shape/dtype.");
        }
        write_dict_file(path, {{"primal", &primal}, {"tangent", &tangent}});
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
        return internal::stored_keys(parent);
    }

    static Shape shape_from_path(const std::string& path) {
        auto file = internal::open_file(path);
        if (internal::object_type(file, "primal") == H5O_TYPE_GROUP)
            throw std::invalid_argument("HDF5 store is a dict; use dict_shape_from_path.");
        auto value = internal::dataset(file, "primal");
        return internal::dataset_shape(value);
    }

    static std::vector<std::pair<std::string, Shape>> dict_shape_from_path(const std::string& path) {
        auto file = internal::open_file(path);
        auto parent = internal::group(file, "primal");
        std::vector<std::pair<std::string, Shape>> shapes;
        for (const auto& key : internal::stored_keys(parent)) {
            auto dataset = internal::dataset(parent, key);
            shapes.emplace_back(key, internal::dataset_shape(dataset));
        }
        return shapes;
    }

    // Gradient loading and writing.
    static Array load_gradient(const std::string& path) {
        auto file = internal::open_file(path);
        internal::sources(file, path);
        if (internal::object_type(file, "gradient") == H5O_TYPE_GROUP)
            throw std::invalid_argument("HDF5 gradient is a dict; use load_dict_gradient.");
        return internal::read_dataset(internal::dataset(file, "gradient"));
    }

    static NamedArrayValues load_dict_gradient(const std::string& path, Columns columns = std::nullopt) {
        auto file = internal::open_file(path);
        internal::sources(file, path);
        return read_key_group(file, "gradient", columns);
    }

    static Array load_backward_gradient(const std::string& primal_path, const std::string& gradient_path) {
        auto primal_file = internal::open_file(primal_path);
        internal::sources(primal_file, primal_path);
        if (internal::object_type(primal_file, "primal") == H5O_TYPE_GROUP)
            throw std::invalid_argument("HDF5 primal is a dict; use load_backward_dict_gradient.");
        auto primal = internal::dataset(primal_file, "primal");
        internal::dataset_shape(primal);
        auto gradient_file = internal::open_file(gradient_path);
        internal::sources(gradient_file, gradient_path);
        if (internal::object_type(gradient_file, "gradient") == H5O_TYPE_GROUP)
            throw std::invalid_argument("HDF5 gradient is a dict; use load_backward_dict_gradient.");
        return internal::read_dataset(internal::dataset(gradient_file, "gradient"));
    }

    static NamedArrayValues load_backward_dict_gradient(const std::string& primal_path, const std::string& gradient_path) {
        const auto layout = validate_gradient_file(primal_path, gradient_path);
        return load_dict_gradient(gradient_path, layout.keys);
    }

    // merge updates/replaces the selected keys; it does not add values.
    static void write_gradient(const std::string& path, ArrayView gradient, bool merge = false) {
        (void)merge;
        replace_gradient_file(path, gradient);
    }

    static void write_gradient(const std::string& path, const NamedArrays& gradient, bool merge = false) {
        if (merge) {
            auto file = internal::open_file(path, true, true);
            if (internal::object_type(file, "gradient") != H5O_TYPE_GROUP)
                throw std::invalid_argument("Cannot merge dict gradients into a flat gradient.");
            auto parent = internal::group(file, "gradient");
            auto stored = internal::stored_keys(parent);
            internal::validate_keys(keys(gradient));
            for (const auto& [key, value] : gradient) {
                if (internal::exists(parent, key)) internal::check(H5Ldelete(parent, key.c_str(), H5P_DEFAULT), "Cannot replace gradient key.");
                internal::create_dsbo_dataset(parent, key, value);
                if (std::find(stored.begin(), stored.end(), key) == stored.end()) stored.push_back(key);
            }
            internal::set_string(parent, columns_attribute, internal::Json(stored).dump());
            parent.close_checked();
            internal::finish_file(file);
            return;
        }
        replace_dict_gradient_file(path, gradient);
    }

    // Gradient aggregation and virtual-concatenation adjoint routing.
    static void aggregate_gradients(const std::vector<std::string>& paths, const std::string& destination) {
        if (paths.empty()) throw std::invalid_argument("Gradient aggregation requires at least one source file.");
        bool any_dict = false, any_flat = false;
        for (const auto& path : paths) {
            auto file = internal::open_file(path);
            if (internal::object_type(file, "gradient") == H5O_TYPE_GROUP) any_dict = true;
            else any_flat = true;
        }
        if (any_dict && any_flat) throw std::invalid_argument("Cannot aggregate flat and TensorDict gradients together.");
        if (!any_dict) {
            auto total = load_gradient(paths[0]);
            for (std::size_t i = 1; i < paths.size(); ++i) {
                const auto value = load_gradient(paths[i]);
                total = internal::add(total.view(), value.view());
            }
            write_gradient(destination, total.view());
            return;
        }
        std::vector<std::string> keys;
        std::map<std::string, Array> totals;
        for (const auto& path : paths) {
            for (auto& [key, value] : load_dict_gradient(path)) {
                auto found = totals.find(key);
                if (found == totals.end()) {
                    keys.push_back(key);
                    totals.emplace(key, std::move(value));
                } else found->second = internal::add(found->second.view(), value.view());
            }
        }
        NamedArrays views;
        views.reserve(keys.size());
        for (const auto& key : keys) views.emplace_back(key, totals.at(key).view());
        write_gradient(destination, views);
    }

    static std::vector<GradientContribution> gradient_contributions(const std::string& path, ArrayView gradient, Columns columns = std::nullopt) {
        std::vector<GradientContribution> output;
        std::set<std::string> ancestors;
        gradient_contributions_impl(path, {true, {{"", gradient}}, columns}, output, ancestors);
        return output;
    }

    static std::vector<GradientContribution> gradient_file_contributions(const std::string& primal_path, const std::string& gradient_path) {
        const auto layout = validate_gradient_file(primal_path, gradient_path);
        if (!layout.virtual_input) return {{primal_path, gradient_path, layout.keys}};
        if (layout.keys.empty()) {
            const auto gradient = load_gradient(gradient_path);
            return gradient_contributions(primal_path, gradient.view(), std::nullopt);
        }
        const auto gradient = load_dict_gradient(gradient_path, layout.keys);
        return gradient_dict_contributions(primal_path, gradient, layout.keys);
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
    static std::vector<std::string> keys(const NamedArrays& values) {
        std::vector<std::string> result;
        result.reserve(values.size());
        for (const auto& [key, value] : values) result.push_back(key);
        return result;
    }

    static void write_key_group(hid_t file, const std::string& root, const NamedArrays& values) {
        const auto names = keys(values);
        internal::validate_keys(names);
        auto parent = internal::require_group(file, root);
        for (const auto& [key, value] : values) internal::create_dsbo_dataset(parent, key, value);
        internal::set_string(parent, columns_attribute, internal::Json(names).dump());
        parent.close_checked();
    }

    static void write_dict_file(const std::string& path,
                                std::initializer_list<std::pair<std::string, const NamedArrays*>> roots) {
        auto file = internal::open_file(path, true);
        internal::set_string(file, store_attribute, "column");
        for (const auto& [root, values] : roots) write_key_group(file, root, *values);
        internal::finish_file(file);
    }

    static NamedArrayValues read_key_group(hid_t file, const std::string& root, Columns columns, bool missing_ok = false) {
        auto parent = internal::group(file, root);
        const auto selected = columns.value_or(internal::stored_keys(parent));
        internal::validate_keys(selected);
        NamedArrayValues values;
        for (const auto& key : selected) {
            if (!internal::exists(parent, key)) {
                if (missing_ok) continue;
                throw KeyError("HDF5 group '" + root + "' has no key '" + key + "'.");
            }
            auto dataset = internal::dataset(parent, key);
            values.emplace_back(key, internal::read_dataset(dataset));
        }
        return values;
    }

    // Physical dataset writes shared by flat primal/tangent and gradient operations.
    static void write_datasets(const std::string& path, const NamedArrays& values) {
        auto file = internal::open_file(path, true);
        internal::set_string(file, store_attribute, "flat");
        for (const auto& [name, value] : values) internal::create_dsbo_dataset(file, name, value);
        internal::finish_file(file);
    }

    // Gradient layout, loading, and write modes.
    struct GradientLayout {
        std::vector<std::string> keys;
        bool virtual_input;
    };

    static GradientLayout validate_gradient_file(const std::string& primal_path, const std::string& gradient_path) {
        std::vector<std::string> keys;
        auto primal_file = internal::open_file(primal_path), gradient_file = internal::open_file(gradient_path);
        internal::sources(primal_file, primal_path);
        std::vector<std::pair<std::string, std::string>> pairs;
        if (!internal::exists(primal_file, "primal") || !internal::exists(gradient_file, "gradient"))
            throw std::invalid_argument("Native HDF5 '/gradient' layout must match the primal dataset or TensorDict group.");
        const auto primal_kind = internal::object_type(primal_file, "primal"), gradient_kind = internal::object_type(gradient_file, "gradient");
        if (primal_kind == H5O_TYPE_DATASET && gradient_kind == H5O_TYPE_DATASET) pairs.emplace_back("primal", "gradient");
        else if (primal_kind == H5O_TYPE_GROUP && gradient_kind == H5O_TYPE_GROUP) {
            auto primal = internal::group(primal_file, "primal"), gradient = internal::group(gradient_file, "gradient");
            keys = internal::stored_keys(gradient);
            internal::validate_keys(keys);
            const auto actual = internal::children(gradient), primal_keys = internal::stored_keys(primal);
            if (keys.empty() || std::set<std::string>(keys.begin(), keys.end()) != std::set<std::string>(actual.begin(), actual.end()))
                throw std::invalid_argument("Native HDF5 gradient keys must name every stored gradient key.");
            for (const auto& key : keys) {
                if (std::find(primal_keys.begin(), primal_keys.end(), key) == primal_keys.end())
                    throw std::invalid_argument("HDF5 primal has no gradient key '" + key + "'.");
                pairs.emplace_back("primal/" + key, "gradient/" + key);
            }
        } else throw std::invalid_argument("Native HDF5 '/gradient' layout must match the primal dataset or TensorDict group.");
        for (const auto& [primal_name, gradient_name] : pairs) {
            if (internal::object_type(primal_file, primal_name) != H5O_TYPE_DATASET || internal::object_type(gradient_file, gradient_name) != H5O_TYPE_DATASET)
                throw std::invalid_argument("Native HDF5 gradient keys must be datasets.");
            auto primal = internal::dataset(primal_file, primal_name), gradient = internal::dataset(gradient_file, gradient_name);
            if (internal::dataset_shape(primal) != internal::dataset_shape(gradient))
                throw std::invalid_argument("Gradient shape does not match HDF5 dataset '" + primal_name + "' shape.");
            try { internal::dataset_dtype(gradient); }
            catch (const TypeError&) { throw std::invalid_argument("Native HDF5 gradient datasets must have numeric values."); }
        }
        return {keys, H5Aexists(primal_file, sources_attribute) > 0};
    }

    static std::string temporary_path(const std::string& path) {
        static std::atomic<std::uint64_t> sequence{0};
        return path + "." + std::to_string(std::random_device{}()) + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(sequence++) + ".partial";
    }

    template<class Writer> static void replace_atomically(const std::string& path, Writer&& write) {
        const auto temporary = temporary_path(path);
        try {
            write(temporary);
            internal::fs::rename(temporary, path);
        } catch (...) {
            std::error_code ignored;
            internal::fs::remove(temporary, ignored);
            throw;
        }
    }

    static void replace_gradient_file(const std::string& path, ArrayView gradient) {
        replace_atomically(path, [&](const std::string& temporary) {
            write_datasets(temporary, {{"gradient", gradient}});
        });
    }

    static void replace_dict_gradient_file(const std::string& path, const NamedArrays& gradient) {
        replace_atomically(path, [&](const std::string& temporary) {
            write_dict_file(temporary, {{"gradient", &gradient}});
        });
    }

    // Recursively split concatenated gradients, retaining the active ancestry
    // until every child is processed. Close each source before descending.
    // stacked holds one axis-0-stacked array; otherwise values holds one view
    // per key. The same split logic serves flat, legacy named, and dict stores.
    struct GradientPieces {
        bool stacked;
        NamedArrays values;
        Columns keys;
    };

    static Array split_partition(ArrayView view, hid_t dataset, hsize_t first, hsize_t count) {
        const auto row_bytes = elements(Shape(view.shape.begin() + 1, view.shape.end())) * view.type.size;
        auto dimensions = internal::dataset_shape(dataset);
        if (elements(dimensions) * view.type.size != count * row_bytes)
            throw std::invalid_argument("Invalid HDF5 concatenation partitions.");
        const auto* data = static_cast<const unsigned char*>(view.data);
        return internal::copy({view.type, dimensions, row_bytes && data ? data + first * row_bytes : data});
    }

    static void gradient_contributions_impl(const std::string& path, const GradientPieces& gradient,
                                            std::vector<GradientContribution>& output,
                                            std::set<std::string>& ancestors) {
        const auto identity = internal::fs::absolute(path).lexically_normal().string();
        if (!ancestors.insert(identity).second) throw std::invalid_argument("Cyclic HDF5 virtual concatenation.");
        auto file = internal::open_file(path);
        const auto inputs = internal::sources(file, path);
        if (inputs.empty()) {
            if (gradient.stacked) {
                output.push_back({path, internal::copy(gradient.values[0].second),
                                  gradient.keys ? *gradient.keys : std::vector<std::string>{}});
            } else {
                NamedArrayValues values;
                values.reserve(gradient.values.size());
                for (const auto& [key, value] : gradient.values) values.emplace_back(key, internal::copy(value));
                output.push_back({path, std::move(values),
                                  gradient.keys ? *gradient.keys : std::vector<std::string>{}});
            }
            ancestors.erase(identity);
            return;
        }
        const bool named = internal::object_type(file, "primal") == H5O_TYPE_GROUP;
        auto pieces = gradient.values;
        auto keys = gradient.keys;
        if (named) {
            auto parent = internal::group(file, "primal");
            if (gradient.stacked) {
                if (!keys) keys = internal::stored_keys(parent);
                internal::validate_keys(*keys, &gradient.values[0].second);
                pieces.clear();
                pieces.reserve(keys->size());
                for (std::size_t j = 0; j < keys->size(); ++j)
                    pieces.emplace_back((*keys)[j], gradient.values[0].second.column(j));
            } else {
                if (!keys) throw std::invalid_argument("TensorDict gradient keys are required.");
                const auto primal_keys = internal::stored_keys(parent);
                for (const auto& key : *keys) {
                    if (std::find(primal_keys.begin(), primal_keys.end(), key) == primal_keys.end())
                        throw std::invalid_argument("HDF5 primal has no gradient key '" + key + "'.");
                }
                if (pieces.size() != keys->size())
                    throw std::invalid_argument("Gradient values do not match the TensorDict keys.");
            }
        } else if (gradient.stacked) {
            if (keys) throw std::invalid_argument("HDF5 dataset 'primal' is not a TensorDict group.");
        } else {
            throw std::invalid_argument("HDF5 dataset 'primal' is not a TensorDict group.");
        }
        std::vector<std::string> names;
        names.reserve(pieces.size());
        for (const auto& [key, value] : pieces) names.push_back(named ? "primal/" + key : "primal");
        std::vector<std::vector<hsize_t>> offsets;
        offsets.reserve(pieces.size());
        for (std::size_t j = 0; j < pieces.size(); ++j) {
            auto value = internal::dataset(file, names[j]);
            if (pieces[j].second.shape != internal::dataset_shape(value))
                throw std::invalid_argument("Gradient shape does not match HDF5 dataset '" + names[j] + "' shape.");
            offsets.push_back(internal::partition_offsets(value, inputs.size()));
        }
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            auto source = internal::open_file(inputs[i]);
            NamedArrayValues split;
            split.reserve(pieces.size());
            for (std::size_t j = 0; j < pieces.size(); ++j) {
                auto dataset = internal::dataset(source, names[j]);
                split.emplace_back(pieces[j].first,
                                   split_partition(pieces[j].second, dataset, offsets[j][i], offsets[j][i + 1] - offsets[j][i]));
            }
            source.reset();
            if (gradient.stacked) {
                std::vector<ArrayView> views;
                views.reserve(split.size());
                for (const auto& [key, value] : split) views.push_back(value.view());
                Array value;
                if (named) value = internal::stack(views);
                else value = std::move(split[0].second);
                gradient_contributions_impl(inputs[i], {true, {{"", value.view()}}, keys}, output, ancestors);
            } else {
                NamedArrays views;
                views.reserve(split.size());
                for (const auto& [key, value] : split) views.emplace_back(key, value.view());
                gradient_contributions_impl(inputs[i], {false, std::move(views), keys}, output, ancestors);
            }
        }
        ancestors.erase(identity);
    }

    static std::vector<GradientContribution> gradient_dict_contributions(const std::string& path,
                                                                         const NamedArrayValues& gradient,
                                                                         const std::vector<std::string>& keys) {
        NamedArrays views;
        views.reserve(gradient.size());
        for (const auto& [key, value] : gradient) views.emplace_back(key, value.view());
        std::vector<GradientContribution> output;
        std::set<std::string> ancestors;
        gradient_contributions_impl(path, {false, std::move(views), Columns(keys)}, output, ancestors);
        return output;
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
        // A TensorDict store whose leaves are all rank zero has no batch
        // dimensions. Reduction can prepend one batch axis, turning each leaf
        // into a vector with one entry per source.
        const bool scalar_store = first_source.store == "column" &&
            std::all_of(first_source.datasets.begin(), first_source.datasets.end(),
                        [](const auto& item) { return item.second.shape.empty(); });
        for (const auto& [name, first] : first_source.datasets) {
            if (first_source.store == "column" && first.shape.empty() && !scalar_store)
                throw std::invalid_argument("Cannot reduce TensorDict stores that mix scalar and non-scalar leaves.");
            for (const auto& input : inputs) {
                const auto& current = input.datasets.at(name);
                if (current.keys != first.keys) throw std::invalid_argument("Cannot reduce HDF5 dataset '" + name + "' with incompatible keys.");
                if (current.type != first.type) throw std::invalid_argument("Cannot reduce HDF5 dataset '" + name + "' with incompatible dtypes.");
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
