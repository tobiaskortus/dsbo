#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <dsbo/hdf5.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
namespace io = dsbo::io;

namespace {

template<class Function> auto native(Function&& function) {
    nb::gil_scoped_release release;
    return function();
}

nb::object numpy_module() {
    return nb::module_::import_("numpy");
}

std::string dtype_string(io::DType type) {
    char kind;
    switch (type.kind) {
        case io::Kind::boolean: return "|b1";
        case io::Kind::signed_integer: kind = 'i'; break;
        case io::Kind::unsigned_integer: kind = 'u'; break;
        case io::Kind::floating: kind = 'f'; break;
        default: throw nb::type_error("Unsupported DSBO dtype.");
    }
    if (type.size == 1) return std::string("|") + kind + "1";
    return std::string(1, kind) + std::to_string(type.size);
}

nb::dlpack::dtype dlpack_dtype(io::DType type) {
    uint8_t code;
    switch (type.kind) {
        case io::Kind::boolean: code = static_cast<uint8_t>(nb::dlpack::dtype_code::Bool); break;
        case io::Kind::signed_integer: code = static_cast<uint8_t>(nb::dlpack::dtype_code::Int); break;
        case io::Kind::unsigned_integer: code = static_cast<uint8_t>(nb::dlpack::dtype_code::UInt); break;
        case io::Kind::floating: code = static_cast<uint8_t>(nb::dlpack::dtype_code::Float); break;
        default: throw nb::type_error("Unsupported DSBO dtype.");
    }
    return {code, static_cast<uint8_t>(type.size * 8), 1};
}

nb::object numpy_dtype(io::DType type) {
    return numpy_module().attr("dtype")(dtype_string(type));
}

bool numeric_dtype(nb::dlpack::dtype dtype) {
    if (dtype.lanes != 1) return false;
    switch (static_cast<nb::dlpack::dtype_code>(dtype.code)) {
        case nb::dlpack::dtype_code::Bool: return dtype.bits == 8;
        case nb::dlpack::dtype_code::Int:
        case nb::dlpack::dtype_code::UInt:
            return dtype.bits == 8 || dtype.bits == 16 || dtype.bits == 32 || dtype.bits == 64;
        case nb::dlpack::dtype_code::Float:
            return dtype.bits == 16 || dtype.bits == 32 || dtype.bits == 64;
        default: return false;
    }
}

io::DType numpy_type(nb::dlpack::dtype dtype) {
    if (!numeric_dtype(dtype))
        throw nb::type_error("DSBO arrays require bool, 8/16/32/64-bit integers, or 16/32/64-bit floating-point scalars.");
    io::Kind kind;
    switch (dtype.code) {
        case static_cast<uint8_t>(nb::dlpack::dtype_code::Bool): kind = io::Kind::boolean; break;
        case static_cast<uint8_t>(nb::dlpack::dtype_code::Int): kind = io::Kind::signed_integer; break;
        case static_cast<uint8_t>(nb::dlpack::dtype_code::UInt): kind = io::Kind::unsigned_integer; break;
        case static_cast<uint8_t>(nb::dlpack::dtype_code::Float): kind = io::Kind::floating; break;
        default: throw nb::type_error("Unsupported DSBO dtype.");
    }
    return {kind, static_cast<std::size_t>(dtype.bits / 8)};
}

using NdArray = nb::ndarray<nb::numpy, nb::c_contig, nb::ro>;

NdArray as_array(nb::handle object) {
    NdArray result;
    if (nb::try_cast(object, result, true)) return result;
    nb::object numpy = numpy_module();
    nb::object array = numpy.attr("asarray")(object);
    if (!nb::cast<bool>(array.attr("flags").attr("c_contiguous")))
        array = numpy.attr("ascontiguousarray")(array);
    if (!nb::try_cast(array, result, false))
        throw nb::type_error("Expected a native-endian numeric array.");
    return result;
}

io::ArrayView view_of(const NdArray& array) {
    io::ArrayView view;
    view.type = numpy_type(array.dtype());
    for (std::size_t i = 0; i < array.ndim(); ++i) view.shape.push_back(array.shape(i));
    view.data = array.data();
    return view;
}

io::Columns selected_columns(nb::handle object) {
    if (object.is_none()) return std::nullopt;
    std::vector<std::string> result;
    for (nb::handle column : nb::borrow<nb::iterable>(object)) {
        if (!nb::isinstance<nb::str>(column)) throw nb::value_error("Column names must be valid, non-empty HDF5 names without '/'.");
        result.push_back(nb::cast<std::string>(column));
    }
    io::internal::validate_keys(result);
    return result;
}

nb::object array(io::Array value) {
    auto* owner = new io::Array(std::move(value));
    nb::capsule lifetime(owner, [](void* pointer) noexcept { delete static_cast<io::Array*>(pointer); });
    std::vector<std::size_t> shape(owner->shape.begin(), owner->shape.end());
    nb::ndarray<nb::numpy> result(
        owner->data(), shape.size(), shape.data(), lifetime,
        nullptr, dlpack_dtype(owner->type));
    return nb::cast(result);
}

nb::dict arrays(io::NamedArrayValues values) {
    nb::dict result;
    for (auto& [key, value] : values) result[nb::cast(key)] = array(std::move(value));
    return result;
}

nb::dict array_views(io::NamedArrayViews value) {
    auto* owner = new io::NamedArrayViews(std::move(value));
    nb::capsule lifetime(owner, [](void* pointer) noexcept { delete static_cast<io::NamedArrayViews*>(pointer); });
    nb::dict result;
    for (auto& [key, view] : owner->views) {
        std::vector<std::size_t> shape(view.shape.begin(), view.shape.end());
        nb::ndarray<nb::numpy> array(
            const_cast<void*>(view.data), shape.size(), shape.data(), lifetime,
            nullptr, dlpack_dtype(view.type));
        result[nb::cast(key)] = nb::cast(array);
    }
    return result;
}

nb::object to_python(io::Array value) { return array(std::move(value)); }
nb::object to_python(io::NamedArrayValues value) { return arrays(std::move(value)); }
nb::object to_python(io::Shape value) {
    return nb::cast(std::vector<hsize_t>(std::move(value)));
}
nb::dict to_python(std::vector<std::pair<std::string, io::Shape>> values) {
    nb::dict result;
    for (auto& [key, shape] : values)
        result[nb::cast(key)] = nb::cast(std::vector<hsize_t>(std::move(shape)));
    return result;
}

template<class Function> nb::object materialize(Function&& function) {
    return to_python(native(std::forward<Function>(function)));
}

template<class DictFunction, class ArrayFunction>
nb::object read_value(bool dictionary, DictFunction&& dict_function, ArrayFunction&& array_function) {
    if (dictionary) return materialize(std::forward<DictFunction>(dict_function));
    return materialize(std::forward<ArrayFunction>(array_function));
}

template<class Function> nb::object tangent_result(Function&& function) {
    auto result = native(std::forward<Function>(function));
    nb::object tangent = result.second ? to_python(std::move(*result.second)) : nb::none();
    return nb::make_tuple(to_python(std::move(result.first)), tangent);
}

bool is_dict_store(const std::string& path) {
    return !native([&] { return io::HDF5Storage::columns(path); }).empty();
}

bool is_dict_gradient(const std::string& path) {
    return native([&] {
        io::Reader reader(path);
        const bool result = !reader.columns("gradient").empty();
        reader.close();
        return result;
    });
}

nb::object statistics(std::optional<io::Statistics> value) {
    if (!value) return nb::none();
    return nb::make_tuple(value->count, array(std::move(value->mean)), array(std::move(value->squared_deviation)));
}

nb::list contributions(std::vector<io::GradientContribution> values) {
    nb::list result;
    for (auto& value : values) {
        nb::object source;
        if (std::holds_alternative<std::string>(value.source)) source = nb::cast(std::get<std::string>(value.source));
        else if (std::holds_alternative<io::Array>(value.source)) source = array(std::move(std::get<io::Array>(value.source)));
        else source = arrays(std::move(std::get<io::NamedArrayValues>(value.source)));
        result.append(nb::make_tuple(value.primal_path, source, value.keys));
    }
    return result;
}

io::Selection selection(nb::handle index, hsize_t length) {
    if (length > static_cast<hsize_t>(PY_SSIZE_T_MAX)) throw std::overflow_error("Dataset exceeds Python indexing limits.");
    if (nb::isinstance<nb::slice>(index)) {
        auto [start, stop, step, count] = nb::borrow<nb::slice>(index).compute(static_cast<std::size_t>(length));
        (void) stop;
        return {start, step, static_cast<std::size_t>(count), false};
    }
    nb::object integer = nb::steal<nb::object>(PyNumber_Index(index.ptr()));
    if (!integer) throw nb::python_error();
    auto value = nb::cast<std::int64_t>(integer);
    if (value < 0) value += static_cast<std::int64_t>(length);
    if (value < 0 || static_cast<hsize_t>(value) >= length) throw nb::index_error("HDF5 dataset index is out of range.");
    return {value, 1, 1, true};
}

io::NamedArrays named_values(nb::dict objects, std::vector<NdArray>& owners) {
    io::NamedArrays values;
    owners.reserve(objects.size());
    values.reserve(objects.size());
    for (const auto& [key, value] : objects) {
        owners.emplace_back(as_array(value));
        values.emplace_back(nb::cast<std::string>(key), view_of(owners.back()));
    }
    return values;
}

template<class Function> void named_write(const std::string& path, nb::dict objects, Function&& function) {
    std::vector<NdArray> owners;
    const auto values = named_values(objects, owners);
    native([&] { function(path, values); });
}

template<class Function> void named_pair_write(const std::string& path, nb::dict primal, nb::dict tangent, Function&& function) {
    std::vector<NdArray> primal_owners, tangent_owners;
    const auto primal_values = named_values(primal, primal_owners);
    const auto tangent_values = named_values(tangent, tangent_owners);
    native([&] { function(path, primal_values, tangent_values); });
}

using DictWriter = void (*)(const std::string&, const io::NamedArrays&);
using DictTangentWriter = void (*)(const std::string&, const io::NamedArrays&, const io::NamedArrays&);

template<class DictFunction, class ArrayFunction>
void write_value(const std::string& path, nb::handle value, DictFunction&& dict_writer, ArrayFunction&& array_writer) {
    if (nb::isinstance<nb::dict>(value)) {
        named_write(path, nb::borrow<nb::dict>(value), std::forward<DictFunction>(dict_writer));
        return;
    }
    NdArray owner = as_array(value);
    const io::ArrayView view = view_of(owner);
    native([&] { array_writer(path, view); });
}

template<class DictFunction, class ArrayFunction>
void write_tangent(const std::string& path, nb::handle primal, nb::handle tangent, DictFunction&& dict_writer, ArrayFunction&& array_writer) {
    const bool primal_dict = nb::isinstance<nb::dict>(primal), tangent_dict = nb::isinstance<nb::dict>(tangent);
    if (primal_dict != tangent_dict) throw nb::type_error("Primal and tangent must both be arrays or both be dicts.");
    if (primal_dict) {
        named_pair_write(path, nb::borrow<nb::dict>(primal), nb::borrow<nb::dict>(tangent),
                         std::forward<DictFunction>(dict_writer));
        return;
    }
    NdArray primal_owner = as_array(primal), tangent_owner = as_array(tangent);
    const io::ArrayView primal_view = view_of(primal_owner), tangent_view = view_of(tangent_owner);
    native([&] { array_writer(path, primal_view, tangent_view); });
}

struct GradientWriter {
    bool merge;
    void operator()(const std::string& path, const io::NamedArrays& values) const {
        io::HDF5Storage::write_gradient(path, values, merge);
    }
};
} // namespace

NB_MODULE(_hdf5, module) {
    module.doc() = "DSBO's header-only native HDF5 storage implementation";
    nb::register_exception_translator([](const std::exception_ptr& pointer, void*) {
        try { if (pointer) std::rethrow_exception(pointer); }
        catch (const io::FileNotFoundError& error) { PyErr_SetString(PyExc_FileNotFoundError, error.what()); }
        catch (const io::IOError& error) { PyErr_SetString(PyExc_OSError, error.what()); }
        catch (const io::KeyError& error) { PyErr_SetString(PyExc_KeyError, error.what()); }
        catch (const io::TypeError& error) { PyErr_SetString(PyExc_TypeError, error.what()); }
        catch (const std::filesystem::filesystem_error& error) { PyErr_SetString(PyExc_OSError, error.what()); }
        catch (...) { throw; }
    });
    try { io::check_environment(); }
    catch (const std::exception& error) { throw nb::import_error(error.what()); }

    module.attr("STORE_ATTRIBUTE") = io::store_attribute;
    module.attr("COLUMNS_ATTRIBUTE") = io::columns_attribute;
    module.attr("ROW_STORE_DATASET") = io::row_store_dataset;
    module.attr("ROW_STORE_KEYS_ATTRIBUTE") = io::keys_attribute;
    module.attr("VDS_SOURCES_ATTRIBUTE") = io::sources_attribute;
    module.attr("PARTITION_SIZES_ATTRIBUTE") = io::partitions_attribute;
    module.attr("STATS_COUNT_ATTRIBUTE") = io::count_attribute;
    module.attr("STATS_MEAN_ATTRIBUTE") = io::mean_attribute;
    module.attr("STATS_SQUARED_DEVIATION_ATTRIBUTE") = io::deviation_attribute;

    module.def("environment", [] {
        unsigned major, minor, patch;
        bool threadsafe = false;
        native([&] {
            io::internal::check(H5get_libversion(&major, &minor, &patch), "Cannot inspect HDF5 version.");
            io::internal::check(H5is_library_threadsafe(&threadsafe), "Cannot inspect HDF5 thread safety.");
        });
        nb::dict result;
        result["version"] = nb::make_tuple(major, minor, patch);
        result["threadsafe"] = threadsafe;
        return result;
    });

    module.def("open_primal", [](const std::string& path, nb::object columns) -> nb::object {
        const auto selected = selected_columns(columns);
        if (is_dict_store(path)) return arrays(native([&] { return io::HDF5Storage::open_dict(path, selected); }));
        if (selected) throw nb::value_error("Flat HDF5 arrays do not support column selection.");
        return array(native([&] { return io::HDF5Storage::open_primal(path); }));
    }, nb::arg("path"), nb::arg("columns") = nb::none());

    module.def("open_primal_tangent", [](const std::string& path, nb::object columns) -> nb::object {
        const auto selected = selected_columns(columns);
        if (is_dict_store(path)) return tangent_result([&] { return io::HDF5Storage::open_dict_tangent(path, selected); });
        if (selected) throw nb::value_error("Flat HDF5 arrays do not support column selection.");
        return tangent_result([&] { return io::HDF5Storage::open_primal_tangent(path); });
    }, nb::arg("path"), nb::arg("columns") = nb::none());

    module.def("write_primal", [](const std::string& path, nb::handle primal) {
        write_value(path, primal,
            static_cast<DictWriter>(&io::HDF5Storage::write_primal),
            [](const std::string& p, io::ArrayView view) { io::HDF5Storage::write_primal(p, view); });
    }, nb::arg("path"), nb::arg("primal"));

    module.def("write_primal_tangent", [](const std::string& path, nb::handle primal, nb::handle tangent) {
        write_tangent(path, primal, tangent,
            static_cast<DictTangentWriter>(&io::HDF5Storage::write_primal_tangent),
            [](const std::string& p, io::ArrayView primal_view, io::ArrayView tangent_view) {
                io::HDF5Storage::write_primal_tangent(p, primal_view, tangent_view);
            });
    }, nb::arg("path"), nb::arg("primal"), nb::arg("tangent"));

    module.def("write_named_datasets", [](const std::string& path, nb::dict values) {
        named_write(path, values, io::HDF5Storage::write_named_datasets);
    });

    module.def("write_row_store", [](const std::string& path, nb::dict values) {
        named_write(path, values, io::HDF5Storage::write_row_store);
    });

    module.def("columns", [](const std::string& path) { return native([&] { return io::HDF5Storage::columns(path); }); });

    module.def("shape_from_path", [](const std::string& path) {
        return read_value(is_dict_store(path),
            [&] { return io::HDF5Storage::dict_shape_from_path(path); },
            [&] { return io::HDF5Storage::shape_from_path(path); });
    });

    module.def("load_gradient", [](const std::string& path, nb::object columns) -> nb::object {
        const auto selected = selected_columns(columns);
        if (is_dict_gradient(path)) return arrays(native([&] { return io::HDF5Storage::load_dict_gradient(path, selected); }));
        if (selected) throw nb::value_error("Flat HDF5 gradients do not support column selection.");
        return array(native([&] { return io::HDF5Storage::load_gradient(path); }));
    }, nb::arg("path"), nb::arg("columns") = nb::none());

    module.def("load_backward_gradient", [](const std::string& primal, const std::string& gradient) {
        return read_value(is_dict_store(primal),
            [&] { return io::HDF5Storage::load_backward_dict_gradient(primal, gradient); },
            [&] { return io::HDF5Storage::load_backward_gradient(primal, gradient); });
    });

    module.def("write_gradient", [](const std::string& path, nb::handle gradient, bool merge) {
        write_value(path, gradient,
            GradientWriter{merge},
            [merge](const std::string& p, io::ArrayView view) {
                io::HDF5Storage::write_gradient(p, view, merge);
            });
    }, nb::arg("path"), nb::arg("gradient"), nb::kw_only(), nb::arg("merge") = false);

    module.def("aggregate_gradients", [](const std::vector<std::string>& paths, const std::string& destination) {
        native([&] { io::HDF5Storage::aggregate_gradients(paths, destination); });
    });

    module.def("reduce", [](const std::vector<std::string>& paths, const std::string& destination) {
        native([&] { io::HDF5Storage::reduce(paths, destination); });
    });

    module.def("gradient_file_contributions", [](const std::string& path, const std::string& gradient) {
        return contributions(native([&] { return io::HDF5Storage::gradient_file_contributions(path, gradient); }));
    });

    module.def("gradient_contributions", [](const std::string& path, nb::handle gradient, nb::object columns) {
        NdArray owner = as_array(gradient);
        const io::ArrayView view = view_of(owner);
        const auto selected = selected_columns(columns);
        return contributions(native([&] { return io::HDF5Storage::gradient_contributions(path, view, selected); }));
    }, nb::arg("path"), nb::arg("gradient"), nb::arg("columns") = nb::none());

    module.def("statistics", [](nb::handle value, std::size_t workers) -> nb::object {
        NdArray array;
        try { array = as_array(value); }
        catch (const nb::builtin_exception&) { return nb::none(); }
        if (!numeric_dtype(array.dtype())) return nb::none();
        const io::ArrayView view = view_of(array);
        return statistics(native([&] { return io::statistics(view, {workers}); }));
    }, nb::arg("value"), nb::arg("workers") = 0);

    module.def("combine_statistics", [](nb::list values) -> nb::object {
        std::vector<std::optional<io::Statistics>> inputs;
        for (nb::handle item : values) {
            if (item.is_none()) inputs.push_back(std::nullopt);
            else {
                nb::tuple parts = nb::cast<nb::tuple>(item);
                nb::object mean = parts[1];
                nb::object deviation = parts[2];
                NdArray mean_owner = as_array(mean), deviation_owner = as_array(deviation);
                const io::ArrayView mean_view = view_of(mean_owner), deviation_view = view_of(deviation_owner);
                const auto count = nb::cast<std::uint64_t>(parts[0]);
                inputs.push_back(io::Statistics{count,
                    io::convert(mean_view, io::dtype<double>()),
                    io::convert(deviation_view, io::dtype<double>())});
            }
        }
        return statistics(native([&] { return io::combine_statistics(inputs); }));
    });

    nb::class_<io::Reader>(module, "Reader")
        .def(nb::init<const std::string&>(), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("store", [](io::Reader& reader) { return native([&] { return reader.store(); }); })
        .def("keys", &io::Reader::keys, nb::arg("root") = "", nb::call_guard<nb::gil_scoped_release>())
        .def("columns", &io::Reader::columns, nb::arg("name") = "primal", nb::call_guard<nb::gil_scoped_release>())
        .def("row_keys", &io::Reader::row_keys, nb::call_guard<nb::gil_scoped_release>())
        .def("shape", [](io::Reader& reader, const std::string& name) {
            return to_python(native([&] { return reader.shape(name); }));
        }, nb::arg("name"))
        .def("dtype", [](io::Reader& reader, const std::string& name) {
            return numpy_dtype(native([&] { return reader.type(name); }));
        })
        .def("partition_sizes", &io::Reader::partition_sizes, nb::call_guard<nb::gil_scoped_release>())
        .def("statistics", [](io::Reader& reader, const std::string& name) {
            return statistics(native([&] { return reader.statistics(name); }));
        })
        .def("read", [](io::Reader& reader, const std::string& name, nb::object index) -> nb::object {
            if (index.is_none()) return array(native([&] { return reader.read(name); }));
            const auto shape = native([&] { return reader.shape(name); });
            const auto selected = selection(index, shape.empty() ? 1 : shape[0]);
            return array(native([&] { return reader.read(name, selected); }));
        }, nb::arg("name"), nb::arg("index") = nb::none())
        .def("read_indices", [](io::Reader& reader, const std::string& name, const std::vector<std::int64_t>& indices) {
            return array(native([&] { return reader.read_indices(name, indices); }));
        })
        .def("read_indices_many", [](io::Reader& reader, const std::vector<std::string>& names,
                                     const std::vector<std::int64_t>& indices) {
            return array_views(native([&] { return reader.read_indices_many(names, indices); }));
        })
        .def("close", &io::Reader::close, nb::call_guard<nb::gil_scoped_release>());
}
