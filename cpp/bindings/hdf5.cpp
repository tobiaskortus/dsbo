#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dsbo/hdf5.hpp>

namespace py = pybind11;
namespace io = dsbo::io;
using namespace pybind11::literals;

namespace {
template<class Function> auto native(Function&& function) {
    py::gil_scoped_release release;
    return function();
}

io::DType numpy_type(const py::dtype& type) {
    io::Kind kind;
    switch (type.kind()) {
        case 'b': kind = io::Kind::boolean; break;
        case 'i': kind = io::Kind::signed_integer; break;
        case 'u': kind = io::Kind::unsigned_integer; break;
        case 'f': kind = io::Kind::floating; break;
        default: throw py::type_error("DSBO arrays require boolean, integer, or floating-point numeric values.");
    }
    const char order = type.byteorder();
    return {kind, static_cast<std::size_t>(type.itemsize()), order == '<' || (order != '>' && io::native_little_endian())};
}

py::dtype numpy_type(io::DType type) {
    char kind;
    switch (type.kind) {
        case io::Kind::boolean: kind = 'b'; break;
        case io::Kind::signed_integer: kind = 'i'; break;
        case io::Kind::unsigned_integer: kind = 'u'; break;
        case io::Kind::floating: kind = 'f'; break;
        default: throw py::type_error("Unsupported DSBO dtype.");
    }
    return py::dtype(std::string(type.size == 1 ? "|" : type.little_endian ? "<" : ">") + kind + std::to_string(type.size));
}

struct Input {
    py::array owner;
    io::ArrayView view;
    explicit Input(py::handle object) : owner(py::array::ensure(object, py::array::c_style)) {
        if (!owner) throw py::type_error("Expected a numeric array.");
        view.type = numpy_type(owner.dtype());
        for (py::ssize_t i = 0; i < owner.ndim(); ++i) view.shape.push_back(owner.shape(i));
        view.data = owner.data();
    }
};

io::Columns selected_columns(py::handle object) {
    if (object.is_none()) return std::nullopt;
    std::vector<std::string> result;
    for (const auto& column : py::reinterpret_borrow<py::iterable>(object)) {
        if (!py::isinstance<py::str>(column)) throw py::value_error("Column names must be valid, non-empty HDF5 names without '/'.");
        result.push_back(py::cast<std::string>(column));
    }
    io::internal::validate_keys(result);
    return result;
}

py::array array(io::Array value) {
    auto owner = std::make_unique<io::Array>(std::move(value));
    std::vector<py::ssize_t> shape(owner->shape.begin(), owner->shape.end()), strides(shape.size());
    py::ssize_t stride = static_cast<py::ssize_t>(owner->type.size);
    for (std::size_t i = shape.size(); i-- > 0;) { strides[i] = stride; stride *= shape[i]; }
    const auto type = numpy_type(owner->type);
    auto* pointer = owner.get();
    py::capsule lifetime(owner.release(), [](void* value) { delete static_cast<io::Array*>(value); });
    return py::array(type, shape, strides, pointer->storage.data(), lifetime);
}

py::dict arrays(io::NamedArrayValues values) {
    py::dict result;
    for (auto& [key, value] : values) result[py::str(key)] = array(std::move(value));
    return result;
}

py::object to_python(io::Array value) { return array(std::move(value)); }
py::object to_python(io::NamedArrayValues value) { return arrays(std::move(value)); }
py::object to_python(io::Shape value) { return py::cast(std::move(value)); }
py::dict to_python(std::vector<std::pair<std::string, io::Shape>> values) {
    py::dict result;
    for (auto& [key, shape] : values) result[py::str(key)] = py::cast(shape);
    return result;
}

template<class Function> py::object materialize(Function&& function) {
    return to_python(native(std::forward<Function>(function)));
}

template<class DictFunction, class ArrayFunction>
py::object read_value(bool dictionary, DictFunction&& dict_function, ArrayFunction&& array_function) {
    if (dictionary) return materialize(std::forward<DictFunction>(dict_function));
    return materialize(std::forward<ArrayFunction>(array_function));
}

template<class Function> py::object tangent_result(Function&& function) {
    auto result = native(std::forward<Function>(function));
    py::object tangent = result.second ? to_python(std::move(*result.second)) : py::object(py::none());
    return py::make_tuple(to_python(std::move(result.first)), tangent);
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

py::object statistics(std::optional<io::Statistics> value) {
    if (!value) return py::none();
    return py::make_tuple(value->count, array(std::move(value->mean)), array(std::move(value->squared_deviation)));
}

py::list contributions(std::vector<io::GradientContribution> values) {
    py::list result;
    for (auto& value : values) {
        py::object source;
        if (std::holds_alternative<std::string>(value.source)) source = py::str(std::get<std::string>(value.source));
        else if (std::holds_alternative<io::Array>(value.source)) source = array(std::move(std::get<io::Array>(value.source)));
        else source = arrays(std::move(std::get<io::NamedArrayValues>(value.source)));
        result.append(py::make_tuple(value.primal_path, source, py::cast(value.keys)));
    }
    return result;
}

io::Selection selection(py::handle index, hsize_t length) {
    if (length > static_cast<hsize_t>(PY_SSIZE_T_MAX)) throw std::overflow_error("Dataset exceeds Python indexing limits.");
    if (py::isinstance<py::slice>(index)) {
        py::ssize_t start, stop, step, count;
        if (!py::reinterpret_borrow<py::slice>(index).compute(static_cast<py::ssize_t>(length), &start, &stop, &step, &count))
            throw py::error_already_set();
        return {start, step, static_cast<std::size_t>(count), false};
    }
    py::object integer = py::reinterpret_steal<py::object>(PyNumber_Index(index.ptr()));
    if (!integer) throw py::error_already_set();
    auto value = py::cast<std::int64_t>(integer);
    if (value < 0) value += length;
    if (value < 0 || static_cast<hsize_t>(value) >= length) throw py::index_error("HDF5 dataset index is out of range.");
    return {value, 1, 1, true};
}

io::NamedArrays named_values(py::dict objects, std::vector<Input>& owners) {
    io::NamedArrays values;
    owners.reserve(objects.size());
    values.reserve(objects.size());
    for (const auto& item : objects) {
        owners.emplace_back(item.second);
        values.emplace_back(py::cast<std::string>(item.first), owners.back().view);
    }
    return values;
}

template<class Function> void named_write(const std::string& path, py::dict objects, Function&& function) {
    std::vector<Input> owners;
    const auto values = named_values(objects, owners);
    native([&] { function(path, values); });
}

template<class Function> void named_pair_write(const std::string& path, py::dict primal, py::dict tangent, Function&& function) {
    std::vector<Input> primal_owners, tangent_owners;
    const auto primal_values = named_values(primal, primal_owners);
    const auto tangent_values = named_values(tangent, tangent_owners);
    native([&] { function(path, primal_values, tangent_values); });
}

using DictWriter = void (*)(const std::string&, const io::NamedArrays&);
using DictTangentWriter = void (*)(const std::string&, const io::NamedArrays&, const io::NamedArrays&);

template<class DictFunction, class ArrayFunction>
void write_value(const std::string& path, py::handle value, DictFunction&& dict_writer, ArrayFunction&& array_writer) {
    if (py::isinstance<py::dict>(value)) {
        named_write(path, py::reinterpret_borrow<py::dict>(value), std::forward<DictFunction>(dict_writer));
        return;
    }
    Input input(value);
    native([&] { array_writer(path, input.view); });
}

template<class DictFunction, class ArrayFunction>
void write_tangent(const std::string& path, py::handle primal, py::handle tangent, DictFunction&& dict_writer, ArrayFunction&& array_writer) {
    const bool primal_dict = py::isinstance<py::dict>(primal), tangent_dict = py::isinstance<py::dict>(tangent);
    if (primal_dict != tangent_dict) throw py::type_error("Primal and tangent must both be arrays or both be dicts.");
    if (primal_dict) {
        named_pair_write(path, py::reinterpret_borrow<py::dict>(primal), py::reinterpret_borrow<py::dict>(tangent),
                         std::forward<DictFunction>(dict_writer));
        return;
    }
    Input p(primal), t(tangent);
    native([&] { array_writer(path, p.view, t.view); });
}

struct GradientWriter {
    bool merge;
    void operator()(const std::string& path, const io::NamedArrays& values) const {
        io::HDF5Storage::write_gradient(path, values, merge);
    }
};
} // namespace

PYBIND11_MODULE(_hdf5, module) {
    module.doc() = "DSBO's header-only native HDF5 storage implementation";
    py::register_local_exception_translator([](std::exception_ptr pointer) {
        try { if (pointer) std::rethrow_exception(pointer); }
        catch (const io::FileNotFoundError& error) { PyErr_SetString(PyExc_FileNotFoundError, error.what()); }
        catch (const io::IOError& error) { PyErr_SetString(PyExc_OSError, error.what()); }
        catch (const io::KeyError& error) { PyErr_SetString(PyExc_KeyError, error.what()); }
        catch (const io::TypeError& error) { PyErr_SetString(PyExc_TypeError, error.what()); }
        catch (const std::filesystem::filesystem_error& error) { PyErr_SetString(PyExc_OSError, error.what()); }
    });
    try { io::check_environment(); }
    catch (const std::exception& error) { throw py::import_error(error.what()); }

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
        return py::dict("version"_a=py::make_tuple(major, minor, patch), "threadsafe"_a=threadsafe);
    });

    module.def("open_primal", [](const std::string& path, py::object columns) -> py::object {
        const auto selected = selected_columns(columns);
        if (is_dict_store(path)) return arrays(native([&] { return io::HDF5Storage::open_dict(path, selected); }));
        if (selected) throw py::value_error("Flat HDF5 arrays do not support column selection.");
        return array(native([&] { return io::HDF5Storage::open_primal(path); }));
    }, py::arg("path"), py::arg("columns") = py::none());

    module.def("open_primal_tangent", [](const std::string& path, py::object columns) {
        const auto selected = selected_columns(columns);
        if (is_dict_store(path)) return tangent_result([&] { return io::HDF5Storage::open_dict_tangent(path, selected); });
        if (selected) throw py::value_error("Flat HDF5 arrays do not support column selection.");
        return tangent_result([&] { return io::HDF5Storage::open_primal_tangent(path); });
    }, py::arg("path"), py::arg("columns") = py::none());

    module.def("write_primal", [](const std::string& path, py::handle primal) {
        write_value(path, primal,
            static_cast<DictWriter>(&io::HDF5Storage::write_primal),
            [](const std::string& p, io::ArrayView view) { io::HDF5Storage::write_primal(p, view); });
    }, py::arg("path"), py::arg("primal"));

    module.def("write_primal_tangent", [](const std::string& path, py::handle primal, py::handle tangent) {
        write_tangent(path, primal, tangent,
            static_cast<DictTangentWriter>(&io::HDF5Storage::write_primal_tangent),
            [](const std::string& p, io::ArrayView primal_view, io::ArrayView tangent_view) {
                io::HDF5Storage::write_primal_tangent(p, primal_view, tangent_view);
            });
    }, py::arg("path"), py::arg("primal"), py::arg("tangent"));

    module.def("write_named_datasets", [](const std::string& path, py::dict values) {
        named_write(path, values, io::HDF5Storage::write_named_datasets);
    });

    module.def("write_row_store", [](const std::string& path, py::dict values) {
        named_write(path, values, io::HDF5Storage::write_row_store);
    });
    
    module.def("columns", [](const std::string& path) { return native([&] { return io::HDF5Storage::columns(path); }); });

    module.def("shape_from_path", [](const std::string& path) {
        return read_value(is_dict_store(path),
            [&] { return io::HDF5Storage::dict_shape_from_path(path); },
            [&] { return io::HDF5Storage::shape_from_path(path); });
    });

    module.def("load_gradient", [](const std::string& path, py::object columns) -> py::object {
        const auto selected = selected_columns(columns);
        if (is_dict_gradient(path)) return arrays(native([&] { return io::HDF5Storage::load_dict_gradient(path, selected); }));
        if (selected) throw py::value_error("Flat HDF5 gradients do not support column selection.");
        return array(native([&] { return io::HDF5Storage::load_gradient(path); }));
    }, py::arg("path"), py::arg("columns") = py::none());

    module.def("load_backward_gradient", [](const std::string& primal, const std::string& gradient) {
        return read_value(is_dict_store(primal),
            [&] { return io::HDF5Storage::load_backward_dict_gradient(primal, gradient); },
            [&] { return io::HDF5Storage::load_backward_gradient(primal, gradient); });
    });

    module.def("write_gradient", [](const std::string& path, py::handle gradient, bool merge) {
        write_value(path, gradient,
            GradientWriter{merge},
            [merge](const std::string& p, io::ArrayView view) {
                io::HDF5Storage::write_gradient(p, view, merge);
            });
    }, py::arg("path"), py::arg("gradient"), py::kw_only(), py::arg("merge") = false);

    module.def("aggregate_gradients", [](const std::vector<std::string>& paths, const std::string& destination) {
        native([&] { io::HDF5Storage::aggregate_gradients(paths, destination); });
    });

    module.def("reduce", [](const std::vector<std::string>& paths, const std::string& destination) {
        native([&] { io::HDF5Storage::reduce(paths, destination); });
    });

    module.def("gradient_file_contributions", [](const std::string& path, const std::string& gradient) {
        return contributions(native([&] { return io::HDF5Storage::gradient_file_contributions(path, gradient); }));
    });

    module.def("gradient_contributions", [](const std::string& path, py::handle gradient, py::object columns) {
        Input input(gradient);
        const auto selected = selected_columns(columns);
        return contributions(native([&] { return io::HDF5Storage::gradient_contributions(path, input.view, selected); }));
    }, py::arg("path"), py::arg("gradient"), py::arg("columns") = py::none());

    module.def("statistics", [](py::handle value, std::size_t workers) -> py::object {
        auto owner = py::array::ensure(value);
        if (!owner) throw py::type_error("Expected an array.");
        if (std::string("biuf").find(owner.dtype().kind()) == std::string::npos) return py::none();
        Input input(owner);
        return statistics(native([&] { return io::statistics(input.view, {workers}); }));
    }, py::arg("value"), py::arg("workers") = 0);
    
    module.def("combine_statistics", [](py::list values) {
        std::vector<std::optional<io::Statistics>> inputs;
        for (const auto& item : values) {
            if (item.is_none()) inputs.push_back(std::nullopt);
            else {
                auto parts = py::cast<py::tuple>(item);
                Input mean(parts[1]), deviation(parts[2]);
                const auto count = py::cast<std::uint64_t>(parts[0]);
                inputs.push_back(io::Statistics{count, io::convert(mean.view, io::dtype<double>()), io::convert(deviation.view, io::dtype<double>())});
            }
        }
        return statistics(native([&] { return io::combine_statistics(inputs); }));
    });

    py::class_<io::Reader>(module, "Reader")
        .def(py::init([](const std::string& path) { return native([&] { return std::make_unique<io::Reader>(path); }); }))
        .def_property_readonly("store", [](io::Reader& reader) { return native([&] { return reader.store(); }); })
        .def("keys", &io::Reader::keys, py::arg("root") = "", py::call_guard<py::gil_scoped_release>())
        .def("columns", &io::Reader::columns, py::arg("name") = "primal", py::call_guard<py::gil_scoped_release>())
        .def("row_keys", &io::Reader::row_keys, py::call_guard<py::gil_scoped_release>())
        .def("shape", &io::Reader::shape, py::call_guard<py::gil_scoped_release>())
        .def("dtype", [](io::Reader& reader, const std::string& name) { return numpy_type(native([&] { return reader.type(name); })); })
        .def("partition_sizes", &io::Reader::partition_sizes, py::call_guard<py::gil_scoped_release>())
        .def("statistics", [](io::Reader& reader, const std::string& name) { return statistics(native([&] { return reader.statistics(name); })); })
        .def("read", [](io::Reader& reader, const std::string& name, py::object index) {
            if (index.is_none()) return array(native([&] { return reader.read(name); }));
            const auto shape = native([&] { return reader.shape(name); });
            const auto selected = selection(index, shape.empty() ? 1 : shape[0]);
            return array(native([&] { return reader.read(name, selected); }));
        }, py::arg("name"), py::arg("index") = py::none())
        .def("read_indices", [](io::Reader& reader, const std::string& name, const std::vector<std::int64_t>& indices) {
            return array(native([&] { return reader.read_indices(name, indices); }));
        })
        .def("close", &io::Reader::close, py::call_guard<py::gil_scoped_release>());
}
