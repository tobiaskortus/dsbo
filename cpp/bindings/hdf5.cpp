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
        case 'c': kind = io::Kind::complex; break;
        default: throw py::type_error("DSBO arrays require boolean, integer, floating-point, or complex numeric values.");
    }
    const char order = type.byteorder();
    return {kind, static_cast<std::size_t>(type.itemsize()), order == '<' || (order != '>' && io::native_little_endian())};
}

py::dtype numpy_type(io::DType type) {
    const char kind = type.kind == io::Kind::boolean ? 'b' : type.kind == io::Kind::signed_integer ? 'i' :
                      type.kind == io::Kind::unsigned_integer ? 'u' : type.kind == io::Kind::floating ? 'f' : 'c';
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
    io::internal::validate_columns(result);
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

py::object statistics(std::optional<io::Statistics> value) {
    if (!value) return py::none();
    return py::make_tuple(value->count, array(std::move(value->mean)), array(std::move(value->squared_deviation)));
}

py::list contributions(std::vector<io::GradientContribution> values) {
    py::list result;
    for (auto& value : values) {
        py::object source;
        if (std::holds_alternative<std::string>(value.source)) source = py::str(std::get<std::string>(value.source));
        else source = array(std::move(std::get<io::Array>(value.source)));
        result.append(py::make_tuple(value.primal_path, source, py::cast(value.columns)));
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

template<class Function> void named_write(const std::string& path, py::dict objects, Function&& function) {
    std::vector<Input> inputs;
    io::NamedArrays values;
    inputs.reserve(objects.size());
    for (const auto& item : objects) {
        if (!py::isinstance<py::str>(item.first)) throw py::value_error("Feature keys must be non-empty strings without NUL characters.");
        inputs.emplace_back(item.second);
        values.emplace_back(py::cast<std::string>(item.first), inputs.back().view);
    }
    native([&] { function(path, values); });
}
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

    module.def("open_primal", [](const std::string& path, py::object columns) {
        const auto selected = selected_columns(columns);
        return array(native([&] { return io::HDF5Storage::open_primal(path, selected); }));
    }, py::arg("path"), py::arg("columns") = py::none());
    module.def("open_primal_tangent", [](const std::string& path, py::object columns) {
        const auto selected = selected_columns(columns);
        auto result = native([&] { return io::HDF5Storage::open_primal_tangent(path, selected); });
        py::object tangent = result.second ? py::object(array(std::move(*result.second))) : py::object(py::none());
        return py::make_tuple(array(std::move(result.first)), tangent);
    }, py::arg("path"), py::arg("columns") = py::none());
    module.def("write_primal", [](const std::string& path, py::handle primal, py::object columns) {
        Input input(primal);
        const auto selected = selected_columns(columns);
        native([&] { io::HDF5Storage::write_primal(path, input.view, selected); });
    }, py::arg("path"), py::arg("primal"), py::arg("columns") = py::none());
    module.def("write_primal_tangent", [](const std::string& path, py::handle primal, py::handle tangent, py::object columns) {
        Input p(primal), t(tangent);
        const auto selected = selected_columns(columns);
        native([&] { io::HDF5Storage::write_primal_tangent(path, p.view, t.view, selected); });
    }, py::arg("path"), py::arg("primal"), py::arg("tangent"), py::arg("columns") = py::none());
    module.def("write_named_datasets", [](const std::string& path, py::dict values) {
        named_write(path, values, io::HDF5Storage::write_named_datasets);
    });
    module.def("write_row_store", [](const std::string& path, py::dict values) {
        named_write(path, values, io::HDF5Storage::write_row_store);
    });
    module.def("columns", [](const std::string& path) { return native([&] { return io::HDF5Storage::columns(path); }); });
    module.def("shape_from_path", [](const std::string& path) { return native([&] { return io::HDF5Storage::shape_from_path(path); }); });
    module.def("load_gradient", [](const std::string& path, py::object columns) {
        const auto selected = selected_columns(columns);
        return array(native([&] { return io::HDF5Storage::load_gradient(path, selected); }));
    }, py::arg("path"), py::arg("columns") = py::none());
    module.def("load_backward_gradient", [](const std::string& primal, const std::string& gradient) {
        return array(native([&] { return io::HDF5Storage::load_backward_gradient(primal, gradient); }));
    });
    module.def("write_gradient", [](const std::string& path, py::handle gradient, py::object columns, bool merge) {
        Input input(gradient);
        const auto selected = selected_columns(columns);
        native([&] { io::HDF5Storage::write_gradient(path, input.view, selected, merge); });
    }, py::arg("path"), py::arg("gradient"), py::arg("columns") = py::none(), py::kw_only(), py::arg("merge") = false);
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
    module.def("statistics", [](py::handle value) -> py::object {
        auto owner = py::array::ensure(value);
        if (!owner) throw py::type_error("Expected an array.");
        if (std::string("biuf").find(owner.dtype().kind()) == std::string::npos) return py::none();
        Input input(owner);
        return statistics(native([&] { return io::statistics(input.view); }));
    });
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
