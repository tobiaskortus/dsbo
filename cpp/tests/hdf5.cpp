#include <dsbo/hdf5.hpp>
#include <cmath>
#include <future>
#include <iostream>

namespace io = dsbo::io;
namespace fs = std::filesystem;
io::DType second_translation_unit();
io::Statistics second_translation_unit_statistics(io::ArrayView value);

static void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function> static void expect_null_rejected(Function&& function) {
    try { function(); }
    catch (const std::invalid_argument& error) {
        expect(std::string(error.what()).find("H5S_NULL") != std::string::npos, "NULL dataspace errors identify H5S_NULL");
        return;
    }
    throw std::logic_error("NULL dataspace was accepted");
}

static void create_null_dataset(hid_t parent, const std::string& name) {
    io::internal::Handle space(H5Screate(H5S_NULL), H5Sclose);
    io::internal::Handle value(H5Dcreate2(parent, name.c_str(), H5T_NATIVE_DOUBLE, space,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
}

static void test_dataspace_classes(const std::string& path) {
    const auto scalar = io::Array::from<double>({3.5}, {});
    const io::Array empty(io::dtype<double>(), {0, 2});
    {
        auto file = io::internal::open_file(path, true);
        io::internal::create_dsbo_dataset(file, "scalar", scalar.view());
        io::internal::create_dsbo_dataset(file, "empty", empty.view());
        create_null_dataset(file, "null");
        io::internal::Handle space(H5Screate(H5S_NULL), H5Sclose);
        io::internal::Handle attribute(H5Acreate2(file, "null_attribute", H5T_NATIVE_DOUBLE, space,
                                             H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    }
    io::Reader reader(path);
    expect(reader.shape("scalar").empty(), "Scalar dataspaces retain rank zero");
    expect(reader.read("scalar").values<double>() == scalar.values<double>(), "Scalar values are preserved");
    expect(reader.read("scalar", {0, 1, 1, true}).values<double>() == scalar.values<double>(), "Scalar selections preserve values");
    expect(reader.read_indices("scalar", {0}).values<double>() == scalar.values<double>(), "Scalar indexed reads preserve values");
    expect(reader.shape("empty") == empty.shape && reader.read("empty").storage.empty(), "Zero-length simple dataspaces remain supported");
    expect_null_rejected([&] { reader.shape("null"); });
    expect_null_rejected([&] { reader.read("null"); });
    expect_null_rejected([&] { reader.read("null", {0, 1, 1, true}); });
    expect_null_rejected([&] { reader.read("null", {0, 1, 1, false}); });
    expect_null_rejected([&] { reader.read("null", {0, 1, 0, false}); });
    expect_null_rejected([&] { reader.read_indices("null", {0}); });
    expect_null_rejected([&] { reader.read_indices("null", {}); });
    auto file = io::internal::open_file(path);
    expect_null_rejected([&] { io::internal::read_array_attribute(file, "null_attribute", io::dtype<double>()); });
}

static void test_null_gradients(const std::string& primal_path, const std::string& gradient_path) {
    const auto scalar = io::Array::from<double>({3.5}, {});
    for (bool null_primal : {false, true}) for (bool null_gradient : {false, true}) {
        if (!null_primal && !null_gradient) continue;
        io::HDF5Storage::write_primal(primal_path, scalar.view());
        io::HDF5Storage::write_gradient(gradient_path, scalar.view());
        if (null_primal) {
            auto file = io::internal::open_file(primal_path, true, true);
            io::internal::check(H5Ldelete(file, "primal", H5P_DEFAULT), "Cannot replace test primal");
            create_null_dataset(file, "primal");
        }
        if (null_gradient) {
            auto file = io::internal::open_file(gradient_path, true, true);
            io::internal::check(H5Ldelete(file, "gradient", H5P_DEFAULT), "Cannot replace test gradient");
            create_null_dataset(file, "gradient");
            expect_null_rejected([&] { io::HDF5Storage::load_gradient(gradient_path); });
        }
        expect_null_rejected([&] { io::HDF5Storage::gradient_file_contributions(primal_path, gradient_path); });
        expect_null_rejected([&] { io::HDF5Storage::load_backward_gradient(primal_path, gradient_path); });
    }
}

static void test_dataset_discovery(const std::string& path) {
    const auto values = io::Array::from<double>({1, 2, 3}, {3});
    const auto external_path = fs::absolute(path + ".external.h5").string();
    {
        auto external = io::internal::open_file(external_path, true);
        io::internal::create_dsbo_dataset(external, "remote/values", values.view());
    }
    {
        auto file = io::internal::open_file(path, true);
        io::internal::create_dsbo_dataset(file, "primal", values.view());
        io::internal::create_dsbo_dataset(file, "nested/values", values.view());
        io::internal::require_group(file, "empty");
        io::internal::Handle type(H5Tcopy(H5T_NATIVE_INT), H5Tclose);
        io::internal::check(H5Tcommit2(file, "named_type", type, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "Cannot create test named datatype");
        io::internal::check(H5Lcreate_hard(file, "/", file, "cycle", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test root cycle");
        io::internal::check(H5Lcreate_hard(file, "nested", file, "nested/self", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test group cycle");
        io::internal::check(H5Lcreate_hard(file, "nested", file, "a_group", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test group alias");
        io::internal::check(H5Lcreate_hard(file, "primal", file, "a_primal", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test dataset alias");
        io::internal::check(H5Lcreate_hard(file, "nested/values", file, "nested/a_values", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test nested alias");
        io::internal::check(H5Lcreate_soft("/primal", file, "soft_dataset", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test soft dataset link");
        io::internal::check(H5Lcreate_soft("/nested", file, "soft_group", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test soft group link");
        io::internal::check(H5Lcreate_soft("/missing", file, "dangling", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test dangling link");
        io::internal::check(H5Lcreate_soft("/soft_cycle", file, "soft_cycle", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test soft cycle");
        io::internal::check(H5Lcreate_external(external_path.c_str(), "/remote/values", file, "external_dataset", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test external dataset link");
        io::internal::check(H5Lcreate_external(external_path.c_str(), "/remote", file, "external_group", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test external group link");
        io::internal::check(H5Lcreate_external((path + ".missing.h5").c_str(), "/", file, "dangling_external", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test dangling external link");
    }
    io::Reader reader(path);
    expect(reader.keys() == std::vector<std::string>({"a_group/a_values", "a_primal"}), "Discovery visits hard-linked objects once and skips symbolic links");
    expect(reader.keys("nested") == std::vector<std::string>({"nested/a_values"}), "Discovery preserves the requested root prefix");
    expect(reader.keys("/nested") == std::vector<std::string>({"/nested/a_values"}), "Discovery preserves absolute root prefixes");
    expect(reader.keys("empty").empty(), "Empty groups contain no datasets");
    expect(reader.read("a_group/a_values").storage == values.storage, "Discovered hard-link paths remain readable");
    expect(reader.read("soft_dataset").storage == values.storage, "Named reads still resolve soft links");
    expect(reader.read("external_dataset").storage == values.storage, "Named reads still resolve external links");
    auto file = io::internal::open_file(path);
    expect(io::internal::dataset_paths(file) == std::vector<std::string>({"a_group/a_values", "a_primal"}),
           "Shared traversal handles cycles and aliases using an existing file handle");
    expect(io::internal::dataset_paths(file, "/nested") == std::vector<std::string>({"/nested/a_values"}),
           "Shared traversal preserves the root prefix");
    expect(io::internal::dataset_paths(file, "empty").empty(), "Shared traversal supports empty groups");
}

static void test_string_attributes(const std::string& path) {
    auto file = io::internal::open_file(path, true);
    expect(!io::internal::string_attribute(file, io::store_attribute), "Missing string attributes remain optional");
    const std::vector<std::optional<io::Shape>> shapes = {
        io::Shape{}, io::Shape{1}, io::Shape{1, 1}, io::Shape{0},
        io::Shape{2}, io::Shape{100}, io::Shape{2, 2}, std::nullopt,
    };
    for (bool variable : {false, true}) {
        io::internal::Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
        constexpr std::size_t width = 8;
        io::internal::check(H5Tset_size(type, variable ? H5T_VARIABLE : width), "Cannot set test string size");
        for (const auto& shape : shapes) {
            auto space = shape ? io::internal::dataspace(*shape) : io::internal::Handle(H5Screate(H5S_NULL), H5Sclose);
            const auto count = shape ? io::elements(*shape) : 0;
            if (H5Aexists(file, io::store_attribute) > 0)
                io::internal::check(H5Adelete(file, io::store_attribute), "Cannot replace test string attribute");
            io::internal::Handle attribute(H5Acreate2(file, io::store_attribute, type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
            if (count) {
                if (variable) {
                    const std::vector<const char*> values(count, "flat");
                    io::internal::check(H5Awrite(attribute, type, values.data()), "Cannot write variable-length test strings");
                } else {
                    std::vector<char> values(count * width, '\0');
                    for (std::size_t i = 0; i < count; ++i) std::memcpy(values.data() + i * width, "flat", 4);
                    io::internal::check(H5Awrite(attribute, type, values.data()), "Cannot write fixed-length test strings");
                }
            }
            if (count == 1) {
                expect(io::internal::string_attribute(file, io::store_attribute).value() == "flat", "Scalar and singleton string attributes are readable");
            } else {
                try {
                    io::internal::string_attribute(file, io::store_attribute);
                    throw std::logic_error("Non-singleton string attribute was accepted");
                } catch (const std::invalid_argument& error) {
                    const std::string message = error.what();
                    expect(message.find(io::store_attribute) != std::string::npos &&
                           message.find("exactly one element") != std::string::npos, "String validation names the attribute and element-count requirement");
                }
            }
        }
    }
}

static void test_statistics_shapes() {
    const auto expect_rejected = [](const std::vector<std::optional<io::Statistics>>& inputs) {
        try { io::combine_statistics(inputs); }
        catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            expect(message.find("statistics") != std::string::npos && message.find("shapes") != std::string::npos,
                   "Statistics validation identifies incompatible shapes");
            return;
        }
        throw std::logic_error("Incompatible statistics shapes were accepted");
    };
    const io::Statistics valid{2, io::Array::from<double>({2, 3}, {2}), io::Array::from<double>({2, 2}, {2})};
    for (const auto& shape : {io::Shape{}, io::Shape{0}, io::Shape{1}, io::Shape{3}, io::Shape{1, 2}}) {
        for (bool malformed_mean : {false, true}) {
            auto invalid = valid;
            (malformed_mean ? invalid.mean : invalid.squared_deviation) = io::Array(io::dtype<double>(), shape);
            expect_rejected({invalid});
            expect_rejected({invalid, valid});
            expect_rejected({valid, invalid});
        }
    }
    const auto scalar = io::Array::from<double>({2}, {});
    const auto singleton = io::Array::from<double>({2}, {1});
    expect_rejected({io::Statistics{2, scalar, singleton}});
    expect_rejected({io::Statistics{2, scalar, scalar}, io::Statistics{2, singleton, singleton}});

    for (const auto& shape : {io::Shape{}, io::Shape{2}, io::Shape{0}}) {
        const auto array = [&](double value) {
            return io::Array::from<double>(std::vector<double>(io::elements(shape), value), shape);
        };
        const io::Statistics first{2, array(2), array(2)}, second{1, array(8), array(0)};
        const auto combined = io::combine_statistics({first, second});
        expect(combined && combined->count == 3, "Combined statistics retain the total count");
        expect(combined->mean.shape == shape && combined->squared_deviation.shape == shape,
               "Scalar, vector, and zero-width statistics retain their shapes");
        expect(combined->mean.values<double>() == array(4).values<double>() &&
               combined->squared_deviation.values<double>() == array(26).values<double>(),
               "Combined means and squared deviations match the complete sample");
        const auto single = io::combine_statistics({first});
        expect(single && single->count == first.count && single->mean.storage == first.mean.storage &&
               single->squared_deviation.storage == first.squared_deviation.storage, "Single-input statistics are preserved");
    }
    expect(!io::combine_statistics({}) && !io::combine_statistics({valid, std::nullopt}), "Missing statistics remain optional");
}

static void test_statistics_accuracy(const std::string& path) {
    const auto close = [](double actual, double expected) {
        return actual == expected || (std::isnan(actual) && std::isnan(expected)) ||
               (std::isfinite(actual) && std::isfinite(expected) && std::abs(actual - expected) <= 2e-15 * std::abs(expected));
    };
    const auto check = [&](const std::vector<double>& values, double mean, double squared_deviation) {
        for (bool matrix : {false, true}) {
            const std::size_t width = matrix ? 2 : 1;
            std::vector<double> data;
            data.reserve(values.size() * width);
            for (auto value : values) {
                data.push_back(value);
                if (matrix) data.push_back(-value);
            }
            const auto array = io::Array::from<double>(data, matrix ? io::Shape{values.size(), width} : io::Shape{values.size()});
            const auto verify = [&](const std::optional<io::Statistics>& stats) {
                expect(stats && stats->count == values.size(), "Accurate statistics retain the count");
                const auto shape = matrix ? io::Shape{width} : io::Shape{};
                expect(stats->mean.shape == shape && stats->squared_deviation.shape == shape, "Accurate statistics retain their shapes");
                const auto means = stats->mean.values<double>(), deviations = stats->squared_deviation.values<double>();
                for (std::size_t column = 0; column < width; ++column) {
                    const double expected_mean = column ? -mean : mean;
                    expect(squared_deviation == 0 ? means[column] == expected_mean : close(means[column], expected_mean),
                           "Statistics means resist cancellation and large offsets");
                    expect(close(deviations[column], squared_deviation), "Statistics squared deviations use stable summation");
                }
            };
            verify(io::statistics(array.view()));
            io::HDF5Storage::write_primal(path, array.view());
            verify(io::Reader(path).statistics("primal"));
        }
    };
    check({1e16, 1, -1e16}, 1.0 / 3, 2e32);
    std::vector<double> cancellation(100002, 1);
    cancellation.front() = 1e16;
    cancellation.back() = -1e16;
    check(cancellation, 100000.0 / 100002, 2e32);
    const double offset = 1e12 + 0.1;
    check(std::vector<double>(9, offset), offset, 0);
    check(std::vector<double>(1000000, offset), offset, 0);
    const double step = std::nextafter(offset, std::numeric_limits<double>::infinity()) - offset;
    check({offset, offset, offset + step}, offset, step * step * (2.0 / 3));
    std::vector<double> varied(100002);
    for (std::size_t i = 0; i < varied.size(); ++i) varied[i] = offset + (static_cast<int>(i % 3) - 1) * 0.25;
    check(varied, offset, 33334.0 / 8);
    std::vector<double> squared_sum(100002);
    for (std::size_t i = 1; i + 1 < squared_sum.size(); ++i) squared_sum[i] = i % 2 ? 1 : -1;
    squared_sum.front() = 1e8;
    squared_sum.back() = -1e8;
    check(squared_sum, 0, 2e16 + 100000);

    const double infinity = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
    check({nan, 1}, nan, nan);
    check({infinity, 1}, infinity, nan);
    check({-infinity, 1}, -infinity, nan);
    check({infinity, infinity}, infinity, nan);
    check({infinity, -infinity}, nan, nan);
    check({1e200, -1e200}, 0, infinity);
}

int main(int argc, char** argv) {
    try {
        io::check_environment();
        unsigned filter_info = 0;
        expect(H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0 &&
               H5Zget_filter_info(H5Z_FILTER_DEFLATE, &filter_info) >= 0 &&
               (filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED) &&
               (filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED), "Bundled zlib must provide gzip encoding/decoding");
        expect(second_translation_unit() == io::dtype<double>(), "Header must work across translation units");
        const auto truth = io::Array::from<std::int64_t>({-1, 0, 256}, {3});
        expect(io::convert(truth.view(), io::dtype<bool>()).values<bool>() == std::vector<bool>({true, false, true}), "Boolean conversion uses truth values");
        if (argc != 2) throw std::invalid_argument("Expected a test directory");
        const fs::path root = argv[1];
        fs::create_directories(root);
        const auto path = [&](const std::string& name) { return (root / name).string(); };
        test_string_attributes(path("string-attributes.h5"));
        test_dataset_discovery(path("discovery.h5"));
        test_dataspace_classes(path("dataspace-classes.h5"));
        test_null_gradients(path("null-primal.h5"), path("null-gradient.h5"));
        test_statistics_shapes();
        test_statistics_accuracy(path("statistics-accuracy.h5"));
        const auto extended = io::Array::from<long double>({1.25L, -2.5L}, {2});
        auto swapped_type = extended.type;
        swapped_type.little_endian = !swapped_type.little_endian;
        const auto swapped = io::convert(extended.view(), swapped_type);
        io::HDF5Storage::write_primal(path("extended.h5"), swapped.view());
        const auto extended_loaded = io::HDF5Storage::open_primal(path("extended.h5"));
        expect(extended_loaded.type == swapped_type && io::convert(extended_loaded.view(), extended.type).values<long double>() == extended.values<long double>(), "Extended float byte order");
        const auto a = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {3, 2});
        const auto linked_stats = second_translation_unit_statistics(a.view());
        expect(linked_stats.count == 3 && linked_stats.mean.values<double>() == std::vector<double>({3, 4}) &&
               linked_stats.squared_deviation.values<double>() == std::vector<double>({8, 8}),
               "Conversion and statistics work across mixed-header translation units");
        io::HDF5Storage::write_primal_tangent(path("a.h5"), a.view(), a.view());
        const auto [primal, tangent] = io::HDF5Storage::open_primal_tangent(path("a.h5"));
        expect(primal.storage == a.storage && tangent && tangent->storage == a.storage, "Primal/tangent round trip");
        io::Reader reader(path("a.h5"));
        expect(reader.read("primal", {2, -1, 3, false}).values<double>() == std::vector<double>({5, 6, 3, 4, 1, 2}), "Reverse hyperslab");
        expect(reader.read_indices("primal", {2, 0, 2, -2}).values<double>() == std::vector<double>({5, 6, 1, 2, 5, 6, 3, 4}), "Indexed rows preserve duplicates/order");
        const auto stats = reader.statistics("primal");
        expect(stats && stats->count == 3 && stats->mean.values<double>() == std::vector<double>({3, 4}), "Persisted statistics");
        reader.close();
        reader.close();
        try { reader.read("primal"); throw std::logic_error("Closed reader accepted access"); }
        catch (const std::runtime_error& error) { expect(std::string(error.what()).find("closed") != std::string::npos, "Closed-reader error"); }

        io::HDF5Storage::reduce({path("a.h5"), path("a.h5")}, path("vds.h5"));
        expect(io::HDF5Storage::open_primal(path("vds.h5")).shape == io::Shape({6, 2}), "VDS concatenation");
        expect(io::Reader(path("vds.h5")).statistics("primal")->count == 6, "Combined VDS statistics");
        const auto gradient = io::Array::from<double>(std::vector<double>(12, 2), {6, 2});
        io::HDF5Storage::write_gradient(path("adjoint.h5"), gradient.view());
        const auto contributions = io::HDF5Storage::gradient_file_contributions(path("vds.h5"), path("adjoint.h5"));
        expect(contributions.size() == 2 && std::get<io::Array>(contributions[0].source).shape == a.shape, "VDS adjoint transpose");

        const auto named = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {2, 3});
        io::HDF5Storage::write_primal(path("named.h5"), named.view(), std::vector<std::string>{"β", "first"});
        expect(io::HDF5Storage::columns(path("named.h5")) == std::vector<std::string>({"β", "first"}), "UTF-8 column order");
        const auto selected = io::Array::from<float>({7, 8, 9}, {1, 3});
        io::HDF5Storage::write_gradient(path("named_gradient.h5"), selected.view(), std::vector<std::string>{"first"});
        const auto backward = io::HDF5Storage::load_backward_gradient(path("named.h5"), path("named_gradient.h5"));
        expect(backward.values<float>() == std::vector<float>({0, 0, 0, 7, 8, 9}), "Missing gradient columns are zero-filled");
        const auto identity = io::HDF5Storage::gradient_file_contributions(path("named.h5"), path("named_gradient.h5"));
        expect(std::holds_alternative<std::string>(identity[0].source), "Physical gradient preserves the native file");

        const auto feature = io::Array::from<std::int64_t>({1, 2, 3}, {3});
        const auto other = io::Array::from<double>({0.5, 1.5, 2.5}, {3});
        io::HDF5Storage::write_row_store(path("rows.h5"), {{"feature/key", feature.view()}, {"energy", other.view()}});
        io::Reader rows(path("rows.h5"));
        expect(rows.row_keys() == std::vector<std::string>({"feature/key", "energy"}), "Row-store keys");
        expect(rows.read(io::row_store_dataset).values<double>() == std::vector<double>({1, 0.5, 2, 1.5, 3, 2.5}), "Row-store promotion/interleaving");

        const auto scalar = io::Array::from<double>({3.5}, {});
        io::HDF5Storage::write_primal(path("scalar.h5"), scalar.view());
        io::HDF5Storage::reduce({path("scalar.h5"), path("scalar.h5")}, path("scalar-vds.h5"));
        expect(io::HDF5Storage::open_primal(path("scalar-vds.h5")).values<double>() == std::vector<double>({3.5, 3.5}), "Scalar VDS");
        const io::Array empty(io::dtype<float>(), {0, 2});
        io::HDF5Storage::write_primal(path("empty.h5"), empty.view());
        io::HDF5Storage::reduce({path("empty.h5"), path("empty.h5")}, path("empty-vds.h5"));
        expect(io::HDF5Storage::open_primal(path("empty-vds.h5")).shape == io::Shape({0, 2}), "Empty VDS");

        const auto maximum = io::Array::from<std::uint64_t>({std::numeric_limits<std::uint64_t>::max()}, {1});
        const auto one = io::Array::from<std::uint64_t>({1}, {1});
        io::HDF5Storage::write_gradient(path("maximum.h5"), maximum.view());
        io::HDF5Storage::write_gradient(path("one.h5"), one.view());
        io::HDF5Storage::aggregate_gradients({path("maximum.h5"), path("one.h5")}, path("sum.h5"));
        expect(io::HDF5Storage::load_gradient(path("sum.h5")).values<std::uint64_t>()[0] == 0, "Integer gradients wrap without losing precision");

        std::vector<std::future<void>> workers;
        for (int i = 0; i < 4; ++i) workers.push_back(std::async(std::launch::async, [&, i] {
            const auto file = path("thread-" + std::to_string(i) + ".h5");
            for (int j = 0; j < 8; ++j) {
                io::HDF5Storage::write_primal(file, a.view());
                expect(io::HDF5Storage::open_primal(file).storage == a.storage, "Concurrent HDF5 calls");
            }
        }));
        for (auto& worker : workers) worker.get();
        std::cout << "Native HDF5 tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
