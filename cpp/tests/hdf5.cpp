#include <dsbo/hdf5.hpp>
#include <future>
#include <iostream>

namespace io = dsbo::io;
namespace fs = std::filesystem;
io::DType second_translation_unit();

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
    io::detail::Handle space(H5Screate(H5S_NULL), H5Sclose);
    io::detail::Handle value(H5Dcreate2(parent, name.c_str(), H5T_NATIVE_DOUBLE, space,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
}

static void test_dataspace_classes(const std::string& path) {
    const auto scalar = io::Array::from<double>({3.5}, {});
    const io::Array empty(io::dtype<double>(), {0, 2});
    {
        auto file = io::detail::open_file(path, true);
        io::detail::create_dataset(file, "scalar", scalar.view());
        io::detail::create_dataset(file, "empty", empty.view());
        create_null_dataset(file, "null");
        io::detail::Handle space(H5Screate(H5S_NULL), H5Sclose);
        io::detail::Handle attribute(H5Acreate2(file, "null_attribute", H5T_NATIVE_DOUBLE, space,
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
    auto file = io::detail::open_file(path);
    expect_null_rejected([&] { io::detail::read_array_attribute(file, "null_attribute", io::dtype<double>()); });
}

static void test_null_gradients(const std::string& primal_path, const std::string& gradient_path) {
    const auto scalar = io::Array::from<double>({3.5}, {});
    for (bool null_primal : {false, true}) for (bool null_gradient : {false, true}) {
        if (!null_primal && !null_gradient) continue;
        io::HDF5Writer::write_primal(primal_path, scalar.view());
        io::HDF5Writer::write_gradient(gradient_path, scalar.view());
        if (null_primal) {
            auto file = io::detail::open_file(primal_path, true, true);
            io::detail::check(H5Ldelete(file, "primal", H5P_DEFAULT), "Cannot replace test primal");
            create_null_dataset(file, "primal");
        }
        if (null_gradient) {
            auto file = io::detail::open_file(gradient_path, true, true);
            io::detail::check(H5Ldelete(file, "gradient", H5P_DEFAULT), "Cannot replace test gradient");
            create_null_dataset(file, "gradient");
            expect_null_rejected([&] { io::HDF5Writer::load_gradient(gradient_path); });
        }
        expect_null_rejected([&] { io::HDF5Writer::gradient_file_contributions(primal_path, gradient_path); });
        expect_null_rejected([&] { io::HDF5Writer::load_backward_gradient(primal_path, gradient_path); });
    }
}

static void test_dataset_discovery(const std::string& path) {
    const auto values = io::Array::from<double>({1, 2, 3}, {3});
    const auto external_path = fs::absolute(path + ".external.h5").string();
    {
        auto external = io::detail::open_file(external_path, true);
        io::detail::create_dataset(external, "remote/values", values.view());
    }
    {
        auto file = io::detail::open_file(path, true);
        io::detail::create_dataset(file, "primal", values.view());
        io::detail::create_dataset(file, "nested/values", values.view());
        io::detail::require_group(file, "empty");
        io::detail::Handle type(H5Tcopy(H5T_NATIVE_INT), H5Tclose);
        io::detail::check(H5Tcommit2(file, "named_type", type, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "Cannot create test named datatype");
        io::detail::check(H5Lcreate_hard(file, "/", file, "cycle", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test root cycle");
        io::detail::check(H5Lcreate_hard(file, "nested", file, "nested/self", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test group cycle");
        io::detail::check(H5Lcreate_hard(file, "nested", file, "a_group", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test group alias");
        io::detail::check(H5Lcreate_hard(file, "primal", file, "a_primal", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test dataset alias");
        io::detail::check(H5Lcreate_hard(file, "nested/values", file, "nested/a_values", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test nested alias");
        io::detail::check(H5Lcreate_soft("/primal", file, "soft_dataset", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test soft dataset link");
        io::detail::check(H5Lcreate_soft("/nested", file, "soft_group", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test soft group link");
        io::detail::check(H5Lcreate_soft("/missing", file, "dangling", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test dangling link");
        io::detail::check(H5Lcreate_soft("/soft_cycle", file, "soft_cycle", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test soft cycle");
        io::detail::check(H5Lcreate_external(external_path.c_str(), "/remote/values", file, "external_dataset", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test external dataset link");
        io::detail::check(H5Lcreate_external(external_path.c_str(), "/remote", file, "external_group", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test external group link");
        io::detail::check(H5Lcreate_external((path + ".missing.h5").c_str(), "/", file, "dangling_external", H5P_DEFAULT, H5P_DEFAULT), "Cannot create test dangling external link");
    }
    io::Reader reader(path);
    expect(reader.keys() == std::vector<std::string>({"a_group/a_values", "a_primal"}), "Discovery visits hard-linked objects once and skips symbolic links");
    expect(reader.keys("nested") == std::vector<std::string>({"nested/a_values"}), "Discovery preserves the requested root prefix");
    expect(reader.keys("/nested") == std::vector<std::string>({"/nested/a_values"}), "Discovery preserves absolute root prefixes");
    expect(reader.keys("empty").empty(), "Empty groups contain no datasets");
    expect(reader.read("a_group/a_values").storage == values.storage, "Discovered hard-link paths remain readable");
    expect(reader.read("soft_dataset").storage == values.storage, "Named reads still resolve soft links");
    expect(reader.read("external_dataset").storage == values.storage, "Named reads still resolve external links");
}

static void test_string_attributes(const std::string& path) {
    auto file = io::detail::open_file(path, true);
    expect(!io::detail::string_attribute(file, io::store_attribute), "Missing string attributes remain optional");
    const std::vector<std::optional<io::Shape>> shapes = {
        io::Shape{}, io::Shape{1}, io::Shape{1, 1}, io::Shape{0},
        io::Shape{2}, io::Shape{100}, io::Shape{2, 2}, std::nullopt,
    };
    for (bool variable : {false, true}) {
        io::detail::Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
        constexpr std::size_t width = 8;
        io::detail::check(H5Tset_size(type, variable ? H5T_VARIABLE : width), "Cannot set test string size");
        for (const auto& shape : shapes) {
            auto space = shape ? io::detail::dataspace(*shape) : io::detail::Handle(H5Screate(H5S_NULL), H5Sclose);
            const auto count = shape ? io::elements(*shape) : 0;
            if (H5Aexists(file, io::store_attribute) > 0)
                io::detail::check(H5Adelete(file, io::store_attribute), "Cannot replace test string attribute");
            io::detail::Handle attribute(H5Acreate2(file, io::store_attribute, type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
            if (count) {
                if (variable) {
                    const std::vector<const char*> values(count, "flat");
                    io::detail::check(H5Awrite(attribute, type, values.data()), "Cannot write variable-length test strings");
                } else {
                    std::vector<char> values(count * width, '\0');
                    for (std::size_t i = 0; i < count; ++i) std::memcpy(values.data() + i * width, "flat", 4);
                    io::detail::check(H5Awrite(attribute, type, values.data()), "Cannot write fixed-length test strings");
                }
            }
            if (count == 1) {
                expect(io::detail::string_attribute(file, io::store_attribute).value() == "flat", "Scalar and singleton string attributes are readable");
            } else {
                try {
                    io::detail::string_attribute(file, io::store_attribute);
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
        const auto extended = io::Array::from<long double>({1.25L, -2.5L}, {2});
        auto swapped_type = extended.type;
        swapped_type.little_endian = !swapped_type.little_endian;
        const auto swapped = io::convert(extended.view(), swapped_type);
        io::HDF5Writer::write_primal(path("extended.h5"), swapped.view());
        const auto extended_loaded = io::HDF5Writer::open_primal(path("extended.h5"));
        expect(extended_loaded.type == swapped_type && io::convert(extended_loaded.view(), extended.type).values<long double>() == extended.values<long double>(), "Extended float byte order");
        const auto a = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {3, 2});
        io::HDF5Writer::write_primal_tangent(path("a.h5"), a.view(), a.view());
        const auto [primal, tangent] = io::HDF5Writer::open_primal_tangent(path("a.h5"));
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

        io::HDF5Writer::reduce({path("a.h5"), path("a.h5")}, path("vds.h5"));
        expect(io::HDF5Writer::open_primal(path("vds.h5")).shape == io::Shape({6, 2}), "VDS concatenation");
        expect(io::Reader(path("vds.h5")).statistics("primal")->count == 6, "Combined VDS statistics");
        const auto gradient = io::Array::from<double>(std::vector<double>(12, 2), {6, 2});
        io::HDF5Writer::write_gradient(path("adjoint.h5"), gradient.view());
        const auto contributions = io::HDF5Writer::gradient_file_contributions(path("vds.h5"), path("adjoint.h5"));
        expect(contributions.size() == 2 && std::get<io::Array>(contributions[0].source).shape == a.shape, "VDS adjoint transpose");

        const auto named = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {2, 3});
        io::HDF5Writer::write_primal(path("named.h5"), named.view(), std::vector<std::string>{"β", "first"});
        expect(io::HDF5Writer::columns(path("named.h5")) == std::vector<std::string>({"β", "first"}), "UTF-8 column order");
        const auto selected = io::Array::from<float>({7, 8, 9}, {1, 3});
        io::HDF5Writer::write_gradient(path("named_gradient.h5"), selected.view(), std::vector<std::string>{"first"});
        const auto backward = io::HDF5Writer::load_backward_gradient(path("named.h5"), path("named_gradient.h5"));
        expect(backward.values<float>() == std::vector<float>({0, 0, 0, 7, 8, 9}), "Missing gradient columns are zero-filled");
        const auto identity = io::HDF5Writer::gradient_file_contributions(path("named.h5"), path("named_gradient.h5"));
        expect(std::holds_alternative<std::string>(identity[0].source), "Physical gradient preserves the native file");

        const auto feature = io::Array::from<std::int64_t>({1, 2, 3}, {3});
        const auto other = io::Array::from<double>({0.5, 1.5, 2.5}, {3});
        io::HDF5Writer::write_row_store(path("rows.h5"), {{"feature/key", feature.view()}, {"energy", other.view()}});
        io::Reader rows(path("rows.h5"));
        expect(rows.row_keys() == std::vector<std::string>({"feature/key", "energy"}), "Row-store keys");
        expect(rows.read(io::row_store_dataset).values<double>() == std::vector<double>({1, 0.5, 2, 1.5, 3, 2.5}), "Row-store promotion/interleaving");

        const auto scalar = io::Array::from<double>({3.5}, {});
        io::HDF5Writer::write_primal(path("scalar.h5"), scalar.view());
        io::HDF5Writer::reduce({path("scalar.h5"), path("scalar.h5")}, path("scalar-vds.h5"));
        expect(io::HDF5Writer::open_primal(path("scalar-vds.h5")).values<double>() == std::vector<double>({3.5, 3.5}), "Scalar VDS");
        const io::Array empty(io::dtype<float>(), {0, 2});
        io::HDF5Writer::write_primal(path("empty.h5"), empty.view());
        io::HDF5Writer::reduce({path("empty.h5"), path("empty.h5")}, path("empty-vds.h5"));
        expect(io::HDF5Writer::open_primal(path("empty-vds.h5")).shape == io::Shape({0, 2}), "Empty VDS");

        const auto maximum = io::Array::from<std::uint64_t>({std::numeric_limits<std::uint64_t>::max()}, {1});
        const auto one = io::Array::from<std::uint64_t>({1}, {1});
        io::HDF5Writer::write_gradient(path("maximum.h5"), maximum.view());
        io::HDF5Writer::write_gradient(path("one.h5"), one.view());
        io::HDF5Writer::aggregate_gradients({path("maximum.h5"), path("one.h5")}, path("sum.h5"));
        expect(io::HDF5Writer::load_gradient(path("sum.h5")).values<std::uint64_t>()[0] == 0, "Integer gradients wrap without losing precision");

        std::vector<std::future<void>> workers;
        for (int i = 0; i < 4; ++i) workers.push_back(std::async(std::launch::async, [&, i] {
            const auto file = path("thread-" + std::to_string(i) + ".h5");
            for (int j = 0; j < 8; ++j) {
                io::HDF5Writer::write_primal(file, a.view());
                expect(io::HDF5Writer::open_primal(file).storage == a.storage, "Concurrent HDF5 calls");
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
