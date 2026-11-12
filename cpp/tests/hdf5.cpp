#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <dsbo/hdf5.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "test_support.hpp"

namespace io = dsbo::io;
namespace fs = std::filesystem;
using dsbo::test::TestDirectory;

io::DType double_dtype_from_separate_cpp_file();
io::Statistics double_statistics_from_separate_cpp_file(io::ArrayView value);

struct Hdf5Fixture {
    Hdf5Fixture() { io::check_environment(); }
};

template<class Function> 
static void expect_null_rejected(Function&& function) {
    CHECK_THROWS_WITH_AS(function(), doctest::Contains("H5S_NULL"), std::invalid_argument);
}

template<class Function>
static void expect_invalid_argument(Function&& function, const std::string& first,
                                    const std::string& second = {}) {
    try {
        function();
        FAIL("Expected std::invalid_argument");
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        CHECK(message.find(first) != std::string::npos);
        if (!second.empty()) CHECK(message.find(second) != std::string::npos);
    }
}

static void create_null_dataset(hid_t parent, const std::string& name) {
    io::internal::Handle space(H5Screate(H5S_NULL), H5Sclose);
    io::internal::Handle value(H5Dcreate2(parent, name.c_str(), H5T_NATIVE_DOUBLE, space,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
}

static void replace_with_null_dataset(const std::string& path, const std::string& name) {
    auto file = io::internal::open_file(path, true, true);
    io::internal::check(H5Ldelete(file, name.c_str(), H5P_DEFAULT), "Cannot replace test dataset");
    create_null_dataset(file, name);
}

static void write_scalar_pair(const std::string& primal_path, const std::string& gradient_path) {
    const auto scalar = io::Array::from<double>({3.5}, {});
    io::HDF5Storage::write_primal(primal_path, scalar.view());
    io::HDF5Storage::write_gradient(gradient_path, scalar.view());
}

static void create_hdf5_files_with_links_and_cycles(const std::string& path) {
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
        io::internal::check(H5Tcommit2(file, "named_type", type, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test named datatype");
        io::internal::check(H5Lcreate_hard(file, "/", file, "cycle", H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test root cycle");
        io::internal::check(H5Lcreate_hard(file, "nested", file, "nested/self", H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test group cycle");
        io::internal::check(H5Lcreate_hard(file, "nested", file, "a_group", H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test group alias");
        io::internal::check(H5Lcreate_hard(file, "primal", file, "a_primal", H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test dataset alias");
        io::internal::check(H5Lcreate_hard(file, "nested/values", file, "nested/a_values", H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test nested alias");
        io::internal::check(H5Lcreate_soft("/primal", file, "soft_dataset", H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test soft dataset link");
        io::internal::check(H5Lcreate_soft("/nested", file, "soft_group", H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test soft group link");
        io::internal::check(H5Lcreate_soft("/missing", file, "dangling", H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test dangling link");
        io::internal::check(H5Lcreate_soft("/soft_cycle", file, "soft_cycle", H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test soft cycle");
        io::internal::check(H5Lcreate_external(external_path.c_str(), "/remote/values", file, "external_dataset",
                                               H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test external dataset link");
        io::internal::check(H5Lcreate_external(external_path.c_str(), "/remote", file, "external_group",
                                               H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test external group link");
        io::internal::check(H5Lcreate_external((path + ".missing.h5").c_str(), "/", file, "dangling_external",
                                               H5P_DEFAULT, H5P_DEFAULT),
                            "Cannot create test dangling external link");
    }
}

static void write_string_attribute(hid_t file, bool variable, const std::optional<io::Shape>& shape,
                                   const std::string& value) {
    if (H5Aexists(file, io::store_attribute) > 0)
        io::internal::check(H5Adelete(file, io::store_attribute), "Cannot replace test string attribute");
    io::internal::Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    constexpr std::size_t width = 8;
    io::internal::check(H5Tset_size(type, variable ? H5T_VARIABLE : width), "Cannot set test string size");
    auto space = shape ? io::internal::dataspace(*shape) : io::internal::Handle(H5Screate(H5S_NULL), H5Sclose);
    const auto count = shape ? io::elements(*shape) : 0;
    io::internal::Handle attribute(H5Acreate2(file, io::store_attribute, type, space, H5P_DEFAULT, H5P_DEFAULT),
                                   H5Aclose);
    if (count) {
        if (variable) {
            const std::vector<const char*> values(static_cast<std::size_t>(count), value.c_str());
            io::internal::check(H5Awrite(attribute, type, values.data()), "Cannot write variable-length test strings");
        } else {
            std::vector<char> values(static_cast<std::size_t>(count) * width, '\0');
            for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i)
                std::memcpy(values.data() + i * width, value.data(), std::min(value.size(), width));
            io::internal::check(H5Awrite(attribute, type, values.data()), "Cannot write fixed-length test strings");
        }
    }
}

static void expect_statistics_rejected(const std::vector<std::optional<io::Statistics>>& inputs) {
    expect_invalid_argument([&] { io::combine_statistics(inputs); }, "statistics", "shapes");
}

static bool close_enough(double actual, double expected) {
    return actual == expected || (std::isnan(actual) && std::isnan(expected)) ||
           (std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <= 2e-15 * std::abs(expected));
}

static void check_statistics(const std::vector<double>& values, double mean, double squared_deviation,
                             const std::string& path) {
    for (bool matrix : {false, true}) {
        CAPTURE(matrix);
        const std::size_t width = matrix ? 2 : 1;
        std::vector<double> data;
        data.reserve(values.size() * width);
        for (auto value : values) {
            data.push_back(value);
            if (matrix) data.push_back(-value);
        }
        const auto array = io::Array::from<double>(
            data, matrix ? io::Shape{values.size(), width} : io::Shape{values.size()});
        const auto verify = [&](const std::optional<io::Statistics>& stats) {
            REQUIRE(stats.has_value());
            CHECK(stats->count == values.size());
            const auto shape = matrix ? io::Shape{width} : io::Shape{};
            CHECK(stats->mean.shape == shape);
            CHECK(stats->squared_deviation.shape == shape);
            const auto means = stats->mean.values<double>();
            const auto deviations = stats->squared_deviation.values<double>();
            for (std::size_t column = 0; column < width; ++column) {
                const double expected_mean = column ? -mean : mean;
                const bool mean_matches = squared_deviation == 0
                    ? means[column] == expected_mean
                    : close_enough(means[column], expected_mean);
                CHECK(mean_matches);
                CHECK(close_enough(deviations[column], squared_deviation));
            }
        };
        verify(io::statistics(array.view()));
        io::HDF5Storage::write_primal(path, array.view());
        verify(io::Reader(path).statistics("primal"));
    }
}

static std::string create_rectangular_file(TestDirectory& directory, const std::string& name = "a.h5") {
    const auto values = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {3, 2});
    const auto path = directory(name);
    io::HDF5Storage::write_primal_tangent(path, values.view(), values.view());
    return path;
}

TEST_CASE_FIXTURE(Hdf5Fixture, "dataspace/scalar-and-empty") {
    TestDirectory directory;
    const auto path = directory("dataspace.h5");
    const auto scalar = io::Array::from<double>({3.5}, {});
    const io::Array empty(io::dtype<double>(), {0, 2});
    {
        auto file = io::internal::open_file(path, true);
        io::internal::create_dsbo_dataset(file, "scalar", scalar.view());
        io::internal::create_dsbo_dataset(file, "empty", empty.view());
    }
    io::Reader reader(path);
    CHECK(reader.shape("scalar").empty());
    CHECK(reader.read("scalar").values<double>() == scalar.values<double>());
    CHECK(reader.read("scalar", {0, 1, 1, true}).values<double>() == scalar.values<double>());
    CHECK(reader.read_indices("scalar", {0}).values<double>() == scalar.values<double>());
    CHECK(reader.shape("empty") == empty.shape);
    CHECK(reader.read("empty").storage.empty());
}

TEST_CASE_FIXTURE(Hdf5Fixture, "dataspace/null-rejected") {
    TestDirectory directory;
    const auto path = directory("null.h5");
    {
        auto file = io::internal::open_file(path, true);
        create_null_dataset(file, "null");
        io::internal::Handle space(H5Screate(H5S_NULL), H5Sclose);
        io::internal::Handle attribute(H5Acreate2(file, "null_attribute", H5T_NATIVE_DOUBLE, space,
                                             H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    }
    io::Reader reader(path);
    expect_null_rejected([&] { reader.shape("null"); });
    expect_null_rejected([&] { reader.read("null"); });
    expect_null_rejected([&] { reader.read("null", {0, 1, 1, true}); });
    expect_null_rejected([&] { reader.read("null", {0, 1, 1, false}); });
    expect_null_rejected([&] { reader.read("null", {0, 1, 0, false}); });
    expect_null_rejected([&] { reader.read_indices("null", {0}); });
    expect_null_rejected([&] { reader.read_indices("null", {}); });
    auto file = io::internal::open_file(path);
    expect_null_rejected([&] {
        io::internal::read_array_attribute(file, "null_attribute", io::dtype<double>());
    });
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/null-primal-rejected") {
    TestDirectory directory;
    const auto primal_path = directory("primal.h5");
    const auto gradient_path = directory("gradient.h5");
    write_scalar_pair(primal_path, gradient_path);
    replace_with_null_dataset(primal_path, "primal");
    expect_null_rejected([&] { io::HDF5Storage::gradient_file_contributions(primal_path, gradient_path); });
    expect_null_rejected([&] { io::HDF5Storage::load_backward_gradient(primal_path, gradient_path); });
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/null-gradient-rejected") {
    TestDirectory directory;
    const auto primal_path = directory("primal.h5");
    const auto gradient_path = directory("gradient.h5");
    write_scalar_pair(primal_path, gradient_path);
    replace_with_null_dataset(gradient_path, "gradient");
    expect_null_rejected([&] { io::HDF5Storage::load_gradient(gradient_path); });
    expect_null_rejected([&] { io::HDF5Storage::gradient_file_contributions(primal_path, gradient_path); });
    expect_null_rejected([&] { io::HDF5Storage::load_backward_gradient(primal_path, gradient_path); });
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/null-primal-and-gradient-rejected") {
    TestDirectory directory;
    const auto primal_path = directory("primal.h5");
    const auto gradient_path = directory("gradient.h5");
    write_scalar_pair(primal_path, gradient_path);
    replace_with_null_dataset(primal_path, "primal");
    replace_with_null_dataset(gradient_path, "gradient");
    expect_null_rejected([&] { io::HDF5Storage::load_gradient(gradient_path); });
    expect_null_rejected([&] { io::HDF5Storage::gradient_file_contributions(primal_path, gradient_path); });
    expect_null_rejected([&] { io::HDF5Storage::load_backward_gradient(primal_path, gradient_path); });
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reader/lists-datasets-inside-file-without-duplicates-or-endless-loops") {
    TestDirectory directory;
    const auto path = directory("discovery.h5");
    create_hdf5_files_with_links_and_cycles(path);
    io::Reader reader(path);
    CHECK(reader.keys() == std::vector<std::string>({"a_group/a_values", "a_primal"}));
    CHECK(reader.keys("nested") == std::vector<std::string>({"nested/a_values"}));
    CHECK(reader.keys("/nested") == std::vector<std::string>({"/nested/a_values"}));
    CHECK(reader.keys("empty").empty());
    CHECK(reader.read("a_group/a_values").storage == io::Array::from<double>({1, 2, 3}, {3}).storage);
    auto file = io::internal::open_file(path);
    CHECK(io::internal::dataset_paths(file) == std::vector<std::string>({"a_group/a_values", "a_primal"}));
    CHECK(io::internal::dataset_paths(file, "/nested") == std::vector<std::string>({"/nested/a_values"}));
    CHECK(io::internal::dataset_paths(file, "empty").empty());
}

TEST_CASE_FIXTURE(Hdf5Fixture, "discovery/named-reads-resolve-soft-links") {
    TestDirectory directory;
    const auto path = directory("discovery.h5");
    create_hdf5_files_with_links_and_cycles(path);
    io::Reader reader(path);
    CHECK(reader.read("soft_dataset").storage == io::Array::from<double>({1, 2, 3}, {3}).storage);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "discovery/named-reads-resolve-external-links") {
    TestDirectory directory;
    const auto path = directory("discovery.h5");
    create_hdf5_files_with_links_and_cycles(path);
    io::Reader reader(path);
    CHECK(reader.read("external_dataset").storage == io::Array::from<double>({1, 2, 3}, {3}).storage);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "attributes/missing-optional") {
    TestDirectory directory;
    auto file = io::internal::open_file(directory("attributes.h5"), true);
    CHECK(!io::internal::string_attribute(file, io::store_attribute));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "attributes/scalar-and-singleton") {
    TestDirectory directory;
    auto file = io::internal::open_file(directory("attributes.h5"), true);
    for (bool variable : {false, true}) {
        CAPTURE(variable);
        for (const auto& shape : {io::Shape{}, io::Shape{1}, io::Shape{1, 1}}) {
            write_string_attribute(file, variable, shape, "flat");
            const auto value = io::internal::string_attribute(file, io::store_attribute);
            REQUIRE(value.has_value());
            CHECK(*value == "flat");
        }
    }
}

TEST_CASE_FIXTURE(Hdf5Fixture, "attributes/non-singleton-rejected") {
    TestDirectory directory;
    auto file = io::internal::open_file(directory("attributes.h5"), true);
    for (bool variable : {false, true}) {
        CAPTURE(variable);
        const std::vector<std::optional<io::Shape>> shapes{
            io::Shape{0}, io::Shape{2}, io::Shape{100}, io::Shape{2, 2}, std::nullopt};
        for (const auto& shape : shapes) {
            write_string_attribute(file, variable, shape, "flat");
            expect_invalid_argument(
                [&] { io::internal::string_attribute(file, io::store_attribute); },
                io::store_attribute, "exactly one element");
        }
    }
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/combine-rejects-incompatible-shapes") {
    const io::Statistics valid{2, io::Array::from<double>({2, 3}, {2}),
                               io::Array::from<double>({2, 2}, {2})};
    for (const auto& shape : {io::Shape{}, io::Shape{0}, io::Shape{1}, io::Shape{3}, io::Shape{1, 2}}) {
        CAPTURE(shape.size());
        for (bool malformed_mean : {false, true}) {
            auto invalid = valid;
            (malformed_mean ? invalid.mean : invalid.squared_deviation) = io::Array(io::dtype<double>(), shape);
            expect_statistics_rejected({invalid});
            expect_statistics_rejected({invalid, valid});
            expect_statistics_rejected({valid, invalid});
        }
    }
    const auto scalar = io::Array::from<double>({2}, {});
    const auto singleton = io::Array::from<double>({2}, {1});
    expect_statistics_rejected({io::Statistics{2, scalar, singleton}});
    expect_statistics_rejected({io::Statistics{2, scalar, scalar}, io::Statistics{2, singleton, singleton}});
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/combine-preserves-scalar-vector-and-empty-shapes") {
    for (const auto& shape : {io::Shape{}, io::Shape{2}, io::Shape{0}}) {
        CAPTURE(shape.size());
        const auto array = [&](double value) {
            return io::Array::from<double>(std::vector<double>(io::elements(shape), value), shape);
        };
        const io::Statistics first{2, array(2), array(2)};
        const io::Statistics second{1, array(8), array(0)};
        const auto combined = io::combine_statistics({first, second});
        REQUIRE(combined.has_value());
        CHECK(combined->count == 3);
        CHECK(combined->mean.shape == shape);
        CHECK(combined->squared_deviation.shape == shape);
        CHECK(combined->mean.values<double>() == array(4).values<double>());
        CHECK(combined->squared_deviation.values<double>() == array(26).values<double>());
        const auto single = io::combine_statistics({first});
        REQUIRE(single.has_value());
        CHECK(single->count == first.count);
        CHECK(single->mean.storage == first.mean.storage);
        CHECK(single->squared_deviation.storage == first.squared_deviation.storage);
    }
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/combine-returns-nullopt-for-empty-or-missing-inputs") {
    const io::Statistics valid{2, io::Array::from<double>({2, 3}, {2}),
                               io::Array::from<double>({2, 2}, {2})};
    CHECK(!io::combine_statistics({}));
    CHECK(!io::combine_statistics({valid, std::nullopt}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/accuracy/cancellation") {
    TestDirectory directory;
    check_statistics({1e16, 1, -1e16}, 1.0 / 3, 2e32, directory("statistics.h5"));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/accuracy/long-cancellation") {
    TestDirectory directory;
    std::vector<double> cancellation(100002, 1);
    cancellation.front() = 1e16;
    cancellation.back() = -1e16;
    check_statistics(cancellation, 100000.0 / 100002, 2e32, directory("statistics.h5"));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/accuracy/constant-large-offset-has-zero-deviation") {
    TestDirectory directory;
    const double offset = 1e12 + 0.1;
    check_statistics(std::vector<double>(9, offset), offset, 0, directory("statistics-9.h5"));
    check_statistics(std::vector<double>(1000000, offset), offset, 0, directory("statistics-1000000.h5"));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/accuracy/ulp-step") {
    TestDirectory directory;
    const double offset = 1e12 + 0.1;
    const double step = std::nextafter(offset, std::numeric_limits<double>::infinity()) - offset;
    check_statistics({offset, offset, offset + step}, offset, step * step * (2.0 / 3),
                     directory("statistics.h5"));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/accuracy/small-variations-around-large-offset") {
    TestDirectory directory;
    const double offset = 1e12 + 0.1;
    std::vector<double> varied(100002);
    for (std::size_t i = 0; i < varied.size(); ++i)
        varied[i] = offset + (static_cast<int>(i % 3) - 1) * 0.25;
    check_statistics(varied, offset, 33334.0 / 8, directory("statistics.h5"));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/accuracy/squared-deviation-retains-small-contributions") {
    TestDirectory directory;
    std::vector<double> squared_sum(100002);
    for (std::size_t i = 1; i + 1 < squared_sum.size(); ++i)
        squared_sum[i] = i % 2 ? 1 : -1;
    squared_sum.front() = 1e8;
    squared_sum.back() = -1e8;
    check_statistics(squared_sum, 0, 2e16 + 100000, directory("statistics.h5"));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/accuracy/non-finite") {
    TestDirectory directory;
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check_statistics({nan, 1}, nan, nan, directory("nan.h5"));
    check_statistics({infinity, 1}, infinity, nan, directory("infinity.h5"));
    check_statistics({-infinity, 1}, -infinity, nan, directory("negative-infinity.h5"));
    check_statistics({infinity, infinity}, infinity, nan, directory("infinity-infinity.h5"));
    check_statistics({infinity, -infinity}, nan, nan, directory("mixed-infinity.h5"));
    check_statistics({1e200, -1e200}, 0, infinity, directory("overflow.h5"));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "environment/bundled-deflate-filter") {
    unsigned filter_info = 0;
    CHECK(H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0);
    REQUIRE(H5Zget_filter_info(H5Z_FILTER_DEFLATE, &filter_info) >= 0);
    CHECK((filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED) != 0);
    CHECK((filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED) != 0);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "headers/double-dtype-works-across-cpp-files") {
    CHECK(double_dtype_from_separate_cpp_file() == io::dtype<double>());
}

TEST_CASE_FIXTURE(Hdf5Fixture, "convert/boolean-truth") {
    const auto truth = io::Array::from<std::int64_t>({-1, 0, 256}, {3});
    CHECK(io::convert(truth.view(), io::dtype<bool>()).values<bool>() ==
          std::vector<bool>({true, false, true}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/extended-float-byte-order") {
    TestDirectory directory;
    const auto path = directory("extended.h5");
    const auto extended = io::Array::from<long double>({1.25L, -2.5L}, {2});
    auto swapped_type = extended.type;
    swapped_type.little_endian = !swapped_type.little_endian;
    const auto swapped = io::convert(extended.view(), swapped_type);
    io::HDF5Storage::write_primal(path, swapped.view());
    const auto loaded = io::HDF5Storage::open_primal(path);
    CHECK(loaded.type == swapped_type);
    CHECK(io::convert(loaded.view(), extended.type).values<long double>() == extended.values<long double>());
}

TEST_CASE_FIXTURE(Hdf5Fixture, "headers/conversion-and-statistics-work-across-cpp-files") {
    const auto values = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {3, 2});
    const auto stats = double_statistics_from_separate_cpp_file(values.view());
    CHECK(stats.count == 3);
    CHECK(stats.mean.values<double>() == std::vector<double>({3, 4}));
    CHECK(stats.squared_deviation.values<double>() == std::vector<double>({8, 8}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/primal-tangent-round-trip") {
    TestDirectory directory;
    const auto path = directory("a.h5");
    const auto values = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {3, 2});
    io::HDF5Storage::write_primal_tangent(path, values.view(), values.view());
    const auto [primal, tangent] = io::HDF5Storage::open_primal_tangent(path);
    CHECK(primal.storage == values.storage);
    REQUIRE(tangent.has_value());
    CHECK(tangent->storage == values.storage);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/reverse-hyperslab") {
    TestDirectory directory;
    io::Reader reader(create_rectangular_file(directory));
    CHECK(reader.read("primal", {2, -1, 3, false}).values<double>() ==
          std::vector<double>({5, 6, 3, 4, 1, 2}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/indexed-rows-preserve-order-and-duplicates") {
    TestDirectory directory;
    io::Reader reader(create_rectangular_file(directory));
    CHECK(reader.read_indices("primal", {2, 0, 2, -2}).values<double>() ==
          std::vector<double>({5, 6, 1, 2, 5, 6, 3, 4}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/persisted-statistics") {
    TestDirectory directory;
    io::Reader reader(create_rectangular_file(directory));
    const auto stats = reader.statistics("primal");
    REQUIRE(stats.has_value());
    CHECK(stats->count == 3);
    CHECK(stats->mean.values<double>() == std::vector<double>({3, 4}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/reader-close-is-idempotent-and-rejects-access") {
    TestDirectory directory;
    io::Reader reader(create_rectangular_file(directory));
    reader.close();
    reader.close();
    CHECK_THROWS_WITH_AS(reader.read("primal"), doctest::Contains("closed"), std::runtime_error);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reduce/vds-concatenation") {
    TestDirectory directory;
    const auto source = create_rectangular_file(directory);
    const auto target = directory("vds.h5");
    io::HDF5Storage::reduce({source, source}, target);
    CHECK(io::HDF5Storage::open_primal(target).shape == io::Shape({6, 2}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reduce/vds-statistics") {
    TestDirectory directory;
    const auto source = create_rectangular_file(directory);
    const auto target = directory("vds.h5");
    io::HDF5Storage::reduce({source, source}, target);
    const auto stats = io::Reader(target).statistics("primal");
    REQUIRE(stats.has_value());
    CHECK(stats->count == 6);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/vds-contributions") {
    TestDirectory directory;
    const auto source = create_rectangular_file(directory);
    const auto target = directory("vds.h5");
    const auto adjoint = directory("adjoint.h5");
    io::HDF5Storage::reduce({source, source}, target);
    const auto gradient = io::Array::from<double>(std::vector<double>(12, 2), {6, 2});
    io::HDF5Storage::write_gradient(adjoint, gradient.view());
    const auto contributions = io::HDF5Storage::gradient_file_contributions(target, adjoint);
    REQUIRE(contributions.size() == 2);
    CHECK(std::get<io::Array>(contributions[0].source).shape == io::Shape({3, 2}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "columns/utf8-order") {
    TestDirectory directory;
    const auto path = directory("named.h5");
    const auto named = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {2, 3});
    io::HDF5Storage::write_primal(path, named.view(), std::vector<std::string>{"β", "first"});
    CHECK(io::HDF5Storage::columns(path) == std::vector<std::string>({"β", "first"}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/missing-columns-zero-filled") {
    TestDirectory directory;
    const auto primal_path = directory("named.h5");
    const auto gradient_path = directory("named_gradient.h5");
    const auto named = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto selected = io::Array::from<float>({7, 8, 9}, {1, 3});
    io::HDF5Storage::write_primal(primal_path, named.view(), std::vector<std::string>{"β", "first"});
    io::HDF5Storage::write_gradient(gradient_path, selected.view(), std::vector<std::string>{"first"});
    const auto backward = io::HDF5Storage::load_backward_gradient(primal_path, gradient_path);
    CHECK(backward.values<float>() == std::vector<float>({0, 0, 0, 7, 8, 9}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/physical-source-contribution-is-a-file-path") {
    TestDirectory directory;
    const auto primal_path = directory("named.h5");
    const auto gradient_path = directory("named_gradient.h5");
    const auto named = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto selected = io::Array::from<float>({7, 8, 9}, {1, 3});
    io::HDF5Storage::write_primal(primal_path, named.view(), std::vector<std::string>{"β", "first"});
    io::HDF5Storage::write_gradient(gradient_path, selected.view(), std::vector<std::string>{"first"});
    const auto contributions = io::HDF5Storage::gradient_file_contributions(primal_path, gradient_path);
    REQUIRE(!contributions.empty());
    CHECK(std::holds_alternative<std::string>(contributions[0].source));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "row-store/keys-and-promotion") {
    TestDirectory directory;
    const auto path = directory("rows.h5");
    const auto feature = io::Array::from<std::int64_t>({1, 2, 3}, {3});
    const auto other = io::Array::from<double>({0.5, 1.5, 2.5}, {3});
    io::HDF5Storage::write_row_store(path, {{"feature/key", feature.view()}, {"energy", other.view()}});
    io::Reader rows(path);
    CHECK(rows.row_keys() == std::vector<std::string>({"feature/key", "energy"}));
    CHECK(rows.read(io::row_store_dataset).values<double>() ==
          std::vector<double>({1, 0.5, 2, 1.5, 3, 2.5}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reduce/scalar-vds") {
    TestDirectory directory;
    const auto path = directory("scalar.h5");
    const auto target = directory("scalar-vds.h5");
    const auto scalar = io::Array::from<double>({3.5}, {});
    io::HDF5Storage::write_primal(path, scalar.view());
    io::HDF5Storage::reduce({path, path}, target);
    CHECK(io::HDF5Storage::open_primal(target).values<double>() == std::vector<double>({3.5, 3.5}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reduce/empty-vds") {
    TestDirectory directory;
    const auto path = directory("empty.h5");
    const auto target = directory("empty-vds.h5");
    const io::Array empty(io::dtype<float>(), {0, 2});
    io::HDF5Storage::write_primal(path, empty.view());
    io::HDF5Storage::reduce({path, path}, target);
    CHECK(io::HDF5Storage::open_primal(target).shape == io::Shape({0, 2}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/integer-wrap") {
    TestDirectory directory;
    const auto maximum_path = directory("maximum.h5");
    const auto one_path = directory("one.h5");
    const auto sum_path = directory("sum.h5");
    const auto maximum = io::Array::from<std::uint64_t>({std::numeric_limits<std::uint64_t>::max()}, {1});
    const auto one = io::Array::from<std::uint64_t>({1}, {1});
    io::HDF5Storage::write_gradient(maximum_path, maximum.view());
    io::HDF5Storage::write_gradient(one_path, one.view());
    io::HDF5Storage::aggregate_gradients({maximum_path, one_path}, sum_path);
    const auto sum = io::HDF5Storage::load_gradient(sum_path).values<std::uint64_t>();
    REQUIRE(sum.size() == 1);
    CHECK(sum[0] == 0);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "concurrency/parallel-file-io") {
    TestDirectory directory;
    const auto values = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {3, 2});
    std::vector<std::future<void>> workers;
    for (int i = 0; i < 4; ++i) {
        workers.push_back(std::async(std::launch::async, [&, i] {
            const auto path = directory("thread-" + std::to_string(i) + ".h5");
            for (int j = 0; j < 8; ++j) {
                io::HDF5Storage::write_primal(path, values.view());
                if (io::HDF5Storage::open_primal(path).storage != values.storage)
                    throw std::runtime_error("Concurrent HDF5 calls returned different values");
            }
        }));
    }
    for (auto& worker : workers) CHECK_NOTHROW(worker.get());
}
