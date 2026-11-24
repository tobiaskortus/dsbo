#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <dsbo/hdf5.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <list>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "test_support.hpp"

namespace io = dsbo::io;
namespace fs = std::filesystem;
using dsbo::test::TestDirectory;
using dsbo::test::same_bytes;
using dsbo::test::same_values;

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
        io::internal::check(H5Lcreate_soft("/external_dataset", file, "soft_external", H5P_DEFAULT, H5P_DEFAULT),
                             "Cannot create test indirect external link");
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
            const auto* means = stats->mean.data<double>();
            const auto* deviations = stats->squared_deviation.data<double>();
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

template<class Buffer, class = void> struct HasTypedWrite : std::false_type {};
template<class Buffer>
struct HasTypedWrite<Buffer, std::void_t<decltype(io::write_primal(
    std::declval<const std::string&>(), std::declval<const Buffer&>()))>> : std::true_type {};

struct DataOnlyBuffer { const double* data() const; };

static_assert(HasTypedWrite<std::array<double, 4>>::value);
static_assert(HasTypedWrite<double[4]>::value);
static_assert(!HasTypedWrite<std::vector<bool>>::value);
static_assert(!HasTypedWrite<std::vector<long double>>::value);
static_assert(!HasTypedWrite<std::list<double>>::value);
static_assert(!HasTypedWrite<DataOnlyBuffer>::value);
static_assert(!std::is_copy_constructible_v<io::internal::ArrayStorage>);
static_assert(std::is_nothrow_move_constructible_v<io::Array>);
static_assert(std::is_nothrow_move_assignable_v<io::Array>);

template<class T>
static void check_typed_array_data(io::internal::Initialization initialization) {
    io::Array array(io::dtype<T>(), {2, 3}, initialization);
    CHECK(array.nbytes() == 6 * sizeof(T));
    auto* data = array.template data<T>();
    if (initialization == io::internal::Initialization::zero)
        for (std::size_t i = 0; i < array.size(); ++i) CHECK(data[i] == T(0));
    for (std::size_t i = 0; i < array.size(); ++i) data[i] = T(i + 1);
    const auto& constant = array;
    static_assert(std::is_same_v<decltype(array.template data<T>()), T*>);
    static_assert(std::is_same_v<decltype(constant.template data<T>()), const T*>);
    static_assert(std::is_same_v<decltype(array.data()), unsigned char*>);
    static_assert(std::is_same_v<decltype(constant.data()), const unsigned char*>);
    CHECK(static_cast<const void*>(data) == static_cast<const void*>(array.data()));
    CHECK(reinterpret_cast<std::uintptr_t>(data) % alignof(T) == 0);
    CHECK(constant.template data<T>() == data);
    CHECK(data[5] == T(6));
    data[4] = T(9);
    CHECK(constant.template data<T>()[4] == T(9));

    auto copy = array;
    CHECK(same_bytes(copy, array));
    CHECK(copy.template data<T>() != data);
    copy.template data<T>()[0] = T(7);
    CHECK(data[0] == T(1));
    const auto* copied_data = copy.template data<T>();
    auto moved = std::move(copy);
    CHECK(moved.template data<T>() == copied_data);
    CHECK(moved.template data<T>()[0] == T(7));
    CHECK(copy.nbytes() == 0);
    CHECK(copy.data() == nullptr);
    array = moved;
    CHECK(same_bytes(array, moved));
    CHECK(array.template data<T>() != moved.template data<T>());
    array = std::move(moved);
    CHECK(array.template data<T>() == copied_data);
    CHECK(moved.nbytes() == 0);
    CHECK(moved.data() == nullptr);

    auto& self = array;
    const auto expected = array;
    array = self;
    CHECK(same_bytes(array, expected));
    const auto* assigned_data = array.template data<T>();
    array = std::move(self);
    CHECK(array.template data<T>() == assigned_data);
    CHECK(same_bytes(array, expected));

    io::Array empty(io::dtype<T>(), {0, 3}, initialization);
    CHECK_NOTHROW(empty.template data<T>());
    CHECK(empty.shape == io::Shape{0, 3});
    const auto empty_copy = empty;
    CHECK(empty_copy.template data<T>() == nullptr);
    CHECK(empty_copy.nbytes() == 0);
    io::Array scalar(io::dtype<T>(), {}, initialization);
    scalar.template data<T>()[0] = T(3);
    CHECK(std::as_const(scalar).template data<T>()[0] == T(3));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "namespace-io/read-write-and-typed-access") {
    TestDirectory directory;
    const auto primal_path = directory("primal.h5");
    const auto gradient_path = directory("gradient.h5");
    const std::vector<double> values{1, 2, 3, 4};
    const std::vector<double> derivatives{5, 6, 7, 8};

    io::write_primal(primal_path, values, {2, 2});
    const auto primal = io::open_primal(primal_path);
    CHECK(primal.shape == io::Shape{2, 2});
    CHECK(primal.data<double>()[3] == 4);
    CHECK(io::columns(primal_path).empty());
    CHECK(io::shape_from_path(primal_path) == primal.shape);
    CHECK_FALSE(io::open_primal_tangent(primal_path).second.has_value());

    io::write_primal_tangent(primal_path, values, derivatives, {2, 2});
    const auto [dual_primal, tangent] = io::open_primal_tangent(primal_path);
    CHECK(dual_primal.data<double>()[0] == 1);
    REQUIRE(tangent.has_value());
    CHECK(tangent->shape == primal.shape);
    CHECK(tangent->data<double>()[3] == 8);

    io::write_gradient(gradient_path, derivatives, {2, 2});
    CHECK(io::load_gradient(gradient_path).data<double>()[0] == 5);
    const auto adjoint = io::load_backward_gradient(primal_path, gradient_path);
    CHECK(adjoint.shape == primal.shape);
    CHECK(adjoint.data<double>()[3] == 8);
    CHECK_THROWS_AS(primal.data<float>(), io::TypeError);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "namespace-io/keyed-read-selection-and-validation") {
    TestDirectory directory;
    const auto primal_path = directory("primal.h5");
    const auto gradient_path = directory("gradient.h5");
    const std::vector<double> absorber{2, 3};
    const std::vector<float> gap{4, 5};
    const auto selected = std::vector<std::string>{"gap", "absorber"};
    const io::NamedArrays fields{io::field("absorber", absorber), io::field("gap", gap)};

    io::write_primal(primal_path, fields);
    CHECK(io::columns(primal_path) == std::vector<std::string>{"absorber", "gap"});
    const auto shapes = io::dict_shape_from_path(primal_path);
    REQUIRE(shapes.size() == 2);
    CHECK(shapes[0].first == "absorber");
    CHECK(shapes[0].second == io::Shape{2});
    CHECK(shapes[1].first == "gap");
    CHECK(shapes[1].second == io::Shape{2});
    const auto primals = io::open_dict(primal_path);
    CHECK(io::at(primals, "absorber").data<double>()[1] == 3);
    CHECK(io::at(primals, "gap").data<float>()[1] == 5);
    CHECK(io::open_dict(primal_path, selected)[0].first == "gap");
    CHECK_FALSE(io::open_dict_tangent(primal_path).second.has_value());
    CHECK_THROWS_AS(io::open_dict(primal_path, std::vector<std::string>{"missing"}), io::KeyError);
    CHECK_THROWS_AS(io::open_primal(primal_path), std::invalid_argument);

    io::write_primal_tangent(primal_path, fields, {io::field("gap", gap)});
    const auto [dual_primals, tangents] = io::open_dict_tangent(primal_path, selected);
    CHECK(dual_primals[0].first == "gap");
    REQUIRE(tangents.has_value());
    REQUIRE(tangents->size() == 1);
    CHECK(io::at(*tangents, "gap").data<float>()[0] == 4);

    io::write_gradient(gradient_path, fields);
    CHECK(io::load_dict_gradient(gradient_path).size() == 2);
    CHECK(io::load_dict_gradient(gradient_path, selected)[0].first == "gap");
    const auto adjoints = io::load_backward_dict_gradient(primal_path, gradient_path);
    CHECK(io::at(adjoints, "absorber").data<double>()[0] == 2);
    CHECK(io::at(adjoints, "gap").data<float>()[1] == 5);
    io::write_gradient(gradient_path, {io::field("gap", gap, {1, 2})});
    CHECK_THROWS_AS(io::load_backward_dict_gradient(primal_path, gradient_path), std::invalid_argument);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "array-data/typed-storage-and-ownership") {
    for (auto initialization : {io::internal::Initialization::zero, io::internal::Initialization::uninitialized}) {
        CAPTURE(initialization == io::internal::Initialization::zero);
        check_typed_array_data<float>(initialization);
        check_typed_array_data<double>(initialization);
        check_typed_array_data<std::int8_t>(initialization);
        check_typed_array_data<std::uint8_t>(initialization);
        check_typed_array_data<std::int16_t>(initialization);
        check_typed_array_data<std::uint16_t>(initialization);
        check_typed_array_data<std::int32_t>(initialization);
        check_typed_array_data<std::uint32_t>(initialization);
        check_typed_array_data<std::int64_t>(initialization);
        check_typed_array_data<std::uint64_t>(initialization);
    }

    for (const auto type : {io::dtype<bool>(), io::DType{io::Kind::floating, 2}}) {
        io::Array zeroed(type, {3});
        const auto* bytes = zeroed.data();
        for (std::size_t i = 0; i < zeroed.nbytes(); ++i) CHECK(bytes[i] == 0);
        io::Array overwritten(type, {3}, io::internal::Initialization::uninitialized);
        std::memset(overwritten.data(), 0x42, overwritten.nbytes());
        auto copy = overwritten;
        CHECK(same_bytes(copy, overwritten));
        CHECK(copy.data() != overwritten.data());
        auto assigned = io::Array::from<double>({1}, {});
        assigned = overwritten;
        CHECK(same_bytes(assigned, overwritten));
        CHECK(assigned.data() != overwritten.data());
    }

    auto array = io::Array::from<double>({1, 2}, {2});
    CHECK_THROWS_AS(array.data<float>(), io::TypeError);
    CHECK_THROWS_AS(std::as_const(array).data<std::int64_t>(), io::TypeError);
    array.type = io::dtype<std::int64_t>();
    CHECK_THROWS_AS(array.data<std::int64_t>(), io::TypeError);
    const auto retagged_copy = array;
    CHECK(same_bytes(retagged_copy, array));
    CHECK_THROWS_AS(retagged_copy.data<std::int64_t>(), io::TypeError);
    array.type = io::dtype<double>();
    array.shape = {3};
    CHECK_THROWS(array.data<double>());
    const auto reshaped_copy = array;
    CHECK(reshaped_copy.nbytes() == 2 * sizeof(double));
    CHECK(same_bytes(reshaped_copy, array));
    CHECK_THROWS_AS(reshaped_copy.data<double>(), io::TypeError);

    const auto integers = io::Array::from<std::int64_t>({1, 2}, {2});
    using OtherInteger = std::conditional_t<std::is_same_v<std::int64_t, long>, long long, long>;
    if constexpr (sizeof(OtherInteger) == sizeof(std::int64_t))
        CHECK_THROWS_AS(integers.data<OtherInteger>(), io::TypeError);

    const auto boolean = io::Array::from<bool>({true, false}, {2});
    CHECK_THROWS_AS(boolean.data<bool>(), io::TypeError);
    CHECK(same_values(boolean, std::vector<bool>{true, false}));
    const auto boolean_bytes = io::convert(boolean.view(), io::dtype<std::uint8_t>());
    CHECK(boolean_bytes.data<std::uint8_t>()[0] == 1);
    CHECK(boolean_bytes.data<std::uint8_t>()[1] == 0);

    auto special = io::Array::from<double>({0, -0.0}, {2});
    const std::uint64_t nan_payload = UINT64_C(0x7ff8000000000042);
    std::memcpy(special.data(), &nan_payload, sizeof(nan_payload));
    const auto special_copy = special;
    CHECK(same_bytes(special_copy, special));
    CHECK(std::signbit(special_copy.data<double>()[1]));

    auto assigned = io::Array::from<std::uint8_t>({1}, {});
    assigned = special;
    CHECK(same_bytes(assigned, special));
    CHECK(assigned.data<double>() != special.data<double>());
    const auto* assigned_data = assigned.data<double>();
    auto moved = io::Array::from<float>({1}, {});
    moved = std::move(assigned);
    CHECK(same_bytes(moved, special));
    CHECK(moved.data<double>() == assigned_data);
    CHECK(assigned.nbytes() == 0);
    CHECK(assigned.data() == nullptr);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "array-data/allocation-rejects-byte-overflow") {
    for (const auto type : {io::dtype<double>(), io::DType{io::Kind::floating, 2}}) {
        const auto count = std::numeric_limits<std::size_t>::max() / type.size + 1;
        CHECK_THROWS_AS(io::Array(type, {count}), std::overflow_error);
    }
    CHECK_THROWS_AS(io::Array(io::DType{io::Kind::floating, 0}, {0}), std::overflow_error);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "array-data/nd-hdf5-read-and-write") {
    TestDirectory directory;
    const std::vector<double> values{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const auto path = directory("nd.h5");
    io::write_primal(path, values, {2, 2, 3});
    auto array = io::HDF5Storage::open_primal(path);
    REQUIRE(array.shape == io::Shape{2, 2, 3});
    auto* data = array.data<double>();
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            for (std::size_t k = 0; k < 3; ++k) {
                const auto offset = (i * array.shape[1] + j) * array.shape[2] + k;
                CHECK(data[offset] == values[offset]);
            }
    data[(1 * 2 + 0) * 3 + 2] = 42;
    io::write_primal(directory("modified.h5"), array);
    const auto modified = io::HDF5Storage::open_primal(directory("modified.h5"));
    CHECK(modified.shape == array.shape);
    CHECK(modified.data<double>()[8] == 42);
    CHECK(modified.data<double>()[0] == 0);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "typed-writes/borrow-containers-and-preserve-dtypes") {
    TestDirectory directory;
    const std::vector<double> values{1.0000000000000002, -2.25, 3.5, 4.0};
    const int rows = 2;
    const std::size_t columns = 2;
    const auto view = io::view_of(values, {rows, columns});
    CHECK(view.data == values.data());
    CHECK(view.type == io::dtype<double>());
    CHECK(view.shape == io::Shape{2, 2});
    io::write_primal(directory("double.h5"), values, {rows, columns});
    const auto result = io::HDF5Storage::open_primal(directory("double.h5"));
    CHECK(result.type == io::dtype<double>());
    CHECK(result.shape == view.shape);
    CHECK(same_values(result, values));

    const std::array<float, 3> floats{{1.0f, 2.5f, -3.0f}};
    io::write_primal(directory("float.h5"), floats);
    CHECK(io::HDF5Storage::open_primal(directory("float.h5")).type == io::dtype<float>());
    const std::uint64_t integers[]{0, (std::uint64_t(1) << 63) + 1, UINT64_MAX};
    io::write_primal(directory("integer.h5"), integers);
    const auto integer_result = io::HDF5Storage::open_primal(directory("integer.h5"));
    CHECK(integer_result.type == io::dtype<std::uint64_t>());
    CHECK(same_values(integer_result, std::vector<std::uint64_t>(integers, integers + 3)));

    const bool flags[]{true, false, true};
    io::write_primal(directory("bool.h5"), flags);
    const auto bool_result = io::HDF5Storage::open_primal(directory("bool.h5"));
    CHECK(bool_result.type == io::dtype<bool>());
    CHECK(same_values(bool_result, std::vector<bool>{true, false, true}));

    io::write_primal(directory("pointer.h5"), values.data() + 1, 2, {1, 2});
    CHECK(same_values(io::HDF5Storage::open_primal(directory("pointer.h5")),
                     std::vector<double>{-2.25, 3.5}));
    io::write_primal(directory("inferred-pointer.h5"), values.data(), values.size());
    CHECK(io::HDF5Storage::shape_from_path(directory("inferred-pointer.h5")) == io::Shape{4});

    const io::Shape dynamic_shape{2, 2};
    CHECK(io::view_of(values, dynamic_shape).shape == dynamic_shape);
    CHECK(io::field("matrix", values, dynamic_shape).second.shape == dynamic_shape);
    io::write_primal(directory("owned.h5"), result);
    CHECK(same_values(io::HDF5Storage::open_primal(directory("owned.h5")), values));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "typed-writes/scalars-empty-and-invalid-shapes") {
    TestDirectory directory;
    const std::array<double, 1> scalar{{3.5}};
    io::write_primal(directory("scalar.h5"), scalar, {});
    CHECK(io::HDF5Storage::shape_from_path(directory("scalar.h5")).empty());
    io::write_gradient(directory("scalar-gradient.h5"), scalar, {});
    CHECK(io::HDF5Storage::load_gradient(directory("scalar-gradient.h5")).shape.empty());

    const std::vector<double> empty;
    io::write_primal(directory("empty.h5"), empty);
    CHECK(io::HDF5Storage::shape_from_path(directory("empty.h5")) == io::Shape{0});
    io::write_primal(directory("empty-matrix.h5"), empty, {0, 3});
    CHECK(io::HDF5Storage::shape_from_path(directory("empty-matrix.h5")) == io::Shape{0, 3});
    io::write_primal(directory("empty-pointer.h5"), static_cast<const float*>(nullptr), 0);
    CHECK(io::HDF5Storage::open_primal(directory("empty-pointer.h5")).nbytes() == 0);

    const auto invalid_path = directory("invalid.h5");
    CHECK_THROWS_AS(io::write_primal(invalid_path, scalar, {2}), std::invalid_argument);
    CHECK_THROWS_AS(io::write_primal(invalid_path, scalar, {0}), std::invalid_argument);
    const double c_array[]{1, 2};
    CHECK_THROWS_AS(io::view_of(c_array, {3}), std::invalid_argument);
    CHECK_THROWS_AS(io::view_of(c_array, {}), std::invalid_argument);
    CHECK_THROWS_AS(io::write_primal(invalid_path, c_array, {3}), std::invalid_argument);
    CHECK_THROWS_AS(io::write_primal(invalid_path, c_array, {}), std::invalid_argument);
    CHECK_THROWS_AS(io::write_gradient(invalid_path, c_array, {3}), std::invalid_argument);
    CHECK_THROWS_AS(io::write_gradient(invalid_path, c_array, {}), std::invalid_argument);
    CHECK_THROWS_AS(io::field("c-array", c_array, {3}), std::invalid_argument);
    CHECK_THROWS_AS(io::field("c-array", c_array, {}), std::invalid_argument);
    CHECK_THROWS_AS(io::write_primal(invalid_path, scalar, {H5S_UNLIMITED}), std::invalid_argument);
    const io::Shape unlimited_shape{H5S_UNLIMITED};
    CHECK_THROWS_AS(io::write_primal(invalid_path, scalar, unlimited_shape), std::invalid_argument);
    const auto maximum = std::numeric_limits<std::size_t>::max() / 2;
    const io::Shape overflow_shape{maximum, maximum};
    CHECK_THROWS_AS(io::write_primal(invalid_path, scalar, overflow_shape), std::overflow_error);
    CHECK_FALSE(fs::exists(invalid_path));

    const double c_scalar[]{3.5};
    CHECK(io::view_of(c_scalar, {}).shape.empty());
    CHECK(io::field("scalar", c_scalar, {}).second.shape.empty());
    io::write_primal(directory("c-scalar.h5"), c_scalar, {});
    CHECK(io::HDF5Storage::shape_from_path(directory("c-scalar.h5")).empty());
}

TEST_CASE_FIXTURE(Hdf5Fixture, "typed-writes/heterogeneous-fields-tangents-and-gradients") {
    TestDirectory directory;
    const std::vector<double> energy{1.0000000000000002, -2.25};
    const std::array<float, 4> position{{1, 2, 3, 4}};
    const std::int32_t labels[]{1, 2};
    auto energy_view = io::view_of(energy);
    const auto* shape_data = energy_view.shape.data();
    const auto energy_field = io::field("energy", std::move(energy_view));
    CHECK(energy_field.second.data == energy.data());
    CHECK(energy_field.second.shape.data() == shape_data);
    const auto path = directory("fields.h5");
    io::write_primal(path, {
        energy_field,
        io::field("position", position, {2, 2}),
        io::field("label", labels),
    });
    const auto result = io::HDF5Storage::open_dict(path);
    REQUIRE(result.size() == 3);
    CHECK(result[0].first == "energy");
    CHECK(result[1].first == "position");
    CHECK(result[2].first == "label");
    CHECK(io::at(result, "energy").type == io::dtype<double>());
    CHECK(same_values(io::at(result, "energy"), energy));
    CHECK(io::at(result, "position").type == io::dtype<float>());
    CHECK(io::at(result, "position").shape == io::Shape{2, 2});
    CHECK(io::at(result, "label").type == io::dtype<std::int32_t>());

    io::write_primal_tangent(directory("flat-tangent.h5"), energy, energy, {1, 2});
    const auto flat_tangent = io::HDF5Storage::open_primal_tangent(directory("flat-tangent.h5"));
    REQUIRE(flat_tangent.second.has_value());
    CHECK(flat_tangent.first.type == io::dtype<double>());
    CHECK(flat_tangent.second->shape == io::Shape{1, 2});
    io::write_primal_tangent(directory("inferred-tangent.h5"), energy, energy);
    io::write_primal_tangent(directory("pointer-tangent.h5"), energy.data(), energy.data(), energy.size());
    io::write_primal_tangent(directory("pointer-tangent.h5"), energy.data(), energy.data(), energy.size(), {1, 2});
    const auto pointer_tangent = io::HDF5Storage::open_primal_tangent(directory("pointer-tangent.h5"));
    REQUIRE(pointer_tangent.second.has_value());
    CHECK(same_values(*pointer_tangent.second, energy));
    CHECK(pointer_tangent.second->shape == io::Shape{1, 2});

    io::write_primal_tangent(path,
        {io::field("energy", energy), io::field("position", position, {2, 2})},
        {io::field("energy", energy)});
    const auto named_tangent = io::HDF5Storage::open_dict_tangent(path);
    REQUIRE(named_tangent.second.has_value());
    CHECK(same_values(io::at(*named_tangent.second, "energy"), energy));

    const auto gradient_path = directory("gradient.h5");
    io::write_gradient(gradient_path, {io::field("energy", energy)});
    io::write_gradient(gradient_path, {io::field("position", position, {2, 2})}, true);
    const auto gradients = io::HDF5Storage::load_dict_gradient(gradient_path);
    CHECK(io::at(gradients, "energy").type == io::dtype<double>());
    CHECK(io::at(gradients, "position").type == io::dtype<float>());

    io::write_gradient(gradient_path, energy, {1, 2});
    CHECK(io::HDF5Storage::load_gradient(gradient_path).shape == io::Shape{1, 2});
    io::write_gradient(gradient_path, energy);
    io::write_gradient(gradient_path, energy.data(), energy.size());
    io::write_gradient(gradient_path, energy.data(), energy.size(), {1, 2});
    CHECK(same_values(io::HDF5Storage::load_gradient(gradient_path), energy));
    CHECK(io::field("prefix", io::view_of(energy.data(), 1)).second.size() == 1);
    CHECK(io::field("matrix", io::view_of(position.data(), position.size(), {2, 2})).second.data == position.data());
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
    CHECK(reader.read("scalar").data<double>()[0] == scalar.data<double>()[0]);
    CHECK(reader.read("scalar", {0, 1, 1, true}).data<double>()[0] == scalar.data<double>()[0]);
    CHECK(reader.read_indices("scalar", {0}).data<double>()[0] == scalar.data<double>()[0]);
    CHECK(reader.shape("empty") == empty.shape);
    CHECK(reader.read("empty").nbytes() == 0);
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
    CHECK(same_bytes(reader.read("a_group/a_values"), io::Array::from<double>({1, 2, 3}, {3})));
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
    CHECK(same_bytes(reader.read("soft_dataset"), io::Array::from<double>({1, 2, 3}, {3})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "discovery/external-links-are-unsupported") {
    TestDirectory directory;
    const auto path = directory("discovery.h5");
    create_hdf5_files_with_links_and_cycles(path);
    io::Reader reader(path);
    for (const auto* name : {"external_dataset", "external_group/values", "soft_external", "dangling_external"}) {
        CAPTURE(name);
        expect_invalid_argument([&] { reader.read(name); }, "External HDF5 links", "not supported");
        expect_invalid_argument([&] { reader.read(name, {0, 1, 1, false}); }, "External HDF5 links");
        expect_invalid_argument([&] { reader.read_indices(name, {0}); }, "External HDF5 links");
        expect_invalid_argument([&] { reader.shape(name); }, "External HDF5 links");
        expect_invalid_argument([&] { reader.type(name); }, "External HDF5 links");
    }
    expect_invalid_argument([&] { reader.keys("external_group"); }, "External HDF5 links");
    expect_invalid_argument([&] { reader.columns("external_group"); }, "External HDF5 links");
    CHECK(same_values(reader.read_indices("primal", {2, 0}), std::vector<double>{3, 1}));
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
        const std::vector<std::optional<io::Statistics>> batches{first, second};
        const auto combined = io::combine_statistics(batches);
        REQUIRE(combined.has_value());
        CHECK(combined->count == 3);
        CHECK(combined->mean.shape == shape);
        CHECK(combined->squared_deviation.shape == shape);
        CHECK(same_bytes(combined->mean, array(4)));
        CHECK(same_bytes(combined->squared_deviation, array(26)));
        CHECK(same_bytes(batches[0]->mean, first.mean));
        CHECK(same_bytes(batches[0]->squared_deviation, first.squared_deviation));
        CHECK(same_bytes(batches[1]->mean, second.mean));
        CHECK(same_bytes(batches[1]->squared_deviation, second.squared_deviation));
        if (combined->mean.size()) {
            CHECK(combined->mean.data<double>() != batches[0]->mean.data<double>());
            CHECK(combined->squared_deviation.data<double>() != batches[0]->squared_deviation.data<double>());
        }
        const auto single = io::combine_statistics({first});
        REQUIRE(single.has_value());
        CHECK(single->count == first.count);
        CHECK(same_bytes(single->mean, first.mean));
        CHECK(same_bytes(single->squared_deviation, first.squared_deviation));
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

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/float32-unaligned-input-accuracy") {
    const auto values = io::Array::from<float>({1e8f, -1e8f, 1, -1, -1e8f, 1e8f}, {3, 2});
    std::vector<unsigned char> bytes(values.nbytes() + 1);
    std::memcpy(bytes.data() + 1, values.data(), values.nbytes());
    const auto original = bytes;
    const auto stats = io::statistics({values.type, values.shape, bytes.data() + 1});
    REQUIRE(stats.has_value());
    CHECK(stats->count == 3);
    CHECK(stats->mean.shape == io::Shape{2});
    const auto* means = stats->mean.data<double>();
    const auto* deviations = stats->squared_deviation.data<double>();
    CHECK(close_enough(means[0], 1.0 / 3));
    CHECK(close_enough(means[1], -1.0 / 3));
    CHECK(close_enough(deviations[0], 2e16));
    CHECK(close_enough(deviations[1], 2e16));
    CHECK(bytes == original);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/chunked-conversion-preserves-sample-order") {
    std::vector<std::int32_t> values(50001 * 3);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<int>(i % 101) - 50;
    const auto array = io::Array::from(values, {50001, 3});
    const auto converted = io::convert(array.view(), io::dtype<double>());
    const auto expected = io::statistics(converted.view());
    const auto actual = io::statistics(array.view());
    REQUIRE(expected.has_value());
    REQUIRE(actual.has_value());
    CHECK(actual->count == expected->count);
    CHECK(actual->mean.shape == expected->mean.shape);
    CHECK(same_bytes(actual->mean, expected->mean));
    CHECK(same_bytes(actual->squared_deviation, expected->squared_deviation));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/basic-mean-and-squared-deviation") {
    TestDirectory directory;
    check_statistics({1, 2, 3, 4}, 2.5, 5, directory("welford.h5"));
    check_statistics({7}, 7, 0, directory("singleton.h5"));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/finite-overflow-and-nan-propagation") {
    TestDirectory directory;
    const double largest = std::numeric_limits<double>::max();
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check_statistics(std::vector<double>(9, largest), largest, 0, directory("constant-overflow.h5"));
    check_statistics({largest, -largest, 1, -1}, 0, infinity, directory("finite-overflow.h5"));
    check_statistics({1e200, -1e200, nan}, nan, nan, directory("overflow-nan.h5"));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/parallel/partition-sum-overflow-cancels") {
    const auto array = io::Array::from<double>({-1e308, 0, 1e308, 1e308, -1e308, 0}, {6});
    for (std::size_t workers : {1, 2, 3, 4}) {
        const auto stats = io::statistics(array.view(), {workers});
        REQUIRE(stats.has_value());
        CHECK(stats->mean.data<double>()[0] == 0);
        CHECK(stats->squared_deviation.data<double>()[0] == std::numeric_limits<double>::infinity());
    }
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/parallel/partitions-with-uneven-sizes") {
    for (const auto& shape : {io::Shape{23, 67}, io::Shape{257}, io::Shape{257, 13}}) {
        const std::size_t width = shape.size() == 1 ? 1 : shape[1];
        CAPTURE(width);
        std::vector<double> samples(io::elements(shape));
        for (std::size_t i = 0; i < samples.size(); ++i)
            samples[i] = 16 + (static_cast<int>((i * 17 + i / width) % 101) - 50) / 8.0;
        const auto array = io::Array::from(samples, shape);
        const auto original = array;
        for (const std::size_t workers : {4, 7}) {
            CAPTURE(workers);
            const auto serial = io::statistics(array.view(), {1});
            const auto parallel = io::statistics(array.view(), {workers});
            REQUIRE(serial.has_value());
            REQUIRE(parallel.has_value());
            CHECK(parallel->count == shape[0]);
            CHECK(parallel->mean.shape == serial->mean.shape);
            CHECK(parallel->squared_deviation.shape == serial->squared_deviation.shape);
            const auto* means = parallel->mean.data<double>();
            const auto* expected_means = serial->mean.data<double>();
            const auto* moments = parallel->squared_deviation.data<double>();
            const auto* expected_moments = serial->squared_deviation.data<double>();
            for (std::size_t column = 0; column < width; ++column) {
                CHECK(std::abs(means[column] - expected_means[column]) <= 8e-15 * std::abs(expected_means[column]));
                CHECK(std::abs(moments[column] - expected_moments[column]) <= 8e-15 * expected_moments[column]);
            }
            const auto repeated = io::statistics(array.view(), {workers});
            REQUIRE(repeated.has_value());
            CHECK(same_bytes(repeated->mean, parallel->mean));
            CHECK(same_bytes(repeated->squared_deviation, parallel->squared_deviation));
            CHECK(same_bytes(array, original));
        }
    }
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/parallel/conversion-and-unaligned-input") {
    std::vector<double> samples(257 * 13);
    for (std::size_t i = 0; i < samples.size(); ++i) samples[i] = static_cast<double>(i % 97);
    const auto source = io::Array::from(samples, {257, 13});
    for (const auto type : {io::dtype<float>(), io::dtype<double>(), io::dtype<std::int16_t>(),
                            io::dtype<std::uint64_t>(), io::dtype<bool>(),
                            io::DType{io::Kind::floating, 2}}) {
        CAPTURE(type.size);
        auto array = io::convert(source.view(), type);
        if (type.kind == io::Kind::boolean)
            for (std::size_t i = 0; i < array.nbytes(); ++i)
                array.data()[i] = i % 3 ? 255 : 0;
        const auto converted = io::convert(array.view(), io::dtype<double>());
        std::vector<unsigned char> bytes(array.nbytes() + 1);
        std::memcpy(bytes.data() + 1, array.data(), array.nbytes());
        const auto original = bytes;
        for (const std::size_t workers : {4, 7}) {
            const auto expected = io::statistics(converted.view(), {workers});
            const auto actual = io::statistics({type, array.shape, bytes.data() + 1}, {workers});
            REQUIRE(expected.has_value());
            REQUIRE(actual.has_value());
            CHECK(actual->count == expected->count);
            CHECK(actual->mean.shape == expected->mean.shape);
            CHECK(same_bytes(actual->mean, expected->mean));
            CHECK(same_bytes(actual->squared_deviation, expected->squared_deviation));
            CHECK(bytes == original);
        }
    }
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/parallel/numerical-edge-cases") {
    const double offset = 1e12 + 0.1;
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double largest = std::numeric_limits<double>::max();
    const double step = std::nextafter(offset, infinity) - offset;
    const auto verify = [&](const std::vector<double>& samples, double mean, double moment) {
        for (const std::size_t width : {1, 5, 17}) {
            CAPTURE(width);
            std::vector<double> data;
            for (const double sample : samples)
                for (std::size_t column = 0; column < width; ++column)
                    data.push_back(column % 2 ? -sample : sample);
            const auto array = io::Array::from(data, {samples.size(), width});
            const auto stats = io::statistics(array.view(), {4});
            REQUIRE(stats.has_value());
            CHECK(stats->count == samples.size());
            const auto* means = stats->mean.data<double>();
            const auto* moments = stats->squared_deviation.data<double>();
            for (std::size_t column = 0; column < width; ++column) {
                const double expected_mean = column % 2 ? -mean : mean;
                CHECK(close_enough(means[column], expected_mean));
                CHECK(close_enough(moments[column], moment));
            }
        }
    };
    verify({1e16, 1, -1e16}, 1.0 / 3, 2e32);
    std::vector<double> cancellation(259, 1);
    cancellation.front() = 1e16;
    cancellation.back() = -1e16;
    verify(cancellation, 257.0 / 259, 2e32);
    verify(std::vector<double>(9, offset), offset, 0);
    verify({offset, offset, offset + step}, offset, step * step * (2.0 / 3));
    verify({offset - 0.25, offset, offset + 0.25}, offset, 0.125);
    verify(std::vector<double>(9, largest), largest, 0);
    verify({largest, -largest, 1, -1}, 0, infinity);
    verify({1e200, -1e200}, 0, infinity);
    verify({1e200, -1e200, nan}, nan, nan);
    verify({infinity, 1, 0}, infinity, nan);
    verify({infinity, -infinity, 0}, nan, nan);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/parallel/empty-singleton-and-null-data-validation") {
    const io::StatisticsOptions defaults;
    CHECK(defaults.workers == 0);
    const io::StatisticsOptions options{8};
    for (const auto& shape : {io::Shape{}, io::Shape{0}, io::Shape{0, 2}, io::Shape{2, 2, 2}}) {
        const io::Array array(io::dtype<double>(), shape);
        CHECK_FALSE(io::statistics(array.view(), options).has_value());
    }
    const io::Array zero_width(io::dtype<double>(), {7, 0});
    const auto empty = io::statistics(zero_width.view(), options);
    REQUIRE(empty.has_value());
    CHECK(empty->count == 7);
    CHECK(empty->mean.shape == io::Shape{0});
    CHECK(empty->squared_deviation.shape == io::Shape{0});
    CHECK(empty->mean.nbytes() == 0);
    CHECK(empty->squared_deviation.nbytes() == 0);
    for (const auto& shape : {io::Shape{1}, io::Shape{1, 1}}) {
        const auto array = io::Array::from<double>({7}, shape);
        const auto stats = io::statistics(array.view(), options);
        REQUIRE(stats.has_value());
        CHECK(stats->count == 1);
        CHECK(stats->mean.shape == (shape.size() == 1 ? io::Shape{} : io::Shape{1}));
        CHECK(stats->mean.data<double>()[0] == 7);
        CHECK(stats->squared_deviation.data<double>()[0] == 0);
    }
    const auto array = io::Array::from<double>({1, 2, 3}, {3});
    CHECK_THROWS_AS(io::statistics({io::dtype<double>(), {7, 2}, nullptr}, options), std::invalid_argument);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "statistics/parallel/unsupported-dtype-error-and-recovery") {
    const io::Array unsupported({io::Kind::signed_integer, 3}, {65, 13});
    CHECK_THROWS_WITH_AS(io::statistics(unsupported.view(), {4}),
                         doctest::Contains("Unsupported numeric dtype"), io::TypeError);
    // A failed parallel request must not poison subsequent calls.
    const auto valid = io::Array::from<double>({1, 2, 3, 4}, {4});
    const auto stats = io::statistics(valid.view(), {4});
    REQUIRE(stats.has_value());
    CHECK(stats->mean.data<double>()[0] == 2.5);
    CHECK(stats->squared_deviation.data<double>()[0] == 5);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "convert/unsupported-widths-rejected") {
    const auto values = io::Array::from<double>({1, 2}, {2});
    for (const auto type : {io::DType{io::Kind::floating, 16}, io::DType{io::Kind::signed_integer, 3},
                            io::DType{io::Kind::unsigned_integer, 16}}) {
        CHECK_THROWS_WITH_AS(io::convert(values.view(), type),
                             doctest::Contains("Unsupported numeric dtype"), io::TypeError);
    }
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
    CHECK(same_values(io::convert(truth.view(), io::dtype<bool>()),
                     std::vector<bool>({true, false, true})));
    const auto floating = io::Array::from<double>(
        {-0.0, std::numeric_limits<double>::denorm_min(), -2.5,
         std::numeric_limits<double>::quiet_NaN()}, {2, 2});
    CHECK(same_values(io::convert(floating.view(), io::dtype<bool>()),
                     std::vector<bool>({false, true, true, true})));
    const io::Array empty(io::dtype<double>(), {0, 2});
    const auto empty_truth = io::convert(empty.view(), io::dtype<bool>());
    CHECK(empty_truth.shape == empty.shape);
    CHECK(empty_truth.nbytes() == 0);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "arrays/stack-preserves-exact-dtypes-and-borrowed-buffers") {
    const auto large = (std::uint64_t(1) << 53) + 1;
    const auto integers = io::Array::from<std::uint64_t>({large, large + 2}, {2});
    const auto stacked = io::internal::stack({integers.view(), integers.view()});
    CHECK(stacked.type == io::dtype<std::uint64_t>());
    CHECK(stacked.shape == io::Shape{2, 2});
    CHECK(same_values(stacked, std::vector<std::uint64_t>{large, large + 2, large, large + 2}));

    const auto floats = io::Array::from<double>({0.5, 1.5}, {2});
    const auto small_ints = io::Array::from<std::int32_t>({1, 2}, {2});
    const auto mixed = io::internal::stack({small_ints.view(), floats.view()});
    CHECK(mixed.type == io::dtype<double>());
    CHECK(same_values(mixed, std::vector<double>{1, 2, 0.5, 1.5}));
    std::vector<unsigned char> unaligned(floats.nbytes() + 1);
    std::memcpy(unaligned.data() + 1, floats.data(), floats.nbytes());
    const io::ArrayView borrowed{io::dtype<double>(), {2}, unaligned.data() + 1};
    CHECK(same_values(io::internal::stack({borrowed, floats.view()}),
                     std::vector<double>{0.5, 1.5, 0.5, 1.5}));

    const io::Array empty(io::dtype<double>(), {0, 2});
    const auto empty_stack = io::internal::stack({empty.view(), empty.view()});
    CHECK(empty_stack.shape == io::Shape{2, 0, 2});
    CHECK(empty_stack.nbytes() == 0);
    const io::ArrayView missing{io::dtype<double>(), {2}, nullptr};
    CHECK_THROWS_AS(io::internal::stack({missing, floats.view()}), std::invalid_argument);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "dict/per-key-shapes-dtypes-and-order") {
    TestDirectory directory;
    const auto path = directory("mixed.h5");
    const auto ints = io::Array::from<std::int32_t>({1, 2, 3, 4}, {2, 2});
    const auto floats = io::Array::from<double>({0.5, 1.5, 2.5}, {3});
    io::HDF5Storage::write_primal(path, io::NamedArrays{{"ints", ints.view()}, {"floats", floats.view()}});
    const auto loaded = io::HDF5Storage::open_dict(path);
    REQUIRE(loaded.size() == 2);
    CHECK(loaded[0].first == "ints");
    CHECK(loaded[1].first == "floats");
    CHECK(loaded[0].second.type == ints.type);
    CHECK(loaded[0].second.shape == ints.shape);
    CHECK(same_bytes(loaded[0].second, ints));
    CHECK(loaded[1].second.type == floats.type);
    CHECK(loaded[1].second.shape == floats.shape);
    CHECK(same_bytes(loaded[1].second, floats));
    const auto subset = io::HDF5Storage::open_dict(path, std::vector<std::string>{"floats"});
    REQUIRE(subset.size() == 1);
    CHECK(subset[0].first == "floats");
    CHECK(same_bytes(subset[0].second, floats));
    CHECK(same_bytes(io::at(loaded, "ints"), ints));
    CHECK_THROWS_AS(io::at(loaded, "missing"), io::KeyError);
    const auto shapes = io::HDF5Storage::dict_shape_from_path(path);
    REQUIRE(shapes.size() == 2);
    CHECK(shapes[0].second == ints.shape);
    CHECK(shapes[1].second == floats.shape);
    CHECK_THROWS_AS(io::HDF5Storage::open_dict(path, std::vector<std::string>{"missing"}), io::KeyError);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "dict/tangents-preserve-key-subsets") {
    TestDirectory directory;
    const auto path = directory("mixed_tangent.h5");
    const auto ints = io::Array::from<std::int32_t>({1, 2, 3, 4}, {2, 2});
    const auto floats = io::Array::from<double>({0.5, 1.5, 2.5}, {3});
    const io::NamedArrays primal_values{{"ints", ints.view()}, {"floats", floats.view()}};
    io::HDF5Storage::write_primal_tangent(path, primal_values, io::NamedArrays{{"ints", ints.view()}});
    const auto [primal, tangent] = io::HDF5Storage::open_dict_tangent(path);
    CHECK(primal.size() == 2);
    REQUIRE(tangent.has_value());
    REQUIRE(tangent->size() == 1);
    CHECK((*tangent)[0].first == "ints");
    CHECK(same_bytes((*tangent)[0].second, ints));
    const auto subset = io::HDF5Storage::open_dict_tangent(path, std::vector<std::string>{"floats"});
    REQUIRE(subset.first.size() == 1);
    CHECK(subset.first[0].first == "floats");
    CHECK(!subset.second);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/dict-preserves-zero-values-and-dtype") {
    TestDirectory directory;
    const auto primal_path = directory("mixed.h5");
    const auto gradient_path = directory("mixed_gradient.h5");
    const auto ints = io::Array::from<std::int32_t>({1, 2, 3, 4}, {2, 2});
    const auto floats = io::Array::from<double>({0.5, 1.5, 2.5}, {3});
    io::HDF5Storage::write_primal(primal_path, io::NamedArrays{{"ints", ints.view()}, {"floats", floats.view()}});
    const auto zero = io::Array::from<float>({0, 0, 0, 0}, {2, 2});
    io::HDF5Storage::write_gradient(gradient_path, io::NamedArrays{{"ints", zero.view()}});
    const auto gradient = io::HDF5Storage::load_dict_gradient(gradient_path);
    REQUIRE(gradient.size() == 1);
    CHECK(gradient[0].second.type == io::dtype<float>());
    const auto backward = io::HDF5Storage::load_backward_dict_gradient(primal_path, gradient_path);
    REQUIRE(backward.size() == 1);
    CHECK(backward[0].first == "ints");
    CHECK(same_values(backward[0].second, std::vector<float>({0, 0, 0, 0})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/dict-rejects-incompatible-shapes-and-unknown-keys") {
    TestDirectory directory;
    const auto primal_path = directory("mixed.h5");
    const auto ints = io::Array::from<std::int32_t>({1, 2, 3, 4}, {2, 2});
    io::HDF5Storage::write_primal(primal_path, io::NamedArrays{{"ints", ints.view()}});
    const auto wrong_shape = io::Array::from<float>({1, 2}, {2});
    const auto wrong_path = directory("wrong_shape.h5");
    io::HDF5Storage::write_gradient(wrong_path, io::NamedArrays{{"ints", wrong_shape.view()}});
    CHECK_THROWS_AS(io::HDF5Storage::load_backward_dict_gradient(primal_path, wrong_path), std::invalid_argument);
    const auto unknown_path = directory("unknown_key.h5");
    io::HDF5Storage::write_gradient(unknown_path, io::NamedArrays{{"missing", ints.view()}});
    CHECK_THROWS_AS(io::HDF5Storage::load_backward_dict_gradient(primal_path, unknown_path), std::invalid_argument);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/dict-aggregation-and-key-replacement") {
    TestDirectory directory;
    const auto first_path = directory("dict_grad_a.h5");
    const auto second_path = directory("dict_grad_b.h5");
    const auto sum_path = directory("dict_sum.h5");
    const auto a = io::Array::from<float>({1, 2}, {2});
    const auto other_a = io::Array::from<double>({1, 1}, {2});
    const auto b = io::Array::from<std::int32_t>({3, 4, 5}, {3});
    io::HDF5Storage::write_gradient(first_path, io::NamedArrays{{"a", a.view()}});
    io::HDF5Storage::write_gradient(second_path, io::NamedArrays{{"a", other_a.view()}, {"b", b.view()}});
    io::HDF5Storage::aggregate_gradients({first_path, second_path}, sum_path);
    const auto sum = io::HDF5Storage::load_dict_gradient(sum_path);
    REQUIRE(sum.size() == 2);
    CHECK(sum[0].first == "a");
    CHECK(sum[1].first == "b");
    CHECK(sum[0].second.type == io::dtype<double>());
    CHECK(same_values(sum[0].second, std::vector<double>({2, 3})));
    CHECK(sum[1].second.type == io::dtype<std::int32_t>());
    CHECK(same_values(sum[1].second, std::vector<std::int32_t>({3, 4, 5})));
    const auto replacement = io::Array::from<float>({9, 10}, {2});
    io::HDF5Storage::write_gradient(sum_path, io::NamedArrays{{"a", replacement.view()}}, true);
    const auto merged = io::HDF5Storage::load_dict_gradient(sum_path);
    REQUIRE(merged.size() == 2);
    CHECK(merged[0].first == "a");
    CHECK(merged[0].second.type == io::dtype<float>());
    CHECK(same_values(merged[0].second, std::vector<float>({9, 10})));
    CHECK(merged[1].first == "b");
    CHECK(same_values(merged[1].second, std::vector<std::int32_t>({3, 4, 5})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reduce/dict-concatenates-each-key-independently") {
    TestDirectory directory;
    const auto first_path = directory("mix_a.h5");
    const auto second_path = directory("mix_b.h5");
    const auto target = directory("mix_vds.h5");
    const auto a = io::Array::from<float>({1, 2}, {2});
    const auto other_a = io::Array::from<float>({3}, {1});
    const auto b = io::Array::from<std::int32_t>({10, 20, 30}, {3});
    const auto other_b = io::Array::from<std::int32_t>({40, 50}, {2});
    io::HDF5Storage::write_primal(first_path, io::NamedArrays{{"a", a.view()}, {"b", b.view()}});
    io::HDF5Storage::write_primal(second_path, io::NamedArrays{{"a", other_a.view()}, {"b", other_b.view()}});
    io::HDF5Storage::reduce({first_path, second_path}, target);
    const auto loaded = io::HDF5Storage::open_dict(target);
    REQUIRE(loaded.size() == 2);
    CHECK(loaded[0].first == "a");
    CHECK(same_values(loaded[0].second, std::vector<float>({1, 2, 3})));
    CHECK(loaded[1].first == "b");
    CHECK(same_values(loaded[1].second, std::vector<std::int32_t>({10, 20, 30, 40, 50})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reduce/dict-rejects-incompatible-dtypes") {
    TestDirectory directory;
    const auto first_path = directory("first.h5");
    const auto second_path = directory("second.h5");
    const auto target = directory("incompatible.h5");
    const auto first = io::Array::from<float>({1, 2}, {2});
    const auto second = io::Array::from<double>({4}, {1});
    io::HDF5Storage::write_primal(first_path, io::NamedArrays{{"a", first.view()}});
    io::HDF5Storage::write_primal(second_path, io::NamedArrays{{"a", second.view()}});
    CHECK_THROWS_AS(io::HDF5Storage::reduce({first_path, second_path}, target), std::invalid_argument);
    CHECK(!fs::exists(target));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reduce/scalar-dict-vds-and-adjoint") {
    TestDirectory directory;
    const auto source = directory("scalar_key.h5");
    const auto target = directory("scalar_key_vds.h5");
    const auto gradient_path = directory("scalar_key_gradient.h5");
    const auto scalar = io::Array::from<double>({1.5}, {});
    io::HDF5Storage::write_primal(source, io::NamedArrays{{"value", scalar.view()}});
    io::HDF5Storage::reduce({source, source}, target);
    const auto loaded = io::HDF5Storage::open_dict(target);
    REQUIRE(loaded.size() == 1);
    CHECK(loaded[0].first == "value");
    CHECK(loaded[0].second.shape == io::Shape({2}));
    CHECK(same_values(loaded[0].second, std::vector<double>({1.5, 1.5})));
    const auto gradient = io::Array::from<double>({2, 3}, {2});
    io::HDF5Storage::write_gradient(gradient_path, io::NamedArrays{{"value", gradient.view()}});
    const auto contributions = io::HDF5Storage::gradient_file_contributions(target, gradient_path);
    REQUIRE(contributions.size() == 2);
    for (std::size_t i = 0; i < contributions.size(); ++i) {
        REQUIRE(std::holds_alternative<io::NamedArrayValues>(contributions[i].source));
        const auto& values = std::get<io::NamedArrayValues>(contributions[i].source);
        REQUIRE(values.size() == 1);
        CHECK(values[0].second.shape.empty());
        CHECK(values[0].second.data<double>()[0] == 2.0 + i);
    }
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reduce/dict-rejects-mixed-scalar-and-non-scalar-fields") {
    TestDirectory directory;
    const auto source = directory("scalar_mix.h5");
    const auto target = directory("scalar_mix_vds.h5");
    const auto scalar = io::Array::from<double>({2.5}, {});
    const auto vector = io::Array::from<double>({1, 2}, {2});
    io::HDF5Storage::write_primal(source, io::NamedArrays{{"scalar", scalar.view()}, {"vector", vector.view()}});
    CHECK_THROWS_WITH_AS(io::HDF5Storage::reduce({source}, target),
                         doctest::Contains("mix scalar and non-scalar"), std::invalid_argument);
    CHECK(!fs::exists(target));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reduce/rejects-mixed-flat-and-dict-stores") {
    TestDirectory directory;
    const auto dict_path = directory("dict.h5");
    const auto flat_path = directory("flat.h5");
    const auto target = directory("mixed_store.h5");
    const auto values = io::Array::from<float>({1, 2}, {2});
    io::HDF5Storage::write_primal(dict_path, io::NamedArrays{{"a", values.view()}});
    io::HDF5Storage::write_primal(flat_path, values.view());
    CHECK_THROWS_AS(io::HDF5Storage::reduce({dict_path, flat_path}, target), std::invalid_argument);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "dict/legacy-column-group-without-metadata") {
    TestDirectory directory;
    const auto path = directory("legacy.h5");
    const auto a = io::Array::from<float>({1, 2}, {2});
    const auto b = io::Array::from<std::int32_t>({10, 20, 30}, {3});
    {
        auto file = io::internal::open_file(path, true);
        io::internal::set_string(file, io::store_attribute, "column");
        auto parent = io::internal::require_group(file, "primal");
        io::internal::create_dsbo_dataset(parent, "a", a.view());
        io::internal::create_dsbo_dataset(parent, "b", b.view());
        parent.close_checked();
        io::internal::finish_file(file);
    }
    const auto legacy = io::HDF5Storage::open_dict(path);
    REQUIRE(legacy.size() == 2);
    CHECK(legacy[0].first == "a");
    CHECK(legacy[1].first == "b");
}

TEST_CASE_FIXTURE(Hdf5Fixture, "headers/conversion-and-statistics-work-across-cpp-files") {
    const auto values = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {3, 2});
    const auto stats = double_statistics_from_separate_cpp_file(values.view());
    CHECK(stats.count == 3);
    CHECK(same_values(stats.mean, std::vector<double>({3, 4})));
    CHECK(same_values(stats.squared_deviation, std::vector<double>({8, 8})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/primal-tangent-round-trip") {
    TestDirectory directory;
    const auto path = directory("a.h5");
    const auto values = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {3, 2});
    io::HDF5Storage::write_primal_tangent(path, values.view(), values.view());
    const auto [primal, tangent] = io::HDF5Storage::open_primal_tangent(path);
    CHECK(same_bytes(primal, values));
    REQUIRE(tangent.has_value());
    CHECK(same_bytes(*tangent, values));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/reverse-hyperslab") {
    TestDirectory directory;
    io::Reader reader(create_rectangular_file(directory));
    CHECK(same_values(reader.read("primal", {2, -1, 3, false}),
                     std::vector<double>({5, 6, 3, 4, 1, 2})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/strided-row-selections") {
    TestDirectory directory;
    io::Reader reader(create_rectangular_file(directory));
    CHECK(same_values(reader.read("primal", {0, 2, 2, false}),
                     std::vector<double>({1, 2, 5, 6})));
    CHECK(same_values(reader.read("primal", {2, -1, 2, false}),
                     std::vector<double>({5, 6, 3, 4})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/indexed-rows-preserve-order-and-duplicates") {
    TestDirectory directory;
    io::Reader reader(create_rectangular_file(directory));
    CHECK(same_values(reader.read_indices("primal", {2, 0, 2, -2}),
                     std::vector<double>({5, 6, 1, 2, 5, 6, 3, 4})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/indexed-rows-unique-unsorted-and-empty") {
    TestDirectory directory;
    io::Reader reader(create_rectangular_file(directory));
    CHECK(same_values(reader.read_indices("primal", {2, 0, 1}),
                     std::vector<double>({5, 6, 1, 2, 3, 4})));
    CHECK(same_values(reader.read_indices("primal", {1}),
                     std::vector<double>({3, 4})));
    CHECK(reader.read_indices("primal", {}).size() == 0);
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/indexed-rows-sorted-unique-fallback") {
    TestDirectory directory;
    const auto path = directory("bool.h5");
    std::array<bool, 40> flags{};
    for (std::size_t i = 0; i < flags.size(); ++i) flags[i] = (i % 2) == 0;
    io::write_primal(path, flags);
    io::Reader reader(path);

    const std::vector<std::int64_t> even{0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
                                         20, 22, 24, 26, 28, 30, 32, 34, 36, 38};
    CHECK(same_values(reader.read_indices("primal", even), std::vector<bool>(20, true)));

    const std::vector<std::int64_t> odd{1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
                                        21, 23, 25, 27, 29, 31, 33, 35, 37, 39};
    CHECK(same_values(reader.read_indices("primal", odd), std::vector<bool>(20, false)));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/indexed-rows-rank-one-and-rank-three") {
    TestDirectory directory;
    const auto vector_path = directory("vector.h5");
    const auto vector_values = io::Array::from<double>({10, 20, 30, 40}, {4});
    io::HDF5Storage::write_primal(vector_path, vector_values.view());
    io::Reader vector_reader(vector_path);
    CHECK(same_values(vector_reader.read_indices("primal", {2, 0, 3}),
                     std::vector<double>({30, 10, 40})));
    CHECK(same_values(vector_reader.read_indices("primal", {1, 1, 0}),
                     std::vector<double>({20, 20, 10})));

    const auto cube_path = directory("cube.h5");
    std::vector<double> values(24);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<double>(i);
    const auto cube_values = io::Array::from(values, {4, 3, 2});
    io::HDF5Storage::write_primal(cube_path, cube_values.view());
    io::Reader cube_reader(cube_path);
    CHECK(same_values(cube_reader.read_indices("primal", {3, 0, 2, 1}),
                     std::vector<double>({18, 19, 20, 21, 22, 23,
                                          0, 1, 2, 3, 4, 5,
                                          12, 13, 14, 15, 16, 17,
                                          6, 7, 8, 9, 10, 11})));
    CHECK(same_values(cube_reader.read_indices("primal", {2, -1, 2}),
                     std::vector<double>({12, 13, 14, 15, 16, 17,
                                          18, 19, 20, 21, 22, 23,
                                          12, 13, 14, 15, 16, 17})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "io/persisted-statistics") {
    TestDirectory directory;
    io::Reader reader(create_rectangular_file(directory));
    const auto stats = reader.statistics("primal");
    REQUIRE(stats.has_value());
    CHECK(stats->count == 3);
    CHECK(same_values(stats->mean, std::vector<double>({3, 4})));
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
    io::Reader reader(target);
    CHECK(same_values(reader.read_indices("primal", {5, 0, 3}), std::vector<double>{5, 6, 1, 2, 1, 2}));
    CHECK(same_values(reader.read("primal", {4, -2, 3, false}), std::vector<double>{3, 4, 5, 6, 1, 2}));
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

TEST_CASE_FIXTURE(Hdf5Fixture, "dict/utf8-key-order") {
    TestDirectory directory;
    const auto path = directory("named.h5");
    const auto named = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {2, 3});
    io::HDF5Storage::write_primal(path, io::NamedArrays{{"β", named.view().column(0)}, {"first", named.view().column(1)}});
    CHECK(io::HDF5Storage::columns(path) == std::vector<std::string>({"β", "first"}));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/missing-keys-omitted") {
    TestDirectory directory;
    const auto primal_path = directory("named.h5");
    const auto gradient_path = directory("named_gradient.h5");
    const auto named = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto selected = io::Array::from<float>({7, 8, 9}, {1, 3});
    io::HDF5Storage::write_primal(primal_path, io::NamedArrays{{"β", named.view().column(0)}, {"first", named.view().column(1)}});
    io::HDF5Storage::write_gradient(gradient_path, io::NamedArrays{{"first", selected.view().column(0)}});
    const auto backward = io::HDF5Storage::load_backward_dict_gradient(primal_path, gradient_path);
    REQUIRE(backward.size() == 1);
    CHECK(backward[0].first == "first");
    CHECK(same_values(backward[0].second, std::vector<float>({7, 8, 9})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "gradient/physical-source-contribution-is-a-file-path") {
    TestDirectory directory;
    const auto primal_path = directory("named.h5");
    const auto gradient_path = directory("named_gradient.h5");
    const auto named = io::Array::from<double>({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto selected = io::Array::from<float>({7, 8, 9}, {1, 3});
    io::HDF5Storage::write_primal(primal_path, io::NamedArrays{{"β", named.view().column(0)}, {"first", named.view().column(1)}});
    io::HDF5Storage::write_gradient(gradient_path, io::NamedArrays{{"first", selected.view().column(0)}});
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
    CHECK(same_values(rows.read(io::row_store_dataset),
                     std::vector<double>({1, 0.5, 2, 1.5, 3, 2.5})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "row-store/exact-dtype-packing-and-empty-buffers") {
    TestDirectory directory;
    const auto large = (std::uint64_t(1) << 53) + 1;
    const std::uint64_t first[]{large, large + 2}, second[]{large + 4, large + 6};
    const auto path = directory("exact-rows.h5");
    io::HDF5Storage::write_row_store(path, {io::field("first", first), io::field("second", second)});
    io::Reader reader(path);
    const auto result = reader.read(io::row_store_dataset);
    CHECK(result.type == io::dtype<std::uint64_t>());
    CHECK(result.shape == io::Shape{2, 2});
    CHECK(same_values(result, std::vector<std::uint64_t>{large, large + 4, large + 2, large + 6}));

    const std::vector<float> empty;
    const auto empty_path = directory("empty-rows.h5");
    io::HDF5Storage::write_row_store(empty_path, {io::field("first", empty), io::field("second", empty)});
    const auto empty_result = io::Reader(empty_path).read(io::row_store_dataset);
    CHECK(empty_result.shape == io::Shape{0, 2});
    CHECK(empty_result.type == io::dtype<float>());
    CHECK(empty_result.nbytes() == 0);
    const io::ArrayView missing{io::dtype<float>(), {2}, nullptr};
    const auto invalid_path = directory("invalid-rows.h5");
    CHECK_THROWS_AS(io::HDF5Storage::write_row_store(invalid_path, {{"missing", missing}}), std::invalid_argument);
    CHECK_FALSE(fs::exists(invalid_path));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reduce/scalar-vds") {
    TestDirectory directory;
    const auto path = directory("scalar.h5");
    const auto target = directory("scalar-vds.h5");
    const auto scalar = io::Array::from<double>({3.5}, {});
    io::HDF5Storage::write_primal(path, scalar.view());
    io::HDF5Storage::reduce({path, path}, target);
    CHECK(same_values(io::HDF5Storage::open_primal(target), std::vector<double>({3.5, 3.5})));
}

TEST_CASE_FIXTURE(Hdf5Fixture, "reduce/vds-reader-mmap-skips-empty-partitions") {
    TestDirectory directory;
    const auto empty_source = directory("empty.h5");
    const auto full_source = directory("full.h5");
    const io::Array empty(io::dtype<double>(), {0, 2});
    const auto values = io::Array::from<double>({1, 2, 3, 4}, {2, 2});
    io::HDF5Storage::write_primal(empty_source, empty.view());
    io::HDF5Storage::write_primal(full_source, values.view());
    const auto target = directory("vds.h5");
    io::HDF5Storage::reduce({empty_source, full_source}, target);
    io::Reader reader(target);
    CHECK(same_values(reader.read_indices("primal", {1, 0}), std::vector<double>{3, 4, 1, 2}));
    CHECK(same_values(reader.read("primal"), std::vector<double>{1, 2, 3, 4}));
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
    const auto sum = io::HDF5Storage::load_gradient(sum_path);
    REQUIRE(sum.size() == 1);
    CHECK(sum.data<std::uint64_t>()[0] == 0);
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
                if (!same_bytes(io::HDF5Storage::open_primal(path), values))
                    throw std::runtime_error("Concurrent HDF5 calls returned different values");
            }
        }));
    }
    for (auto& worker : workers) CHECK_NOTHROW(worker.get());
}
