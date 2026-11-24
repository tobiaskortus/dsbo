#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <dsbo/hdf5.hpp>

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "test_support.hpp"

namespace io = dsbo::io;
namespace fs = std::filesystem;
using dsbo::test::TestDirectory;
using dsbo::test::same_values;

static fs::path native_step_path;

static void expect_array(io::Reader& reader, const std::string& dataset,
                         const io::Shape& shape, const std::vector<float>& values) {
    const auto array = reader.read(dataset);
    CHECK(array.shape == shape);
    CHECK(same_values(array, values));
}

static void run_backward(const TestDirectory& directory) {
    const std::string command = "\"" + native_step_path.string() + "\" backward \"" +
                                directory("input_gradient.h5") + "\" \"" +
                                directory("output_gradient.h5") + "\"";
    REQUIRE(std::system(command.c_str()) == 0);
}

using ExpectedGradients = std::initializer_list<std::pair<std::string, std::vector<float>>>;

static void check_named_backward(const io::NamedArrays& gradients, ExpectedGradients expected) {
    TestDirectory directory;
    const float primal_a[]{1, 2}, primal_b[]{3, 4};
    // Keep an adjacent primal to detect accidental alignment of incoming gradient keys.
    io::write_primal(directory("input.h5"), {io::field("a", primal_a), io::field("b", primal_b)});
    io::write_gradient(directory("input_gradient.h5"), gradients);
    run_backward(directory);

    io::Reader reader(directory("output_gradient.h5"));
    std::vector<std::string> columns, keys;
    for (const auto& [name, values] : expected) {
        columns.push_back(name);
        keys.push_back("gradient/" + name);
        expect_array(reader, keys.back(), {2}, values);
    }
    CHECK(reader.columns("gradient") == columns);
    std::sort(keys.begin(), keys.end());
    CHECK(reader.keys() == keys);
}

static constexpr float gradient_a[]{3, 5}, gradient_b[]{7, 11};

TEST_CASE("workflow/backward/matching-columns-double-gradient-values") {
    check_named_backward({io::field("a", gradient_a), io::field("b", gradient_b)},
                         {{"a", {6, 10}}, {"b", {14, 22}}});
}

TEST_CASE("workflow/backward/reordered-columns-retain-their-own-gradients") {
    check_named_backward({io::field("b", gradient_b), io::field("a", gradient_a)},
                         {{"b", {14, 22}}, {"a", {6, 10}}});
}

TEST_CASE("workflow/backward/partial-columns-preserve-only-incoming-columns") {
    check_named_backward({io::field("b", gradient_b)}, {{"b", {14, 22}}});
}

TEST_CASE("workflow/backward/unnamed-gradients-remain-unnamed") {
    TestDirectory directory;
    const float primal[]{1, 2, 3, 4}, gradient[]{3, 5, 7, 11};
    io::write_primal(directory("input.h5"), primal, {2, 2});
    io::write_gradient(directory("input_gradient.h5"), gradient, {2, 2});
    run_backward(directory);

    io::Reader reader(directory("output_gradient.h5"));
    CHECK(reader.columns("gradient").empty());
    CHECK(reader.keys() == std::vector<std::string>{"gradient"});
    expect_array(reader, "gradient", {2, 2}, {6, 10, 14, 22});
}

TEST_CASE("workflow/forward/doubles-primal-and-tangent") {
    TestDirectory directory;
    const io::Shape shape{2, 2};
    const std::vector<float> primal{1.25f, -2.5f, 0, 3.125f}, tangent{1, 2, 3, 4};
    io::write_primal_tangent(directory("input.h5"), primal, tangent, shape);
    const std::string command = "\"" + native_step_path.string() + "\" forward \"" +
                                directory("input.h5") + "\" \"" + directory("output.h5") + "\"";
    REQUIRE(std::system(command.c_str()) == 0);

    io::Reader output(directory("output.h5")), source(directory("input.h5"));
    expect_array(output, "primal", shape, {2.5f, -5, 0, 6.25f});
    expect_array(output, "tangent", shape, {2, 4, 6, 8});
    expect_array(source, "primal", shape, primal);
    expect_array(source, "tangent", shape, tangent);
}

int main(int argc, char** argv) {
    if (const char* configured = std::getenv("DSBO_NATIVE_STEP")) {
        native_step_path = configured;
    } else {
        native_step_path = fs::absolute(argv[0]).parent_path() / "dsbo_native_step";
        if (!fs::exists(native_step_path))
            native_step_path = fs::current_path() / "build/native/dsbo_native_step";
    }
    doctest::Context context(argc, argv);
    return context.run();
}
