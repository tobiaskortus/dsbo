#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <dsbo/hdf5.hpp>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_support.hpp"

namespace io = dsbo::io;
namespace fs = std::filesystem;
using dsbo::test::TestDirectory;

static fs::path native_step_path;

static void expect_gradient(io::Reader& reader, const std::string& dataset,
                            const io::Shape& shape, const std::vector<float>& values) {
    const auto gradient = reader.read(dataset);
    CHECK(gradient.shape == shape);
    CHECK(gradient.type == io::dtype<float>());
    CHECK(gradient.values<float>() == values);
}

static std::vector<std::string> scenario_names(const std::string& scenario) {
    if (scenario == "reordered") return {"b", "a"};
    if (scenario == "partial") return {"b"};
    if (scenario == "unnamed") return {};
    if (scenario == "matching") return {"a", "b"};
    throw std::invalid_argument("Unknown workflow test case: " + scenario);
}

static void setup(const fs::path& root, const std::string& scenario) {
    const auto path = [&](const std::string& name) { return (root / name).string(); };
    const auto names = scenario_names(scenario);
    const io::Columns columns = names.empty() ? std::nullopt : io::Columns(names);

    fs::create_directories(root);
    const auto primal = io::Array::from<float>({1, 2, 3, 4}, {2, 2});
    const io::Columns primal_columns = io::Columns(std::vector<std::string>{"a", "b"});
    // The adjacent primal and _gradient suffix trigger implicit column alignment.
    io::HDF5Storage::write_primal(path("input.h5"), primal.view(), primal_columns);
    const auto gradient = scenario == "partial" ? io::Array::from<float>({7, 11}, {1, 2}) :
        scenario == "reordered" ? io::Array::from<float>({7, 11, 3, 5}, {2, 2}) :
        io::Array::from<float>({3, 5, 7, 11}, {2, 2});
    io::HDF5Storage::write_gradient(path("input_gradient.h5"), gradient.view(), columns);
}

static void verify(const fs::path& root, const std::string& scenario) {
    const auto path = [&](const std::string& name) { return (root / name).string(); };
    const auto names = scenario_names(scenario);
    io::Reader reader(path("output_gradient.h5"));
    CHECK(reader.columns("gradient") == names);
    if (names.empty()) {
        CHECK(reader.keys() == std::vector<std::string>({"gradient"}));
        expect_gradient(reader, "gradient", {2, 2}, {6, 10, 14, 22});
    } else {
        const std::vector<std::string> keys = scenario == "partial" ?
            std::vector<std::string>{"gradient/b"} : std::vector<std::string>{"gradient/a", "gradient/b"};
        CHECK(reader.keys() == keys);
        // Read datasets by name so primal-based alignment cannot mask swapped values.
        for (const auto& name : names)
            expect_gradient(reader, "gradient/" + name, {2}, name == "a" ?
                            std::vector<float>{6, 10} : std::vector<float>{14, 22});
    }
}

static int run_native_step(const fs::path& input, const fs::path& output) {
    const std::string command = "\"" + native_step_path.string() + "\" backward \"" +
                                input.string() + "\" \"" + output.string() + "\"";
    return std::system(command.c_str());
}

static void run_scenario(const std::string& scenario) {
    TestDirectory directory;
    setup(directory.path(), scenario);
    const auto status = run_native_step(directory.path() / "input_gradient.h5",
                                        directory.path() / "output_gradient.h5");
    REQUIRE(status == 0);
    verify(directory.path(), scenario);
}

TEST_CASE("workflow/backward/matching-columns-double-gradient-values") {
    run_scenario("matching");
}

TEST_CASE("workflow/backward/reordered-columns-retain-their-own-gradients") {
    run_scenario("reordered");
}

TEST_CASE("workflow/backward/partial-columns-preserve-only-incoming-columns") {
    run_scenario("partial");
}

TEST_CASE("workflow/backward/unnamed-gradients-remain-unnamed") {
    run_scenario("unnamed");
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
