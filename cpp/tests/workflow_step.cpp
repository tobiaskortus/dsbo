#include <dsbo/hdf5.hpp>
#include <iostream>

namespace io = dsbo::io;
namespace fs = std::filesystem;

static void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static void expect_gradient(io::Reader& reader, const std::string& dataset,
                            const io::Shape& shape, const std::vector<float>& values) {
    const auto gradient = reader.read(dataset);
    expect(gradient.shape == shape, "Backward must preserve gradient shape");
    expect(gradient.type == io::dtype<float>(), "Backward must preserve gradient dtype");
    expect(gradient.values<float>() == values, "Backward must double each column's own gradient values");
}

int main(int argc, char** argv) {
    try {
        io::check_environment();
        if (argc != 4) throw std::invalid_argument("Expected setup|verify, a scenario, and a test directory");
        const std::string mode = argv[1], scenario = argv[2];
        const fs::path root = argv[3];
        const auto path = [&](const std::string& name) { return (root / name).string(); };
        std::vector<std::string> names{"a", "b"};
        if (scenario == "reordered") names = {"b", "a"};
        else if (scenario == "partial") names = {"b"};
        else if (scenario == "unnamed") names.clear();
        else if (scenario != "matching") throw std::invalid_argument("Unknown workflow test case: " + scenario);
        const io::Columns columns = names.empty() ? std::nullopt : io::Columns(names);

        if (mode == "setup") {
            fs::create_directories(root);
            const auto primal = io::Array::from<float>({1, 2, 3, 4}, {2, 2});
            const io::Columns primal_columns = names.empty() ? std::nullopt : io::Columns(std::vector<std::string>{"a", "b"});
            // The adjacent primal and _gradient suffix trigger implicit column alignment.
            io::HDF5Writer::write_primal(path("input.h5"), primal.view(), primal_columns);
            const auto gradient = scenario == "partial" ? io::Array::from<float>({7, 11}, {1, 2}) :
                scenario == "reordered" ? io::Array::from<float>({7, 11, 3, 5}, {2, 2}) :
                io::Array::from<float>({3, 5, 7, 11}, {2, 2});
            io::HDF5Writer::write_gradient(path("input_gradient.h5"), gradient.view(), columns);
        } else if (mode == "verify") {
            io::Reader reader(path("output_gradient.h5"));
            expect(reader.columns("gradient") == names, "Backward must preserve the gradient's column order");
            if (names.empty()) {
                expect(reader.keys() == std::vector<std::string>({"gradient"}), "Unnamed gradients must remain unnamed");
                expect_gradient(reader, "gradient", {2, 2}, {6, 10, 14, 22});
            } else {
                const std::vector<std::string> keys = scenario == "partial" ?
                    std::vector<std::string>{"gradient/b"} : std::vector<std::string>{"gradient/a", "gradient/b"};
                expect(reader.keys() == keys, "Backward must preserve exactly the incoming gradient columns");
                // Read datasets by name so primal-based alignment cannot mask swapped values.
                for (const auto& name : names)
                    expect_gradient(reader, "gradient/" + name, {2}, name == "a" ?
                                    std::vector<float>{6, 10} : std::vector<float>{14, 22});
            }
        } else throw std::invalid_argument("Unknown workflow test mode: " + mode);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
