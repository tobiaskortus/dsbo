// A Python-free executable used both standalone and in the container example.
#include <dsbo/hdf5.hpp>
#include <iostream>

namespace io = dsbo::io;

static io::Array twice(io::ArrayView input) {
    auto values = io::convert(input, io::dtype<double>()).values<double>();
    for (auto& value : values) value *= 2;
    const auto result = io::Array::from(values, input.shape);
    return io::convert(result.view(), input.type);
}

int main(int argc, char** argv) {
    try {
        io::check_environment();
        if (argc != 4) throw std::invalid_argument("Usage: dsbo_native_step forward|backward INPUT.h5 OUTPUT.h5");
        const std::string mode = argv[1], input = argv[2], output = argv[3];
        if (mode == "forward") {
            auto [primal, tangent] = io::HDF5Writer::open_primal_tangent(input);
            const auto names = io::HDF5Writer::columns(input);
            const io::Columns columns = names.empty() ? std::nullopt : io::Columns(names);
            const auto result = twice(primal.view());
            if (tangent) {
                const auto derivative = twice(tangent->view());
                io::HDF5Writer::write_primal_tangent(output, result.view(), derivative.view(), columns);
            } else io::HDF5Writer::write_primal(output, result.view(), columns);
        } else if (mode == "backward") {
            io::Reader reader(input);
            const auto names = reader.columns("gradient");
            const io::Columns columns = names.empty() ? std::nullopt : io::Columns(names);
            // Preserve gradient order and subsets instead of aligning to the adjacent primal.
            const auto gradient = io::HDF5Writer::load_gradient(input, columns);
            const auto derivative = twice(gradient.view());
            io::HDF5Writer::write_gradient(output, derivative.view(), columns);
        } else throw std::invalid_argument("Unknown native workflow mode: " + mode);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
