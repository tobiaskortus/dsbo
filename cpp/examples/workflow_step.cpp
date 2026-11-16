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
        io::Reader reader(input);
        if (mode == "forward") {
            if (!reader.columns("primal").empty()) {
                auto [primal_values, tangent_values] = io::HDF5Storage::open_dict_tangent(input);
                io::NamedArrayValues result_values;
                for (const auto& [key, value] : primal_values) result_values.emplace_back(key, twice(value.view()));
                io::NamedArrays result_views;
                for (const auto& [key, value] : result_values) result_views.emplace_back(key, value.view());
                if (tangent_values) {
                    io::NamedArrayValues derivative_values;
                    for (const auto& [key, value] : *tangent_values) derivative_values.emplace_back(key, twice(value.view()));
                    io::NamedArrays derivative_views;
                    for (const auto& [key, value] : derivative_values) derivative_views.emplace_back(key, value.view());
                    io::HDF5Storage::write_primal_tangent(output, result_views, derivative_views);
                } else io::HDF5Storage::write_primal(output, result_views);
            } else {
                auto [primal, tangent] = io::HDF5Storage::open_primal_tangent(input);
                const auto result = twice(primal.view());
                if (tangent) {
                    const auto derivative = twice(tangent->view());
                    io::HDF5Storage::write_primal_tangent(output, result.view(), derivative.view());
                } else io::HDF5Storage::write_primal(output, result.view());
            }
        } else if (mode == "backward") {
            if (!reader.columns("gradient").empty()) {
                auto gradient_values = io::HDF5Storage::load_dict_gradient(input);
                io::NamedArrayValues derivative_values;
                for (const auto& [key, value] : gradient_values) derivative_values.emplace_back(key, twice(value.view()));
                io::NamedArrays derivative_views;
                for (const auto& [key, value] : derivative_values) derivative_views.emplace_back(key, value.view());
                io::HDF5Storage::write_gradient(output, derivative_views);
            } else {
                const auto gradient = io::HDF5Storage::load_gradient(input);
                const auto derivative = twice(gradient.view());
                io::HDF5Storage::write_gradient(output, derivative.view());
            }
        } else throw std::invalid_argument("Unknown native workflow mode: " + mode);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
