// A Python-free float32 workflow step used standalone and in the container example.
#include <dsbo/hdf5.hpp>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace io = dsbo::io;

static io::NamedArrays twice(io::NamedArrayValues& values) {
    io::NamedArrays fields;
    fields.reserve(values.size());
    for (auto& [key, array] : values) {
        auto* data = array.data<float>();
        const auto count = array.size();
        for (std::size_t i = 0; i < count; ++i) data[i] *= 2;
        fields.push_back(io::field(key, array));
    }
    return fields;
}

int main(int argc, char** argv) {
    try {
        io::check_environment();
        if (argc != 4) throw std::invalid_argument("Usage: dsbo_native_step forward|backward INPUT.h5 OUTPUT.h5");
        const std::string mode = argv[1], input = argv[2], output = argv[3];
        io::Reader reader(input);
        if (mode == "forward") {
            if (!reader.columns("primal").empty()) {
                auto [primal, tangent] = io::open_dict_tangent(input);
                const auto fields = twice(primal);
                if (tangent) {
                    io::write_primal_tangent(output, fields, twice(*tangent));
                } else {
                    io::write_primal(output, fields);
                }
            } else {
                auto [primal, tangent] = io::open_primal_tangent(input);
                auto* data = primal.data<float>();
                const auto count = primal.size();
                for (std::size_t i = 0; i < count; ++i) data[i] *= 2;
                if (tangent) {
                    auto* tangent_data = tangent->data<float>();
                    const auto tangent_count = tangent->size();
                    for (std::size_t i = 0; i < tangent_count; ++i) tangent_data[i] *= 2;
                    io::write_primal_tangent(output, primal, *tangent);
                } else {
                    io::write_primal(output, primal);
                }
            }
        } else if (mode == "backward") {
            if (!reader.columns("gradient").empty()) {
                auto gradient = io::load_dict_gradient(input);
                io::write_gradient(output, twice(gradient));
            } else {
                auto gradient = io::load_gradient(input);
                auto* data = gradient.data<float>();
                const auto count = gradient.size();
                for (std::size_t i = 0; i < count; ++i) data[i] *= 2;
                io::write_gradient(output, gradient);
            }
        } else {
            throw std::invalid_argument("Unknown native workflow mode: " + mode);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
