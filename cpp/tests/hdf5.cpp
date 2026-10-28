#include <dsbo/hdf5.hpp>
#include <future>
#include <iostream>

namespace io = dsbo::io;
namespace fs = std::filesystem;
io::DType second_translation_unit();

static void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        io::check_environment();
        expect(second_translation_unit() == io::dtype<double>(), "Header must work across translation units");
        const auto truth = io::Array::from<std::int64_t>({-1, 0, 256}, {3});
        expect(io::convert(truth.view(), io::dtype<bool>()).values<bool>() == std::vector<bool>({true, false, true}), "Boolean conversion uses truth values");
        if (argc != 2) throw std::invalid_argument("Expected a test directory");
        const fs::path root = argv[1];
        fs::create_directories(root);
        const auto path = [&](const std::string& name) { return (root / name).string(); };
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
