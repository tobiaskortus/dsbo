#include <hdf5.h>

namespace {
struct Environment {
    unsigned major = 2;
    bool threadsafe = true;
    herr_t version_result = 0;
    herr_t threadsafe_result = 0;
    unsigned version_calls = 0;
    unsigned threadsafe_calls = 0;
} environment;

herr_t test_hdf5_version(unsigned* major, unsigned* minor, unsigned* patch) {
    ++environment.version_calls;
    *major = environment.major;
    *minor = *patch = 0;
    return environment.version_result;
}

herr_t test_hdf5_threadsafe(bool* threadsafe) {
    ++environment.threadsafe_calls;
    *threadsafe = environment.threadsafe;
    return environment.threadsafe_result;
}
} // namespace

// Exercise the actual inline guard with controlled query results. The real
// HDF5 declarations above remain untouched; no second HDF5 SDK is required.
#define H5get_libversion test_hdf5_version
#define H5is_library_threadsafe test_hdf5_threadsafe
#include <dsbo/hdf5.hpp>
#undef H5get_libversion
#undef H5is_library_threadsafe

#include <iostream>

static void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Expected an environment test case");
        const std::string scenario = argv[1];
        std::string expected_error;
        bool expect_threadsafe_query = true;
        if (scenario == "old_version") {
            environment.major = 1;
            expected_error = "HDF5 >= 2.0 at runtime";
            expect_threadsafe_query = false;
        } else if (scenario == "version_query_failure") {
            environment.version_result = -1;
            expected_error = "HDF5 >= 2.0 at runtime";
            expect_threadsafe_query = false;
        } else if (scenario == "non_threadsafe") {
            environment.threadsafe = false;
            expected_error = "HDF5_ENABLE_THREADSAFE=ON";
        } else if (scenario == "threadsafe_query_failure") {
            environment.threadsafe_result = -1;
            expected_error = "HDF5_ENABLE_THREADSAFE=ON";
        } else if (scenario != "supported") {
            throw std::invalid_argument("Unknown environment test case: " + scenario);
        }

        bool rejected = false;
        try {
            dsbo::io::check_environment();
        } catch (const dsbo::io::IOError& error) {
            rejected = true;
            expect(!expected_error.empty(), "Supported environment was rejected");
            expect(std::string(error.what()).find(expected_error) != std::string::npos,
                   "Environment rejection did not explain the requirement");
        }
        expect(rejected == !expected_error.empty(), "Unsupported environment was accepted");
        expect(environment.version_calls == 1, "Guard must query the runtime version");
        expect(environment.threadsafe_calls == (expect_threadsafe_query ? 1u : 0u),
               "Guard must check thread safety only after validating the version");
        if (!rejected) {
            dsbo::io::check_environment();
            expect(environment.version_calls == 1 && environment.threadsafe_calls == 1,
                   "Successful environment validation must be cached");
        }
        std::cout << "Environment guard passed: " << scenario << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
