#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <hdf5.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#include "test_support.hpp"

#ifdef _WIN32
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

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

namespace io = dsbo::io;

static std::string executable;
static bool isolated_process = false;

// Keep the production guard untouched and isolate cases regardless of the test runner.
static bool run_in_isolated_process() {
    if (isolated_process) return false;

    dsbo::test::TestDirectory directory;
    const auto log_path = directory("environment.log");
    const auto filter = std::string("--test-case=") + doctest::getContextOptions()->currentTest->m_name;
    const auto output = "--out=" + log_path;
    const char* arguments[] = {executable.c_str(), "--dsbo-isolated", filter.c_str(),
                               "--no-colors", "--no-breaks", output.c_str(), nullptr};
#ifdef _WIN32
    const auto status = _spawnvp(_P_WAIT, executable.c_str(), arguments);
    REQUIRE_MESSAGE(status != -1, "Cannot start isolated test: ", std::strerror(errno));
    const bool passed = status == 0;
#else
    pid_t child;
    const int error = posix_spawnp(&child, executable.c_str(), nullptr, nullptr,
                                   const_cast<char* const*>(arguments), environ);
    REQUIRE_MESSAGE(error == 0, "Cannot start isolated test: ", std::strerror(error));
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    REQUIRE_MESSAGE(waited == child, "Cannot wait for isolated test: ", std::strerror(errno));
    const bool passed = WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
    std::ifstream log(log_path);
    const std::string details((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
    CHECK_MESSAGE(passed, "Isolated test failed (status ", status, "):\n", details);
    return true;
}

TEST_CASE("environment/supported-check-is-cached") {
    if (run_in_isolated_process()) return;
    CHECK_NOTHROW(io::check_environment());
    CHECK(environment.version_calls == 1);
    CHECK(environment.threadsafe_calls == 1);
    CHECK_NOTHROW(io::check_environment());
    CHECK(environment.version_calls == 1);
    CHECK(environment.threadsafe_calls == 1);
}

TEST_CASE("environment/old-version") {
    if (run_in_isolated_process()) return;
    environment.major = 1;
    CHECK_THROWS_WITH_AS(io::check_environment(), doctest::Contains("HDF5 >= 2.0 at runtime"), io::IOError);
    CHECK(environment.version_calls == 1);
    CHECK(environment.threadsafe_calls == 0);
}

TEST_CASE("environment/version-query-failure") {
    if (run_in_isolated_process()) return;
    environment.version_result = -1;
    CHECK_THROWS_WITH_AS(io::check_environment(), doctest::Contains("HDF5 >= 2.0 at runtime"), io::IOError);
    CHECK(environment.version_calls == 1);
    CHECK(environment.threadsafe_calls == 0);
}

TEST_CASE("environment/non-threadsafe") {
    if (run_in_isolated_process()) return;
    environment.threadsafe = false;
    CHECK_THROWS_WITH_AS(io::check_environment(), doctest::Contains("HDF5_ENABLE_THREADSAFE=ON"), io::IOError);
    CHECK(environment.version_calls == 1);
    CHECK(environment.threadsafe_calls == 1);
}

TEST_CASE("environment/threadsafe-query-failure") {
    if (run_in_isolated_process()) return;
    environment.threadsafe_result = -1;
    CHECK_THROWS_WITH_AS(io::check_environment(), doctest::Contains("HDF5_ENABLE_THREADSAFE=ON"), io::IOError);
    CHECK(environment.version_calls == 1);
    CHECK(environment.threadsafe_calls == 1);
}

int main(int argc, char** argv) {
    executable = argv[0];
    if (argc > 1 && std::string(argv[1]) == "--dsbo-isolated") {
        isolated_process = true;
        for (int i = 1; i < argc; ++i) argv[i] = argv[i + 1];
        --argc;
    }
    doctest::Context context(argc, argv);
    return context.run();
}
