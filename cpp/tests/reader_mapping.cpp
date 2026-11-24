#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <dsbo/hdf5/internal/datasets.hpp>
#include "test_support.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <utility>

namespace io = dsbo::io;
using dsbo::test::same_values;

static bool fail_handle = false, fail_duplicate = false;
static int original_fd = -1, duplicate_fd = -1;

static herr_t mapping_vfd_handle(hid_t file, hid_t access, void** handle) {
    if (std::exchange(fail_handle, false)) return -1;
    return H5Fget_vfd_handle(file, access, handle);
}

static int mapping_duplicate(int fd) {
    original_fd = fd;
    duplicate_fd = std::exchange(fail_duplicate, false) ? -1 : ::dup(fd);
    return duplicate_fd;
}

// Keep the instrumented Reader in its own executable to avoid mixing inline definitions.
#define H5Fget_vfd_handle mapping_vfd_handle
#define dup mapping_duplicate
#include <dsbo/hdf5/reader.hpp>
#undef dup
#undef H5Fget_vfd_handle

static void write_values(const std::string& path, const std::vector<double>& values) {
    auto file = io::internal::open_file(path, true);
    const auto array = io::Array::from(values, {values.size()});
    io::internal::create_dsbo_dataset(file, "primal", array.view());
    io::internal::finish_file(file);
}

static void check_reads(io::Reader& reader) {
    CHECK(same_values(reader.read("primal"), std::vector<double>{11, 12, 13}));
    CHECK(same_values(reader.read("primal", {2, -1, 3, false}), std::vector<double>{13, 12, 11}));
    CHECK(same_values(reader.read_indices("primal", {2, 0, 2}), std::vector<double>{13, 11, 13}));
}

TEST_CASE("reader-mapping/descriptor-failures-use-hdf5") {
    dsbo::test::TestDirectory directory;
    const auto path = directory("values.h5");
    write_values(path, {11, 12, 13});
    SUBCASE("cannot-get-vfd-handle") { fail_handle = true; }
    SUBCASE("cannot-duplicate-descriptor") { fail_duplicate = true; }
    io::Reader reader(path);
    CHECK_FALSE(fail_handle);
    CHECK_FALSE(fail_duplicate);
    check_reads(reader);
}

TEST_CASE("reader-mapping/close-only-owns-duplicate") {
    dsbo::test::TestDirectory directory;
    const auto path = directory("values.h5");
    write_values(path, {11, 12, 13});
    auto file = io::internal::open_file(path);
    io::Reader reader(path);
    REQUIRE(duplicate_fd >= 0);
    CHECK(duplicate_fd != original_fd);
    check_reads(reader);
    reader.close();
    CHECK(::fcntl(duplicate_fd, F_GETFD) == -1);
    CHECK(::fcntl(original_fd, F_GETFD) >= 0);
    const auto dataset = io::internal::dataset(file, "primal");
    CHECK(same_values(io::internal::read_dataset(dataset), std::vector<double>{11, 12, 13}));
    CHECK_NOTHROW(reader.close());
}
