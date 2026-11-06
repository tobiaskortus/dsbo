# Native C++ Development

This directory contains DSBO's header-only HDF5 backend, Python bindings, native
tests, and example executable. For integration into your own C++ application,
see the [Native C++ HDF5 I/O guide](../docs/guide/native_io.md).

## Requirements

The native test/example build requires CMake >=3.26, a C++17 compiler, and a
build tool such as Make or Ninja. It builds thread-safe HDF5 and zlib from the
checksum-verified archives in [`vendor/`](vendor/README.md). No Python, external
HDF5/zlib installation, network access, or installation step is required; all
dependency sources and build artifacts stay in the selected build directory.

## Build and Run the C++ Tests

From the repository or extracted native source release root:

```bash
cmake -S cpp -B build/native
cmake --build build/native --target check --parallel 4
```

The `check` target compiles the headers, test executables, and native example,
then runs CTest with failure output. It runs tests on every invocation and
rebuilds dependencies only when needed. After building, rerun just the tests with:

```bash
ctest --test-dir build/native --output-on-failure
```

For multi-configuration generators, add `--config Release` to the build command
and `-C Release` to the standalone CTest command.

## Package and Release Native Sources

```bash
cmake -DVERSION=0.1.0 -DOUTPUT_DIR=dist/native -P cpp/cmake/PackageSource.cmake
```

Produces `dsbo-native-0.1.0.tar.gz` and its `.tar.gz.sha256` checksum in
`dist/native/`. The archive includes headers, vendored sources, native build/tests/
example, documentation, licenses, and version/commit metadata—not Python bindings
or binaries. Extract it and use the build commands above.

Normal CI tests the extracted archive without retaining Actions artifacts. Push
a tag such as `native-v0.1.0` to publish the validated archive and checksum as a
GitHub Release. Tags with a prerelease suffix, such as native-v0.1.0-rc.1,
publish GitHub Releases marked as prereleases. Before publishing, see
[`LICENSES/README.md`](../LICENSES/README.md).

## Test a Copied Include Tree

Point `DSBO_SDK_INCLUDE_DIR` at the complete copied include directory:

```bash
cmake -S cpp -B build/native-copied -DDSBO_SDK_INCLUDE_DIR=/path/to/dsbo-sdk/include
cmake --build build/native-copied --target check --parallel 4
```

## Python Integration Checks

After installing DSBO's Python package, run its separate Python test suite from
the repository root. It covers bindings and file-format interoperability;
standalone C++ compilation and execution are covered by the CMake checks above.

```bash
python -m pytest tests
```

To exercise the Python workflow with the locally built native executable:

```bash
python examples/cpp_container/workflow.py --executable build/native/dsbo_native_step
```

For a multi-configuration build, use the executable in the selected configuration
directory, such as `build/native/Release/dsbo_native_step`. The
[container example](../docs/guide/native_io.md#use-in-a-dsbo-workflow) provides the
equivalent workflow using Docker.
