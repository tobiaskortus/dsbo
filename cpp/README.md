# Native C++ development

This directory contains DSBO's header-only HDF5 backend, Python bindings, native
tests, and example executable. For integration into your own C++ application,
see the [Native C++ HDF5 I/O guide](../docs/guide/native_io.md).

## Requirements

The native test/example build requires CMake >=3.26, C and C++17 compilers, and a
build tool such as Make or Ninja. It builds thread-safe HDF5 and zlib from the
checksum-verified archives in [`vendor/`](vendor/README.md). No Python, external
HDF5/zlib installation, network access, or installation step is required; all
dependency sources and build artifacts stay in the selected build directory.

## Build and run the C++ tests

From the repository root:

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

The checks cover:

- Independent header compilation, mixed/repeated includes, and multiple
  translation units.
- The native API against both the original headers and a copied include tree,
  including concurrent calls, selections, virtual datasets, gradients, and
  numerical statistics.
- Runtime rejection of unsupported HDF5 versions, non-thread-safe libraries,
  and failed environment queries, using controlled query results.
- The example executable's backward operation with named and unnamed gradients.

## Test a copied include tree

Point `DSBO_SDK_INCLUDE_DIR` at the complete copied include directory:

```bash
cmake -S cpp -B build/native-copied \
  -DDSBO_SDK_INCLUDE_DIR=/path/to/dsbo-sdk/include
cmake --build build/native-copied --target check --parallel 4
```

## Reconfigure or clean

To discard cached configuration, including paths from the earlier external-SDK
setup, configure with `cmake --fresh -S cpp -B build/native`. To remove the native
build and its bundled dependencies, delete `build/native/`.

## Python integration checks

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
