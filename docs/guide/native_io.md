# Native C++ HDF5 I/O

Use DSBO's header-only C++17 HDF5 backend to read and write workflow files from
your own C++ code. Native executables can run without Python, including in
containers. DSBO's Python implementation uses this same backend via **pybind11**.

## Set up your C++ project

You need a C++17 compiler and **HDF5 >=2.0 with thread safety enabled**. Use a
prebuilt thread-safe installation or build HDF5 with
`HDF5_ENABLE_THREADSAFE=ON`. DSBO uses HDF5's core C library; HDF5's own C++
wrapper is unnecessary. DSBO checks the loaded library's version and thread
safety at runtime.

Add the checkout's `cpp/include/` directory to your compiler's include path and
include `<dsbo/hdf5.hpp>`. If copying headers into another project or container
build context, copy the **complete include directory**, including `dsbo/hdf5/`
and `dsbo/third_party/`. See [Licensing](#licensing) before redistributing.

Headers are also available in installed wheels under
`dsbo/utils/io/storage/include/`; locate them with `python -m pip show -f dsbo`.
For editable installs, use the checkout's `cpp/include/`. Link your application
to its own HDF5 installation, not the Python extension's bundled libraries.

## Read and write arrays

Save this as `step.cpp`; it writes a primal array and reads its values back:

```cpp
#include <dsbo/hdf5.hpp>
#include <iostream>

int main() {
    namespace io = dsbo::io;
    auto values = io::Array::from<double>({1.0, 2.0, 3.0}, {3});
    io::HDF5Storage::write_primal("values.h5", values.view());
    auto loaded = io::HDF5Storage::open_primal("values.h5");
    for (double value : loaded.values<double>()) {
        std::cout << value << '\n';
    }
}
```

On Unix/macOS, compile and run with the command below, adjusting the DSBO include
path and pointing `PKG_CONFIG_PATH` at your HDF5 installation's `lib/pkgconfig`
directory. This links the HDF5 C library and embeds its runtime library path:

```bash
export PKG_CONFIG_PATH="/path/to/hdf5/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
c++ -std=c++17 -pthread -I/path/to/dsbo/cpp/include step.cpp \
  $(pkg-config --cflags --libs hdf5) \
  -Wl,-rpath,"$(pkg-config --variable=libdir hdf5)" -o step
./step
```

`Array` owns its data; `values<T>()` returns a typed copy and requires a matching
dtype. `ArrayView` borrows contiguous, row-major data: keep the buffer alive and
unmodified for the duration of the call. Use
`dsbo::io::convert(array.view(), dsbo::io::dtype<T>())` for numeric conversions.

## Available operations and file compatibility

The main APIs in `dsbo::io` are:

| Task | API |
|---|---|
| Read/write primal values | `HDF5Storage::open_primal`, `write_primal` |
| Read/write values with forward tangents | `HDF5Storage::open_primal_tangent`, `write_primal_tangent` |
| Read/write reverse-mode gradients | `HDF5Storage::load_gradient`, `write_gradient` |
| Write training data | `HDF5Storage::write_named_datasets`, `write_row_store` |
| Combine files or gradient contributions | `HDF5Storage::reduce`, `aggregate_gradients` |
| Inspect or partially read datasets | `Reader::shape`, `type`, `statistics`, `read`, `read_indices`, `close` |

Arrays support booleans, signed/unsigned 8/16/32/64-bit integers, float16/32/64,
platform long double, and supported complex types. Scalar and empty shapes are
supported. Strings, objects, ragged arrays, references, and arbitrary compound
dtypes are unsupported and raise `dsbo::io::TypeError`.

Named-column primal I/O places columns on axis 0; training dataset views place
columns on the final axis. Plain `/primal` and `/gradient` datasets can be read
without DSBO store tags. Training dataset views and reductions require those
tags; use DSBO's writers to create them.

External readers, including h5py, must use **HDF5 >=2.0**. Python users do not
need a separate HDF5 installation: DSBO bundles it. For files shared with the
Python backend, use uncompressed or DEFLATE/gzip-compressed datasets; the bundled
backend does not support LZF.

Files written with the older sequential-summation writer may have inaccurate
normalization statistics. To refresh them, rewrite the source datasets with the
current writer and rebuild virtual reductions. Updating DSBO alone does not
repair existing attributes; rebuild native executables to use updated headers.

Calls on a `Reader`, including `close()`, are serialized. Keep live readers
within one process; reopen them after a fork. Coordinate writes to the same
file externally: thread-safe HDF5 does not make multi-call updates transactional.

## Use in a DSBO workflow

Have your Python module pass resolved file paths to the native executable:

| Native operation | Python handle |
|---|---|
| Forward input | `self.input()` |
| Forward output | `self.output()` |
| Incoming adjoint | `self.output.gradient()` |
| Outgoing gradient contribution | `self.input.gradient()` |

Let Python allocate gradient contribution paths rather than constructing
filenames in C++. The native step computes and writes its derivatives; the
Python wrapper handles workflow integration. See
[Computing a Gradient Contribution](gradient-flow.md#computing-a-gradient-contribution).

For a complete example, `cpp/examples/workflow_step.cpp` doubles values, forward
tangents, and incoming adjoints. Its Python wrapper is
`examples/cpp_container/workflow.py`. With Docker and DSBO installed on the host,
run these commands from the checkout to build and run a Python-free container:

```bash
mkdir -p build/container-context/dsbo-sdk/{include,licenses}
cp -R cpp/include/. build/container-context/dsbo-sdk/include/
cp -R LICENSES/. build/container-context/dsbo-sdk/licenses/
cp cpp/examples/workflow_step.cpp build/container-context/
cp examples/cpp_container/Dockerfile build/container-context/
docker build -t dsbo-native-example:latest build/container-context
python examples/cpp_container/workflow.py --image dsbo-native-example:latest
```

Mount the workdir at the same absolute path inside the container. Keep all
virtual dataset (VDS) source files accessible with their relative paths intact;
mounting only a reduced `.h5` file is insufficient. Close output files before
the step completes so downstream tasks can read them.

## Licensing

Copy the accompanying `LICENSES/` notices with redistributed headers. In wheels,
these are under `dsbo-<version>.dist-info/licenses/LICENSES/`. Include notices for
any HDF5 libraries, filters, or other dependencies you redistribute as well.

Third-party notices do not grant a license for DSBO itself. Obtain
maintainer/rightsholder approval for the applicable DSBO license before external
redistribution; see
[`LICENSES/README.md`](https://github.com/tobiaskortus/dsbo/blob/main/LICENSES/README.md).
