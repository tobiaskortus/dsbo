# Native C++ HDF5 I/O

DSBO's HDF5 storage implementation is header-only C++17, in
`dsbo/utils/io/storage/include/dsbo/hdf5.hpp`. Python uses the same implementation
through pybind11. PyTorch autograd, workflow setup, path resolution, scheduling,
and gradient contribution bookkeeping remain in Python.

Native steps can run as standalone executables, including inside containers
without Python, PyTorch, pybind11, or an installed DSBO package.

## Requirements

- C++17 compiler and HDF5 development headers/libraries for compilation.
- **HDF5 >=2.0 built with `HDF5_ENABLE_THREADSAFE=ON`.** The actual loaded library
  is checked, as well as the headers used during the build.
- `pkg-config`, or explicit HDF5 installation paths, for the Python build.
- CMake for the standalone examples/tests, not ordinary Python package builds.

Installing `h5py` does not provide a supported native development SDK. Do not
link against its private bundled libraries. DSBO itself no longer uses h5py;
it is used by tests as an independent format reader/writer.

Thread-safe HDF5 protects library internals; it does not make multi-call file
updates transactional or allow unrelated processes to write the same file.
The conventional thread-safe mode serializes HDF5 calls internally, so this is
a safety requirement, not a promise of parallel I/O throughput. DSBO readers
serialize access and closing on each reader instance. Do not transfer live
readers across process/fork boundaries. Different readers may be used from
different threads. Coordinate writes to the same destination externally.

## Install HDF5

Use a suitable prebuilt thread-safe variant, or build the core C library from
an official HDF5 source release. For example, after extracting HDF5 2.0.0:

```bash
cmake -S hdf5-2.0.0 -B hdf5-build \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/hdf5-threadsafe" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHDF5_ENABLE_THREADSAFE=ON \
  -DHDF5_BUILD_CPP_LIB=OFF \
  -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_BUILD_HL_LIB=OFF \
  -DHDF5_ENABLE_PARALLEL=OFF \
  -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF \
  -DBUILD_TESTING=OFF -DHDF5_BUILD_EXAMPLES=OFF
cmake --build hdf5-build --parallel
cmake --install hdf5-build
```

These flags avoid combinations HDF5 2.0 marks unsupported with thread safety.
Disabling `HDF5_BUILD_CPP_LIB` disables HDF5's own C++ wrapper, **not** DSBO's C++
implementation, which uses the C API. HDF5 source builds require a sufficiently
new CMake (2.0.0 requires CMake 3.26 or newer). Optional compression/filter
dependencies must be installed/configured separately when needed.

## Build the Python package

The build backend remains setuptools. From a checkout:

```bash
export HDF5_ROOT="$HOME/.local/hdf5-threadsafe"
python -m pip install -e '.[dev]'
```

Alternatively configure `PKG_CONFIG_PATH` to find the desired `hdf5.pc`.
`HDF5_PKGCONFIG_NAME` selects a different pkg-config package name. Explicit
`HDF5_INCLUDE_DIR` and `HDF5_LIBRARY_DIR` override paths; `HDF5_LIBRARIES` is an
optional space-separated list of link libraries (default `hdf5`). No HDF5
installation is downloaded automatically by the Python build.

After editing a header or binding source, rebuild using the same install
command. Header dependencies are declared explicitly. Editable installation
does not mean C++ edits are compiled when Python imports a module.

For cross-compilation, `DSBO_CROSS_COMPILE=1` skips execution of the build probe;
compile-time header validation and runtime SDK/extension validation still apply.

The extension needs the selected HDF5 shared library at runtime. Local builds
record its library search directory. A wheel produced this way is not necessarily
portable to another machine: use platform-appropriate dependency bundling and
wheel repair tools if distributing self-contained binary wheels. Preserve the
licenses/notices for the actual bundled libraries. No CI setup is required or
provided for this feature.

## Copy the native headers

The Python package ships the C++ headers in `dsbo/utils/io/storage/include/`
and third-party notices in `dsbo/utils/io/storage/licenses/`. Copy both directories
into your native build context; there is no Python SDK setup or export API.
This is a **build/setup step**, not something workflow steps do at runtime.

From the root of a source checkout:

```bash
DSBO_STORAGE=dsbo/utils/io/storage
mkdir -p build/container-context
mkdir build/container-context/dsbo-sdk
cp -R "$DSBO_STORAGE/include" "$DSBO_STORAGE/licenses" build/container-context/dsbo-sdk/
```

To copy from a regular installed package instead, use `python -m pip show -f dsbo`
to locate its files and set `DSBO_STORAGE` to
`<Location>/dsbo/utils/io/storage` before running the copy commands. For an
editable installation, use the source checkout. Use a new destination directory
when updating the headers so files from different versions are not mixed.

The resulting build input contains only sources and notices:

```text
dsbo-sdk/
├── include/dsbo/hdf5.hpp
├── include/dsbo/third_party/json.hpp
└── licenses/
```

Do not copy host binaries, host HDF5 libraries, or Python modules into the native
build context. Compile against the target environment's own thread-safe HDF5
>=2.0 installation. For a local build, you can also use the original `include/`
directory directly without copying it.

Pin the DSBO source revision/package version and image versions used to construct
a workflow. The native headers share DSBO's release version; there is no separate
native package, generated manifest, or installed `find_package(DSBOIO)` package.

## Use from C++

```cpp
#include <dsbo/hdf5.hpp>

int main() {
    namespace io = dsbo::io;
    auto values = io::Array::from<double>({1.0, 2.0, 3.0}, {3});
    io::HDF5Writer::write_primal("values.h5", values.view());
    auto loaded = io::HDF5Writer::open_primal("values.h5");
    auto numbers = loaded.values<double>();
}
```

Supply the original or copied `include/` directory and link the HDF5 core C
library. For example on Unix, with the appropriate `PKG_CONFIG_PATH`:

```bash
c++ -std=c++17 -pthread -I/path/to/dsbo-sdk/include step.cpp \
  $(pkg-config --cflags --libs hdf5) -o step
```

Configure a runtime library search path if HDF5 is outside standard locations.
`ArrayView` borrows a contiguous buffer for a synchronous call; the caller owns
its lifetime and must not mutate it concurrently. `Array` owns returned data;
`values<T>()` returns a typed copy and requires a matching dtype. Use
`convert(array.view(), dtype<T>())` for explicit numeric conversions.

`HDF5Writer` also supports primal/tangent pairs, ordered named columns, named
dataset updates, row stores, gradient files, aggregation, virtual reduction,
and virtual-concatenation adjoint redistribution. `Reader` offers dataset
metadata, persisted statistics, full reads, sample-axis `Selection` reads,
ordered/duplicate indexed reads, and explicit `close()`.

Numeric arrays support booleans, signed/unsigned 8/16/32/64-bit integers,
float16/32/64, platform long double, and their supported complex equivalents.
Byte order and scalar/empty shapes are retained. This SDK is not a general
replacement for h5py's object, ragged, string, reference, or arbitrary compound
dtype APIs; unsupported array types raise `TypeError`.

The existing DSBO store tags, metadata, statistics, and column ordering are
retained. Named-column primal reads place columns on axis 0; dataset views place
columns on the final axis. Ordinary native `/primal` and `/gradient` files need
no DSBO store tag for primal/gradient operations; training dataset views and
reductions require the store tag. Writers use explicit HDF5 2.0 format bounds
and retain the existing compound representation for complex outputs.

## Python-free container example

After copying the headers and notices into `build/container-context/dsbo-sdk`
as above, add the example source and Dockerfile from the checkout:

```bash
cp cpp/examples/workflow_step.cpp build/container-context/
cp examples/cpp_container/Dockerfile build/container-context/
docker build -t dsbo-native-example:latest build/container-context
python examples/cpp_container/workflow.py --image dsbo-native-example:latest
```

The image uses a checksum-verified HDF5 2.0.0 source archive. It builds thread-safe
HDF5 and the executable in a build stage, then copies only the executable,
runtime libraries, and notices into the runtime stage.
The runtime stage has no Python. The example doubles values and optional forward
tangents, and its backward operation doubles incoming adjoints. The Python
workflow checks `loss=6` and `d(loss)/dx=2` for an input of 3.

The wrapper passes **resolved paths**:

| Native operation | Python path |
|---|---|
| Forward input | `Input()` |
| Forward output | `Output()` |
| Incoming adjoint | `Output.gradient()` |
| Outgoing gradient contribution | `Input.gradient()` |

Do not construct gradient filenames in native code: the Python writer scope
allocates contribution paths and routes completed results. For differentiation
with respect to design parameters, compute native derivatives and transfer
them back through the wrapper as appropriate for the module's contract.

The example mounts the workdir at the same absolute path inside the container.
The container wrapper does not translate arbitrary host path arguments.
**All VDS source files must be visible**, preserving their relative relationships;
mounting only a reduced `.h5` file is insufficient. Sources outside the workdir
require corresponding mounts too. Writers close their files before successful
task completion; downstream readers run afterwards. No SWMR, MPI, or RPC layer
is introduced by this integration.

## Local validation

```bash
python -m pytest tests
cmake -S cpp -B build/native -DHDF5_ROOT="$HDF5_ROOT"
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
DSBO_TEST_NATIVE=1 python -m pytest tests/test_native_storage.py
DSBO_TEST_CONTAINER_IMAGE=dsbo-native-example:latest python -m pytest tests/test_native_storage.py -k container
python examples/cpp_container/workflow.py --executable build/native/dsbo_native_step
```

The optional native pytest test builds against a **manually copied include tree**,
not the original package header path. Native tests include multiple translation
units, concurrent calls, lazy selection, VDS, gradients, and statistics. Python tests
use h5py to inspect the format independently. Container tests verify that Python
is absent and exchange forward/backward files with the host.

If a separate non-thread-safe HDF5 >=2.0 build is available, set
`DSBO_TEST_UNSAFE_HDF5=/path/to/that/prefix` to run rejection tests for the Python
build, the actual loaded Python runtime library, and a standalone native caller.

For packaging validation:

```bash
python -m build
```

Build a wheel from the resulting source archive as well, and inspect both
archives for SDK headers and notices. Package-building commands use setuptools;
no change to the project's Python build backend is needed.

## Licensing

Third-party notices accompany the Python package and must be copied with the
native headers. This does not choose or change DSBO's license.
Maintainer/rightsholder approval is required
before externally redistributing DSBO's native implementation. If redistributing
HDF5 with additional filters/libraries, include those artifacts' notices too.
