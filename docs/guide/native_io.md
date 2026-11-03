# Native C++ HDF5 I/O

DSBO's HDF5 storage implementation is header-only C++17, in
`cpp/include/dsbo/hdf5.hpp`. Python uses the same implementation
through pybind11. PyTorch autograd, workflow setup, path resolution, scheduling,
and gradient contribution bookkeeping remain in Python.

Native steps can run as standalone executables, including inside containers
without Python, PyTorch, pybind11, or an installed DSBO package.

## Python installation

**Python users do not need an HDF5 installation.** The Python extension contains
private, statically linked copies of thread-safe **HDF5 2.0.0** and **zlib 1.3.2**,
including DEFLATE/gzip support. Their symbols are hidden so DSBO can coexist with
packages such as h5py that load different HDF5/zlib libraries.

The bundled Python backend **does not support LZF compression**. Use uncompressed
or DEFLATE/gzip-compressed datasets when preparing input files for DSBO.

- When a compatible prebuilt wheel is available, pip installs it without needing
  a compiler, CMake, or HDF5 on the user's machine.
- Source and editable installs require C and C++17 compilers. Pip supplies the
  declared CMake/Ninja/pybind11 build dependencies in an isolated environment;
  setuptools remains the Python build backend.

From a checkout:

```bash
python -m pip install -e '.[dev]'
```

The build extracts checksum-verified source archives shipped in `cpp/vendor/`
and source distributions. It does not download native sources, discover a system
HDF5, or install anything system-wide. There is no HDF5 runtime dependency on a
build directory. Python builds ignore `HDF5_ROOT`, `HDF5_INCLUDE_DIR`,
`HDF5_LIBRARY_DIR`, and pkg-config settings; these are not needed for Python use.
Pip may still need network access to install Python/build dependencies. Once
those are available, the native build itself works offline.

After editing a header or binding source, rebuild with the same install command.
Editable installation does not mean C++ edits are compiled on import. Set
`CMAKE_BUILD_PARALLEL_LEVEL` to limit build concurrency. Advanced cross builds
can pass `DSBO_CMAKE_TOOLCHAIN_FILE`; `DSBO_CROSS_COMPILE=1` skips executing the
target-library validation probe on the build host. Compile-time validation and
the extension's mandatory runtime HDF5 check still apply.

## Standalone C++ requirements

These requirements apply when integrating `hdf5.hpp` into your own application,
not when using DSBO's Python functionality or running the repository's
self-contained native tests:

- C++17 compiler and your own HDF5 development headers/libraries.
- **HDF5 >=2.0 built with `HDF5_ENABLE_THREADSAFE=ON`.** The actual loaded library
  is checked, not just the build headers.
- Your preferred build system. Adopters do not need DSBO's repository CMake files.

DSBO's Python extension is not an HDF5 development SDK for native applications.
Do not link against it or h5py's private libraries. DSBO itself does not import
h5py; tests use it as an independent format reader/writer.

Thread-safe HDF5 protects library internals; it does not make multi-call file
updates transactional or allow unrelated processes to write the same file.
The conventional thread-safe mode serializes HDF5 calls internally, so this is
a safety requirement, not a promise of parallel I/O throughput. DSBO readers
serialize access and closing on each reader instance. Do not transfer live
readers across process/fork boundaries. Different readers may be used from
different threads. Coordinate writes to the same destination externally.

## Install HDF5 for standalone C++

Use a suitable prebuilt thread-safe variant, or install a standalone SDK from
the bundled source archives as below. This is **optional for adopters**: Python
installation and the repository's native `check` target do not require it.

### Install the bundled HDF5 and zlib sources

The checkout and source distribution contain HDF5 **2.0.0** and zlib **1.3.2**
under `cpp/vendor/`. The following Unix/macOS commands install both into a
user-selected prefix, with no dependency-source downloads or `sudo`. They
require CMake >=3.26, C/C++ compilers, and a build tool such as Make or Ninja.
Compare the archive hashes with `cpp/vendor/README.md` before extracting:

```bash
cmake -E sha256sum cpp/vendor/hdf5-2.0.0.tar.gz cpp/vendor/zlib-1.3.2.tar.gz
```

From the repository root, first build and install static, position-independent
zlib so that it can be linked into the shared HDF5 library:

```bash
HDF5_PREFIX="$HOME/.local/dsbo-hdf5"
mkdir -p build/hdf5-sdk-src
tar -xzf cpp/vendor/zlib-1.3.2.tar.gz -C build/hdf5-sdk-src
tar -xzf cpp/vendor/hdf5-2.0.0.tar.gz -C build/hdf5-sdk-src

cmake --fresh -S build/hdf5-sdk-src/zlib-1.3.2 -B build/hdf5-sdk-zlib \
  -DCMAKE_INSTALL_PREFIX="$HDF5_PREFIX" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DZLIB_BUILD_SHARED=OFF -DZLIB_BUILD_STATIC=ON \
  -DZLIB_BUILD_TESTING=OFF -DZLIB_INSTALL=ON
cmake --build build/hdf5-sdk-zlib --parallel 4
cmake --install build/hdf5-sdk-zlib
```

Then build the thread-safe HDF5 core C library with DEFLATE/gzip support,
using the zlib installation above instead of a system zlib:

```bash
cmake --fresh -S build/hdf5-sdk-src/hdf5-2.0.0 -B build/hdf5-sdk-hdf5 \
  -DCMAKE_INSTALL_PREFIX="$HDF5_PREFIX" \
  -DHDF5_INSTALL_LIB_DIR=lib \
  -DCMAKE_PREFIX_PATH="$HDF5_PREFIX" -DZLIB_ROOT="$HDF5_PREFIX" \
  -DZLIB_USE_STATIC_LIBS=ON -DHDF5_USE_ZLIB_STATIC=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DHDF5_ENABLE_THREADSAFE=ON \
  -DHDF5_ENABLE_PARALLEL=OFF \
  -DHDF5_BUILD_CPP_LIB=OFF -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_BUILD_HL_LIB=OFF -DHDF5_BUILD_JAVA=OFF \
  -DHDF5_ENABLE_ZLIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=OFF \
  -DHDF5_ALLOW_EXTERNAL_SUPPORT=NO \
  -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF \
  -DBUILD_TESTING=OFF -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_TOOLS=OFF
cmake --build build/hdf5-sdk-hdf5 --parallel 4
cmake --install build/hdf5-sdk-hdf5
```

These flags avoid combinations HDF5 2.0 marks unsupported with thread safety.
Disabling `HDF5_BUILD_CPP_LIB` disables HDF5's own C++ wrapper, **not** DSBO's C++
implementation, which uses the C API. Other optional filters require their own
dependencies and redistribution notices.

The installation prefix contains `include/`, `lib/`, and HDF5's CMake package
files. For an external CMake project using `find_package(HDF5 ...)`, pass
`-DHDF5_ROOT="$HDF5_PREFIX" -DCMAKE_PREFIX_PATH="$HDF5_PREFIX"`. For pkg-config,
add `$HDF5_PREFIX/lib/pkgconfig` to `PKG_CONFIG_PATH`. This SDK is independent of
DSBO's Python extension and remains installed if you delete the checkout's
`build/` directory. Use a fresh prefix when upgrading to avoid mixing versions.

## Copy the native headers

The canonical C++ headers live in `cpp/include/`. Python wheel builds copy them
into `dsbo/utils/io/storage/include/` in the installed package. In source
checkouts, including editable installations, use `cpp/include/` directly; builds
do not create a duplicate header tree inside the Python sources.
Third-party notices live in the repository's `LICENSES/` directory and are
packaged as distribution metadata, not inside the Python package. Copy the
headers and notices into your native build context; there is no Python SDK
setup or export API.
This is a **build/setup step**, not something workflow steps do at runtime.

From the root of a source checkout:

```bash
DSBO_INCLUDE=cpp/include
DSBO_LICENSES=LICENSES
mkdir -p build/container-context
mkdir build/container-context/dsbo-sdk
cp -R "$DSBO_INCLUDE" build/container-context/dsbo-sdk/include
cp -R "$DSBO_LICENSES" build/container-context/dsbo-sdk/licenses
```

To copy from a regular installed package instead, use `python -m pip show -f dsbo`
to locate its files. Set `DSBO_INCLUDE` to
`<Location>/dsbo/utils/io/storage/include` and
`DSBO_LICENSES` to the reported
`<Location>/dsbo-<version>.dist-info/licenses/LICENSES` directory before running
the copy commands. For an editable installation, use the source checkout. Use
a new destination directory when updating so files from different versions
are not mixed.

The resulting build input contains only sources and notices:

```text
dsbo-sdk/
├── include/dsbo/hdf5.hpp
├── include/dsbo/third_party/json.hpp
└── licenses/
```

Do not copy host binaries, host HDF5 libraries, or Python modules into the native
build context. Compile against the target environment's own thread-safe HDF5
>=2.0 installation. For a local build, you can also use the original `cpp/include/`
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

External tools reading DSBO outputs must use **HDF5 >=2.0**, including the HDF5
library used by h5py. Older HDF5 versions are outside the supported environment.

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

## Run the repository tests

The native test/example project builds its own thread-safe HDF5 and zlib from
the checksum-verified archives in `cpp/vendor/`, using the same dependency
configuration as the Python extension. It needs CMake >=3.26, C/C++17 compilers,
and Make or Ninja, but **no Python, external HDF5/zlib SDK, network access, or
installation step**. All dependency sources, headers, and libraries stay under
the selected build directory.

```bash
cmake -S cpp -B build/native
cmake --build build/native --target check --parallel 4
```

The `check` target builds the native test executables and example before running
CTest with failure output. It runs on every invocation; dependencies are only
rebuilt when needed. After a build, you can rerun just the tests with
`ctest --test-dir build/native --output-on-failure`. For a multi-configuration
generator, pass `--config Release` to the build command (and `-C Release` when
calling CTest directly).

If reusing a build directory from the old external-SDK setup, configure once with
`cmake --fresh -S cpp -B build/native` to discard its cached paths. To clean all
native build artifacts, delete `build/native/`; no separately installed SDK is
affected.

Native compilation and execution tests run through CMake/CTest, not pytest.
They cover multiple translation units, concurrent calls, lazy selection, VDS,
gradients, and statistics. To test a manually copied include tree, pass
`-DDSBO_SDK_INCLUDE_DIR=/path/to/dsbo-sdk/include` when configuring CMake.

CTest also checks the runtime environment guard with controlled HDF5 query
results: supported libraries are accepted, while old versions, non-thread-safe
libraries, and failed version/thread-safety queries are rejected. These tests
use the bundled dependency build; no separate non-thread-safe SDK is needed.

Python tests use h5py to inspect the format independently and verify hidden
dependency symbols, gzip/shuffle interoperability, both h5py/DSBO import orders,
and operation without external HDF5 configuration. Running pytest does not
require Docker or an external HDF5 SDK, and does not compile standalone C++
executables. The container workflow above remains a manually runnable example.

After installing DSBO's Python package, run its separate Python test suite with
`python -m pytest tests`. To manually exercise the Python workflow around the
native example, run
`python examples/cpp_container/workflow.py --executable build/native/dsbo_native_step`.

For packaging validation:

```bash
python -m build
```

The default build command builds a wheel from the resulting source archive.
Inspect the sdist for `cpp/include/`, `cpp/cmake/`, both pinned dependency archives, and
`LICENSES/`, and the wheel for SDK headers under `dsbo/utils/io/storage/include/`
and notices under `.dist-info/licenses/LICENSES/` (the source archives must not
be in the wheel). Install the wheel in a clean environment without HDF5 to check
relocation and runtime independence.
`otool -L` on macOS or `readelf -d` on Linux should show no HDF5/zlib dependency
or build-directory RPATH; only `PyInit__hdf5` should be exported from the extension.

Build wheels for each supported Python/platform/architecture combination. Static
HDF5/zlib linking does not remove the platform C/C++ runtime or Python ABI
requirements: use an appropriate deployment target/build image, and inspect or
repair release wheels with tools such as auditwheel (Linux), delocate (macOS),
or delvewheel (Windows). A wheel built on a recent Linux distribution is not
automatically a manylinux wheel. No CI configuration is added by this feature.

## Licensing

Third-party notices accompany the Python package and must be copied with the
native headers. This does not choose or change DSBO's license.
Maintainer/rightsholder approval is required
before externally redistributing DSBO's native implementation. If redistributing
HDF5 with additional filters/libraries, include those artifacts' notices too.
