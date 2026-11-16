# Native C++ HDF5 I/O

Use DSBO's header-only C++17 HDF5 backend to read and write workflow files from
your own C++ code. Native executables can run without Python, including in
containers, and exchange files directly with Python modules using `HDF5File`.

## Set Up Your C++ Project

Native source releases are published on
[GitHub Releases](https://github.com/tobiaskortus/dsbo/releases). The
`dsbo-native-<version>.tar.gz` archive includes headers, vendored dependency sources,
tests, and a CMake build. It builds offline without Python; no precompiled libraries
are included. See the
[native build instructions](https://github.com/tobiaskortus/dsbo/blob/main/cpp/README.md).

You need a C++17 compiler and **HDF5 >=2.0 with thread safety enabled**. Use a
prebuilt thread-safe installation or build HDF5 with
`HDF5_ENABLE_THREADSAFE=ON`. DSBO uses HDF5's core C library; HDF5's own C++
wrapper is unnecessary.

Add `cpp/include/` from the checkout or extracted release to your compiler's
include path and include `<dsbo/hdf5.hpp>`. If copying headers into another project
or container build context, copy the **complete include directory**, including
`dsbo/hdf5/` and `dsbo/third_party/`. See [Licensing](#licensing) before redistributing.

Headers are also available in installed wheels under
`dsbo/utils/io/storage/include/`; locate them with `python -m pip show -f dsbo`.
For editable installs, use the checkout's `cpp/include/`. Link your application
to its own HDF5 installation, not the Python extension's bundled libraries.

## Read and Write Arrays

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

On Unix-like systems (e.g., Linux, macOS), compile and run with the command below,
adjusting the DSBO include path and pointing `PKG_CONFIG_PATH` at your HDF5
installation's `lib/pkgconfig` directory. This links the HDF5 C library and embeds
its runtime library path:

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

### TensorDict Stores

Use `NamedArrays` to write several named fields without stacking or converting
them to one dtype. It is an ordered vector of `(key, ArrayView)` pairs. Reads
return `NamedArrayValues`, an ordered vector of `(key, Array)` pairs that **owns**
its data. The following snippets use the headers and `io` alias from `step.cpp`
and can be placed inside `main()`:

```cpp
auto energy = io::Array::from<double>({1.0, 2.0, 3.0}, {3});
auto position = io::Array::from<float>({0, 1, 2, 3, 4, 5}, {3, 2});
io::HDF5Storage::write_primal("events.h5", io::NamedArrays{
    {"energy", energy.view()},
    {"position", position.view()},
});

auto selected = io::HDF5Storage::open_dict(
    "events.h5", std::vector<std::string>{"position", "energy"});
std::cout << selected[0].first << '\n';  // position
auto coordinates = selected[0].second.values<float>();  // six row-major values
```

Reads preserve key order; a selection returns fields in the requested order.
Omit the selection to read all fields. Missing requested primal or
gradient keys raise `dsbo::io::KeyError`; duplicate or invalid keys are rejected.
Keys must be nonempty strings without `/` or NUL characters and cannot be the
literal `"."`. Nested dictionaries are not supported.

Each field keeps its own shape and dtype. The native API permits independently
shaped fields, including scalars. To read a file through Python's `HDF5File`, use
the [TensorDict batch-shape rules](io.md#tensordict-storage): all batched fields
must share their leading dimension, and scalar fields cannot be mixed with
batched fields. The example above can therefore be read directly in Python:

```python
import torch

from dsbo.utils.io.storage import HDF5File

with torch.no_grad():
    events = HDF5File.open("events.h5", device="cpu")
    print(events["energy"].dtype)    # torch.float64
    print(events["position"].shape)  # torch.Size([3, 2])
```

Flat and keyed I/O are separate APIs. `open_primal()`, `open_primal_tangent()`,
`load_gradient()`, and `load_backward_gradient()` return arrays and reject dict
stores. Use their dict counterparts rather than passing column names to the
array methods. Inspect shapes with `shape_from_path()` for flat primals and
`dict_shape_from_path()` for keyed primals.

### Forward Tangents

`write_primal_tangent()` accepts either two array views or two `NamedArrays`.
For keyed data, tangents may cover only a subset of primal keys. Each supplied
tangent must have exactly its primal field's shape and dtype:

```cpp
auto energy_tangent = io::Array::from<double>({1.0, 1.0, 1.0}, {3});
io::HDF5Storage::write_primal_tangent(
    "dual-events.h5",
    io::NamedArrays{{"energy", energy.view()}, {"position", position.view()}},
    io::NamedArrays{{"energy", energy_tangent.view()}});

auto [primals, tangents] = io::HDF5Storage::open_dict_tangent("dual-events.h5");
// primals contains energy and position; tangents contains only energy.
auto [positions, position_tangent] = io::HDF5Storage::open_dict_tangent(
    "dual-events.h5", std::vector<std::string>{"position"});
// position_tangent is std::nullopt, not a zero-filled tangent.
```

`open_dict_tangent()` returns the primals and an optional collection of tangent
arrays. The optional is empty when none of the selected fields has a tangent.
Writing or reading a tangent does not compute derivatives; your native numerical
routine must propagate the directional derivative itself.

### Reverse-Mode Gradients

Write a flat adjoint with `write_gradient(path, array.view())`, or a keyed
adjoint with `write_gradient(path, NamedArrays{...})`. Keyed gradients are sparse:
omit fields without a derivative contribution, but keep explicit zero gradients.
`load_dict_gradient()` reads the stored keys; `load_backward_dict_gradient()`
also checks them against the primal file, requiring known keys and exact shapes.
Keep primal and gradient outputs in separate files; writes overwrite existing
destination contents by default.

Updates and additive aggregation are different operations:

```cpp
auto energy_gradient = io::Array::from<double>({2.0, 4.0, 6.0}, {3});
auto position_gradient = io::Array::from<float>({0, 0, 0, 0, 0, 0}, {3, 2});
io::HDF5Storage::write_gradient(
    "branch-a.h5", io::NamedArrays{{"energy", energy_gradient.view()}});
io::HDF5Storage::write_gradient(
    "branch-a.h5", io::NamedArrays{{"position", position_gradient.view()}}, true);
// merge=true preserves energy and adds the position key.
// Writing energy again with merge=true would REPLACE it, not add to it.

io::HDF5Storage::write_gradient(
    "branch-b.h5", io::NamedArrays{{"energy", energy_gradient.view()}});
io::HDF5Storage::aggregate_gradients({"branch-a.h5", "branch-b.h5"}, "incoming.h5");

auto adjoints = io::HDF5Storage::load_backward_dict_gradient("events.h5", "incoming.h5");
// energy is [4, 8, 12]; position is retained as an explicit zero gradient.
```

The default `merge=false` replaces the gradient file. For dict gradients,
`merge=true` replaces supplied keys while preserving other keys and their order;
new keys are appended. Create the dict gradient file with `write_gradient()`
before using `merge=true`. For flat gradients, a write always replaces the
array, regardless of `merge`.

`aggregate_gradients()` adds contributions by key, retaining the union of keys
in first-seen order. Matching keys use numeric dtype promotion and NumPy-style
broadcasting; flat and dict gradients cannot be mixed. Workflow contributions
must still match the primal's exact shapes when validated, so do not rely on
broadcasting to repair incorrectly shaped derivatives.

## Available Operations and File Compatibility

The main APIs in `dsbo::io` are:

| Task | API |
|---|---|
| Read/write flat primal values | `HDF5Storage::open_primal`, `write_primal` |
| Read/write TensorDict primal values | `HDF5Storage::open_dict`, `write_primal(NamedArrays)` |
| Read/write values with forward tangents | `HDF5Storage::open_primal_tangent` / `open_dict_tangent`, `write_primal_tangent` |
| Read/write reverse-mode gradients | `HDF5Storage::load_gradient` / `load_dict_gradient`, `write_gradient` |
| Validate and read a keyed adjoint against its primal | `HDF5Storage::load_backward_dict_gradient` |
| Inspect primal shapes | `HDF5Storage::shape_from_path` / `dict_shape_from_path` |
| Write training data | `HDF5Storage::write_named_datasets`, `write_row_store` |
| Combine files or gradient contributions | `HDF5Storage::reduce`, `aggregate_gradients` |
| Inspect or partially read datasets | `Reader::shape`, `type`, `statistics`, `read`, `read_indices`, `close` |

Arrays support booleans, signed/unsigned 8/16/32/64-bit integers, float16/32/64,
platform long double, and supported complex types. Scalar and empty shapes are
supported. Strings, objects, ragged arrays, references, and arbitrary compound
dtypes are unsupported and raise `dsbo::io::TypeError`.

Use DSBO's write methods to create files for training dataset reads and
reductions. Older named-column files are read as named arrays, not stacked
arrays; the old array-plus-columns write overloads are replaced by `NamedArrays`.

`reduce()` concatenates each named field along axis 0. Sources must have the same
keys, dtype for each key, rank, and trailing shapes. All-scalar named inputs
produce one vector per key; mixing scalar and non-scalar fields within one named
input is rejected. Flat scalar inputs also produce a vector. During a workflow
backward pass, each replica receives gradients matching the shapes of its
original outputs, including scalars. Use `aggregate_gradients()` to add
contributions instead of concatenating them.

External readers, including h5py, must use **HDF5 >=2.0**. Python users do not
need a separate HDF5 installation: DSBO bundles it. For files shared with the
Python backend, use uncompressed or DEFLATE/gzip-compressed datasets; the bundled
backend does not support LZF.

Keep each `Reader` within one process; open a separate reader in every worker.
Allow only one writer at a time for a given file.

## Use in a DSBO Workflow

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
`examples/cpp_container/workflow.py`. The native executable handles flat arrays
and TensorDict stores; keyed tangents and adjoints are matched by name rather
than by their position in a vector, and absent derivative keys stay absent.
The Python wrapper demonstrates a flat-array workflow. If you already built
the native example with the [CMake instructions](https://github.com/tobiaskortus/dsbo/blob/main/cpp/README.md),
run it locally without Docker:

```bash
python examples/cpp_container/workflow.py --executable build/native/dsbo_native_step
```

This prints `Native workflow: loss=6, d(loss)/dx=2`. With Docker and DSBO installed on the host,
run these commands from the full checkout to build and run a Python-free container
(the Python wrapper and Dockerfile are not included in native releases):

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
source files for combined inputs accessible with their relative paths intact;
mounting only the combined `.h5` file is insufficient. Close output files before
the step completes so downstream tasks can read them.

## Licensing

Copy the accompanying `LICENSES/` notices with redistributed headers. In wheels,
these are under `dsbo-<version>.dist-info/licenses/LICENSES/`. Include notices for
any HDF5 libraries, filters, or other dependencies you redistribute as well.