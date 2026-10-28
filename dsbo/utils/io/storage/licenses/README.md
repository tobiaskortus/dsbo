# Native storage SDK notices

- `include/dsbo/third_party/json.hpp` is nlohmann/json **3.12.0**, imported
  unmodified from its `single_include/nlohmann/json.hpp`. Its MIT license is in
  `nlohmann-json.txt`.
- Python bindings use pybind11 (BSD-3-Clause); its license is in `pybind11.txt`.
- HDF5 is an external dependency. `HDF5.txt` preserves the complete HDF5 2.0.0
  license, including additional notices. If bundling another HDF5 release or
  optional libraries, preserve the licenses/notices for those actual artifacts.

These are third-party notices, not a license grant for DSBO itself. The native
implementation remains part of DSBO. Obtain maintainer/rightsholder approval
for the applicable DSBO license before external redistribution. Copying these
sources does not change ownership or licensing of the original implementation.
