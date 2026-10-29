# Native storage SDK notices

- `include/dsbo/third_party/json.hpp` is nlohmann/json **3.12.0**, imported
  unmodified from its `single_include/nlohmann/json.hpp`. Its MIT license is in
  `nlohmann-json.txt`.
- Python bindings use pybind11 (BSD-3-Clause); its license is in `pybind11.txt`.
- The Python extension statically bundles thread-safe HDF5 **2.0.0** and zlib
  **1.3.2**, built from the unmodified, checksum-pinned archives in `cpp/vendor/`.
  `HDF5.txt` preserves HDF5's complete license, including additional notices;
  `zlib.txt` preserves zlib's license. Standalone C++ applications supply their
  own HDF5. Preserve notices for the actual artifacts and optional libraries you
  redistribute if these differ from the bundled Python dependencies.

These are third-party notices, not a license grant for DSBO itself. The native
implementation remains part of DSBO. Obtain maintainer/rightsholder approval
for the applicable DSBO license before external redistribution. Copying these
sources does not change ownership or licensing of the original implementation.
