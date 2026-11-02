# Bundled native dependencies

These are unmodified upstream source archives, not host binaries. They are
included in source distributions. `cpp/cmake/BundledHDF5.cmake` builds them for
both the Python extension and the Python-free C++ test/example project.
Both libraries are built statically with hidden symbols inside the respective
build directory; nothing is installed system-wide. Adopters can also install a
standalone SDK from these archives as documented in `docs/guide/native_io.md`.

| Archive | Upstream URL | SHA-256 |
|---|---|---|
| `hdf5-2.0.0.tar.gz` | https://github.com/HDFGroup/hdf5/releases/download/2.0.0/hdf5-2.0.0.tar.gz | `f4c2edc5668fb846627182708dbe1e16c60c467e63177a75b0b9f12c19d7efed` |
| `zlib-1.3.2.tar.gz` | https://zlib.net/fossils/zlib-1.3.2.tar.gz | `bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16` |

CMake validates these hashes before extraction and never downloads dependencies.
When updating, replace the archives and hashes together, preserve the upstream
licenses in `LICENSES/`, and rerun the isolation, compression,
native interoperability, and source-distribution-to-wheel checks documented in
`docs/guide/native_io.md`. The archives themselves are not included in wheels.
