"""Setuptools configuration for the single pybind11 extension.

Use pip/build to invoke this file, not legacy ``setup.py install`` commands.
HDF5 is an external development dependency; it is never downloaded here.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from setuptools.errors import CompileError


INCLUDE = Path("dsbo/utils/io/storage/include")


def hdf5_configuration():
    """Prefer explicit paths, otherwise honor the user's pkg-config setup."""
    prefix = os.environ.get("HDF5_ROOT")
    include = os.environ.get("HDF5_INCLUDE_DIR")
    library = os.environ.get("HDF5_LIBRARY_DIR")
    if prefix or include or library:
        if prefix:
            include = include or str(Path(prefix) / "include")
            library = library or str(Path(prefix) / ("lib64" if (Path(prefix) / "lib64").is_dir() else "lib"))
        if not include or not library:
            raise RuntimeError("Set HDF5_ROOT, or both HDF5_INCLUDE_DIR and HDF5_LIBRARY_DIR, to thread-safe HDF5 >=2.0.")
        return dict(
            include_dirs=[include], library_dirs=[library],
            libraries=shlex.split(os.environ.get("HDF5_LIBRARIES", "hdf5")),
            extra_compile_args=[], extra_link_args=[],
        )

    executable = os.environ.get("PKG_CONFIG", "pkg-config")
    package = os.environ.get("HDF5_PKGCONFIG_NAME", "hdf5")
    try:
        subprocess.run([executable, "--atleast-version=2.0", package], check=True, capture_output=True)
        flags = shlex.split(subprocess.check_output([executable, "--cflags", "--libs", package], text=True))
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "DSBO requires thread-safe HDF5 >=2.0 development headers and libraries. "
            "Set HDF5_ROOT to its installation prefix or configure PKG_CONFIG_PATH. "
            "Installing h5py does not provide this SDK. See docs/guide/native_io.md."
        ) from error
    result = dict(include_dirs=[], library_dirs=[], libraries=[], extra_compile_args=[], extra_link_args=[])
    for flag in flags:
        if flag.startswith("-I"):
            result["include_dirs"].append(flag[2:])
        elif flag.startswith("-L"):
            result["library_dirs"].append(flag[2:])
        elif flag.startswith("-l"):
            result["libraries"].append(flag[2:])
        elif flag.startswith("-Wl,") or flag.endswith((".so", ".dylib", ".a", ".lib")):
            result["extra_link_args"].append(flag)
        else:
            result["extra_compile_args"].append(flag)
            result["extra_link_args"].append(flag)
    return result


class HDF5BuildExt(build_ext):
    def build_extensions(self):
        extension = self.extensions[0]
        temporary = Path(self.build_temp)
        temporary.mkdir(parents=True, exist_ok=True)
        source = temporary / "check_hdf5.cpp"
        source.write_text(
            '#include <hdf5.h>\n'
            '#if H5_VERS_MAJOR < 2\n#error DSBO requires HDF5 >=2.0\n#endif\n'
            '#if !defined(H5_HAVE_THREADSAFE) || !H5_HAVE_THREADSAFE\n'
            '#error DSBO requires HDF5_ENABLE_THREADSAFE=ON\n#endif\n'
            'int main() { unsigned a, b, c; bool safe = false;\n'
            'return H5get_libversion(&a, &b, &c) < 0 || a < 2 || '
            'H5is_library_threadsafe(&safe) < 0 || !safe; }\n'
        )
        try:
            objects = self.compiler.compile(
                [str(source)], output_dir=str(temporary), include_dirs=extension.include_dirs,
                extra_postargs=extension.extra_compile_args,
            )
            probe = str(temporary / "check_hdf5")
            self.compiler.link_executable(
                objects, probe, libraries=extension.libraries, library_dirs=extension.library_dirs,
                runtime_library_dirs=extension.runtime_library_dirs,
                extra_postargs=extension.extra_link_args, target_lang="c++",
            )
            # Cross-compilers can skip executing the probe. Header checks and the
            # mandatory runtime check in the extension/SDK still apply.
            if os.environ.get("DSBO_CROSS_COMPILE") != "1":
                executable = probe + (".exe" if sys.platform == "win32" else "")
                subprocess.run([str(Path(executable).resolve())], check=True)
        except Exception as error:
            raise CompileError(
                "HDF5 validation failed: DSBO requires matching HDF5 >=2.0 headers/libraries "
                "with HDF5_ENABLE_THREADSAFE=ON. Check HDF5_ROOT and runtime library search paths."
            ) from error
        super().build_extensions()


# Metadata-only operations and SDK source archives do not need a local HDF5
# installation. Actual extension builds always discover and validate HDF5.
building = any(command in sys.argv for command in ("build_ext", "build", "bdist_wheel", "editable_wheel", "develop"))
configuration = hdf5_configuration() if building else dict(
    include_dirs=[], library_dirs=[], libraries=[], extra_compile_args=[], extra_link_args=[],
)
runtime_directories = configuration["library_dirs"] if sys.platform != "win32" else []
extension = Pybind11Extension(
    "dsbo.utils.io.storage._hdf5",
    ["cpp/bindings/hdf5.cpp"],
    cxx_std=17,
    depends=[str(path) for path in sorted(INCLUDE.rglob("*.hpp"))],
    runtime_library_dirs=runtime_directories,
    **{**configuration, "include_dirs": [str(INCLUDE), *configuration["include_dirs"]]},
)
setup(ext_modules=[extension], cmdclass={"build_ext": HDF5BuildExt})
