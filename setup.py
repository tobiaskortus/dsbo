"""Setuptools builds a self-contained extension using private HDF5/zlib sources.

Use pip/build to invoke this file. CMake only builds native code; setuptools
remains the Python build backend. No system HDF5 or installation-time downloads
are used, and no native dependency is installed outside the build directory.
"""

import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.errors import CompileError


ROOT = Path(__file__).resolve().parent
INCLUDE = Path("cpp/include")


class NativeHeadersBuildPy(build_py):
    """Package the canonical C++ headers without copying them into the checkout."""

    def _header_mapping(self):
        destination = Path(self.build_lib) / "dsbo/utils/io/storage/include"
        return {
            str(destination / path.relative_to(INCLUDE)): str(path)
            for path in sorted(INCLUDE.rglob("*.hpp"))
        }

    def run(self):
        super().run()
        if not self.editable_mode:
            for destination, source in self._header_mapping().items():
                self.mkpath(str(Path(destination).parent))
                self.copy_file(source, destination)

    def get_source_files(self):
        return [*super().get_source_files(), *self._header_mapping().values()]

    def get_outputs(self, include_bytecode=True):
        return list(dict.fromkeys([*super().get_outputs(include_bytecode), *self._header_mapping()]))

    def get_output_mapping(self):
        return {**super().get_output_mapping(), **self._header_mapping()}


class BundledBuildExt(build_ext):
    def build_extension(self, extension):
        destination = Path(self.get_ext_fullpath(extension.name)).resolve()
        temporary = (Path(self.build_temp) / extension.name).resolve()
        temporary.mkdir(parents=True, exist_ok=True)
        configuration = "Debug" if self.debug else "Release"
        arguments = [
            "cmake", "-S", str(ROOT / "cpp/python"), "-B", str(temporary), "-G", "Ninja",
            f"-DCMAKE_BUILD_TYPE={configuration}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DDSBO_EXTENSION_DIRECTORY={destination.parent}",
            f"-DDSBO_EXTENSION_SUFFIX={sysconfig.get_config_var('EXT_SUFFIX')}",
        ]
        if sys.platform == "darwin":
            architectures = shlex.split(os.environ.get("ARCHFLAGS", ""))
            architectures = [architectures[i + 1] for i, flag in enumerate(architectures[:-1]) if flag == "-arch"]
            if architectures:
                arguments.append(f"-DCMAKE_OSX_ARCHITECTURES={';'.join(architectures)}")
            target = os.environ.get("MACOSX_DEPLOYMENT_TARGET") or sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")
            if target:
                arguments.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={target}")
            if os.environ.get("SDKROOT"):
                arguments.append(f"-DCMAKE_OSX_SYSROOT={os.environ['SDKROOT']}")
        if os.environ.get("DSBO_CMAKE_TOOLCHAIN_FILE"):
            arguments.append(f"-DCMAKE_TOOLCHAIN_FILE={Path(os.environ['DSBO_CMAKE_TOOLCHAIN_FILE']).resolve()}")

        build = ["cmake", "--build", str(temporary), "--config", configuration]
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build += ["--parallel", str(self.parallel or min(os.cpu_count() or 2, 8))]
        try:
            subprocess.run(arguments, check=True)
            subprocess.run([*build, "--target", "dsbo_check_hdf5", "_hdf5"], check=True)
            # Cross-builds cannot necessarily execute target binaries on the host.
            if os.environ.get("DSBO_CROSS_COMPILE") != "1":
                probe = temporary / ("dsbo_check_hdf5.exe" if sys.platform == "win32" else "dsbo_check_hdf5")
                subprocess.run([str(probe)], check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise CompileError(
                "Building DSBO's bundled HDF5 extension failed. Source installations require "
                "a C/C++17 compiler and the CMake/Ninja build dependencies; a system HDF5 "
                "installation is not required. See docs/guide/native_io.md."
            ) from error


extension = Extension(
    "dsbo.utils.io.storage._hdf5",
    ["cpp/bindings/hdf5.cpp"],
    depends=[
        *[str(path) for path in sorted(INCLUDE.rglob("*.hpp"))],
        *[str(path) for path in sorted(Path("cpp/python").glob("*"))],
        *[str(path) for path in sorted(Path("cpp/cmake").glob("*.cmake"))],
        *[str(path) for path in sorted(Path("cpp/vendor").glob("*.tar.gz"))],
    ],
)
setup(ext_modules=[extension], cmdclass={"build_ext": BundledBuildExt, "build_py": NativeHeadersBuildPy})
