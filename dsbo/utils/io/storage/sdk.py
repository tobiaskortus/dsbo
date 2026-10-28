"""Export the source-only C++ SDK from an installed DSBO package."""

import json
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_cpp_include() -> Path:
    """Return the include directory containing ``dsbo/hdf5.hpp``."""
    path = Path(__file__).resolve().parent / "include"
    if not (path / "dsbo" / "hdf5.hpp").is_file():
        raise FileNotFoundError("This DSBO installation does not contain the C++ SDK headers.")
    return path


def export_cpp_sdk(destination: str | Path) -> Path:
    """Copy headers, notices, and version information into a new directory.

    The destination must not exist. No host binaries, HDF5 installation paths,
    or Python dependencies are exported. Use the result as a container build
    input, then compile against the container's thread-safe HDF5 >=2.0 SDK.
    """
    source = Path(__file__).resolve().parent
    destination = Path(destination).expanduser().resolve()
    if destination.is_relative_to(source):
        raise ValueError("Export the SDK outside its installed source directory.")
    include = get_cpp_include()
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(include, destination / "include")
    shutil.copytree(source / "licenses", destination / "licenses")
    try:
        package_version = version("dsbo")
    except PackageNotFoundError:
        package_version = "uninstalled-source"
    (destination / "sdk.json").write_text(
        json.dumps(
            {
                "dsbo_version": package_version,
                "cxx_standard": 17,
                "hdf5_minimum": "2.0",
                "hdf5_threadsafe": True,
                "nlohmann_json_version": "3.12.0",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return destination
