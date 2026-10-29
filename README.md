<img src="dsbo.png" width="190" align="left" style="margin-right: 20px; margin-bottom: 40px;">

<h1 style="font-size: clamp(1.0rem, 5vw, 2.5rem);">dsbo - Distributed Surrogate-based Optimization for Detector Design</h1>
<br clear="left"/>

[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![Tests](https://img.shields.io/github/actions/workflow/status/tobiaskortus/dsbo/tests.yml)](https://github.com/tobiaskortus/dsbo/actions/workflows/tests.yml)
[![Issues](https://img.shields.io/github/issues/tobiaskortus/dsbo)](https://github.com/tobiaskortus/dsbo/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/tobiaskortus/dsbo)](https://github.com/tobiaskortus/dsbo/pulls)


> [!WARNING]
> dsbo is prototype software! If you notice any issues or have any suggestions for improvements, please write up a GitHub issue detailing the problem or proposal. We also recommend “watching” the GitHub repository to keep track of new releases, and upgrading promptly when they occur.


## Getting started

DSBO's Python package includes its own thread-safe HDF5 implementation; no HDF5
installation is required. A compatible prebuilt wheel needs no compiler either.
Source and editable installations require C/C++17 compilers and automatically
build the bundled dependencies using pip-provided CMake/Ninja. Python packaging
still uses setuptools. Only standalone C++ users need their own thread-safe
HDF5 >=2.0 installation. See [Native C++ HDF5 I/O](docs/guide/native_io.md) for
details, copying native headers, and Python-free container examples.

<!--
```sh
pip install dsbo        #Full installation
```
-->

```
git clone git@github.com:tobiaskortus/dsbo.git
pip install .
```

```
pip install git+https://github.com/tobiaskortus/dsbo
```

### Extending dsbo 

dsbo is designed to be easily extensible. TODO: CLI

## License
