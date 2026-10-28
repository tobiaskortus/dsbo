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

Building DSBO requires a C++17 compiler and **HDF5 >=2.0 built with thread safety
enabled**. Install its development headers/libraries first and set `HDF5_ROOT`
to that installation prefix (or configure `pkg-config`). Python packaging still
uses setuptools. See [Native C++ HDF5 I/O](docs/guide/native_io.md) for installation,
copying native headers, and Python-free container examples.

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
