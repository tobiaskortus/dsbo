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

dsbo requires Python 3.10 or newer and is not yet available on PyPI. Install it from GitHub in an activated virtual environment:

```sh
python -m pip install "git+https://github.com/tobiaskortus/dsbo.git"
```

For development or to run the repository's examples, clone the source and install it in editable mode:

```sh
git clone https://github.com/tobiaskortus/dsbo.git
cd dsbo
python -m pip install -e .
```

See [installation instructions](docs/index.md#installation) for environment setup and optional GPU support.

## Documentation

The documentation in `docs/` covers workflow construction, optimization, and advanced topics. Serve it locally with:

```sh
python -m pip install -e ".[dev]"
mkdocs serve
```
