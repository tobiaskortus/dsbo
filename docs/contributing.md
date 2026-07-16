# Contributing

## Reporting Issues

Search [open and closed issues](https://github.com/tobiaskortus/dsbo/issues) before filing a new one.

### Bug Reports

- Provide a minimal reproducible example — a short script that reproduces the problem without external dependencies or complex setup.
- Include environment details: Python version, PyTorch version, Dask version, OS, and the output of `pip list`.
- Include the full traceback if an exception occurred, and any relevant warnings or log output.
- Describe what you expected to happen and what actually happened.

### Feature Requests

Open an issue explaining the use case and proposing an API sketch if applicable. Substantial features should be discussed before implementation.

## Making Changes

1. Create a topic branch from `main`. Use a descriptive prefix: `fix/` for bug fixes, `feature/` for new features, `docs/` for documentation, `refactor/` for code restructuring. For example: `git checkout -b feature/new_feature main`.
2. Install in editable mode: `pip install -e ".[dev]"`
3. Make commits of atomic, logical units. Each commit should leave the test suite passing: `pytest tests/`
4. Keep commit messages clear: the first line is a short imperative statement ("Fix ...", "Add ..."), followed by a body explaining the change if needed.
5. Check for unnecessary whitespace before committing: `git diff --check`
6. Squash intermediate commits. Do not create merge commits — rebase instead: `git rebase $(git merge-base HEAD upstream/main) --onto upstream/main && git push --force origin HEAD`
7. Run the linter and type checker: `ruff check .` and `mypy src/dsbo/`
8. Build the docs locally: `mkdocs build --strict`

## Submitting Changes

1. Push your changes to a topic branch in your fork of the repository.
2. Open a pull request against `main` with a clear title and description.
3. If a GitHub Issue exists, link to it.
4. Respond to feedback in a timely manner so the pull request can stay active.

## Development Setup

```sh
git clone git@github.com:tobiaskortus/dsbo.git
cd dsbo
pip install -e ".[dev]"
mkdocs serve
```

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
