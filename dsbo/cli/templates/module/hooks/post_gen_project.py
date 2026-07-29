from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


PROJECT_SLUG = "{{ cookiecutter.project_slug }}"
MODULE = {
    "class_name": "{{ cookiecutter.module_class_name }}",
    "import_statement": "{{ cookiecutter.module_import }}",
    "base_name": "{{ cookiecutter.module_import }}".split()[-1],
}
MODULE_IMPORT_PATH = "{{ cookiecutter.module_import_path }}"
SHARED_TEMPLATES = Path(r"{{ cookiecutter._template }}").resolve().parent / "shared"

environment = Environment(
    loader=FileSystemLoader(SHARED_TEMPLATES),
    keep_trailing_newline=True,
    undefined=StrictUndefined,
)
project_root = Path.cwd()
package_dir = project_root / PROJECT_SLUG
tests_dir = project_root / "tests"
tests_dir.mkdir(parents=True, exist_ok=True)

(package_dir / f"{PROJECT_SLUG}.py").write_text(
    environment.get_template("module.py.jinja").render(modules=[MODULE]),
    encoding="utf-8",
)
(tests_dir / "autotests_module.py").write_text(
    environment.get_template("module_test.py.jinja").render(
        module=MODULE,
        module_import_path=MODULE_IMPORT_PATH,
    ),
    encoding="utf-8",
)
