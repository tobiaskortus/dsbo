import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


PROJECT_SLUG = "{{ cookiecutter.project_slug }}"
MODULES = "{{ cookiecutter.modules }}"
MODULE_IMPORTS = "{{ cookiecutter.module_imports }}"
DEFAULT_IMPORT = "from dsbo.modules import Module"
SHARED_TEMPLATES = Path(r"{{ cookiecutter._template }}").resolve().parent / "shared"


def snake_case(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    return re.sub(r"[^a-z0-9_]+", "_", value).strip("_")


module_specs = [value.strip() for value in MODULES.split(",") if value.strip()]
import_statements = [
    value.strip() for value in MODULE_IMPORTS.split("|") if value.strip()
]
modules = []

for index, module_spec in enumerate(module_specs):
    class_name, file_name, *_ = module_spec.split(":") + [module_spec]
    import_statement = (
        import_statements[index] if index < len(import_statements) else DEFAULT_IMPORT
    )
    modules.append(
        {
            "class_name": class_name,
            "file_name": file_name,
            "import_statement": import_statement,
            "base_name": import_statement.split()[-1],
        }
    )

environment = Environment(
    loader=FileSystemLoader(SHARED_TEMPLATES),
    keep_trailing_newline=True,
    undefined=StrictUndefined,
)
project_root = Path.cwd()
modules_dir = project_root / PROJECT_SLUG / "modules"
tests_dir = project_root / "tests"
modules_dir.mkdir(parents=True, exist_ok=True)
tests_dir.mkdir(parents=True, exist_ok=True)

modules_by_file = {}
for module in modules:
    modules_by_file.setdefault(module["file_name"], []).append(module)

for file_name, grouped_modules in modules_by_file.items():
    (modules_dir / f"{file_name}.py").write_text(
        environment.get_template("module.py.jinja").render(modules=grouped_modules),
        encoding="utf-8",
    )

for module in modules:
    (tests_dir / f"autotests_{snake_case(module['class_name'])}.py").write_text(
        environment.get_template("module_test.py.jinja").render(
            module=module,
            module_import_path=f"{PROJECT_SLUG}.modules",
        ),
        encoding="utf-8",
    )

(modules_dir / "__init__.py").write_text(
    environment.get_template("modules_init.py.jinja").render(
        modules=modules,
        project_slug=PROJECT_SLUG,
    ),
    encoding="utf-8",
)
