from __future__ import annotations

import base64
import keyword
import os
import platform
import re
import tempfile
import zlib
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import questionary
from cookiecutter.main import cookiecutter
from prompt_toolkit.styles import Style as PromptStyle
from rich import box
from rich.color import Color
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.style import Style as RichStyle
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from dsbo.cli._logo_data import LOGO_FRAMES
from dsbo.cli.registry import get_registered_types, get_type_import

import dsbo.modules
import dsbo.modules.reconstruction
import dsbo.modules.simulation

try:
    import dsbo.backends.aido.surrogate
except ModuleNotFoundError:
    pass


console = Console()
template_root = Path(__file__).parent / "templates"

PROMPT_STYLE = PromptStyle([]) if "NO_COLOR" in os.environ else PromptStyle(
    [
        ("qmark", "fg:ansicyan bold"),
        ("question", "bold"),
        ("answer", "fg:ansicyan bold"),
        ("pointer", "fg:ansicyan bold"),
        ("highlighted", "fg:ansicyan bold"),
        ("selected", "fg:ansicyan"),
        ("instruction", "fg:ansibrightblack"),
        ("text", ""),
        ("disabled", "fg:ansibrightblack italic"),
    ]
)

TEMPLATE_CHOICES = [
    questionary.Choice(
        "Module       One reusable processing unit",
        value="module",
    ),
    questionary.Choice(
        "Workflow     A project composed of multiple modules",
        value="workflow",
    ),
]

TYPE_DESCRIPTIONS = {
    "Module": "General-purpose module",
    "SimulationBase": "Simulation step",
    "ReconstructionBase": "Training and validation step",
    "Surrogate": "Learned surrogate model",
}


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class ModuleSpec:
    class_name: str
    file_name: str
    base_type: str
    import_statement: str

    def cookiecutter_value(self) -> str:
        return f"{self.class_name}:{self.file_name}"


def ask(question):
    answer = question.ask()
    if answer is None:
        raise Cancelled
    return answer


def heading(number: int, title: str, detail: str | None = None) -> None:
    console.print()
    console.print(
        Text.assemble(
            (f"{number:02}  ", "bold cyan"),
            (title, "bold"),
        )
    )
    if detail:
        console.print(f"    [dim]{detail}[/dim]")
    console.print()


def terminal_color(pixel: bytes, background: tuple[int, int, int]) -> Color:
    red, green, blue, alpha = pixel
    if alpha < 255:
        red = (red * alpha + background[0] * (255 - alpha)) // 255
        green = (green * alpha + background[1] * (255 - alpha)) // 255
        blue = (blue * alpha + background[2] * (255 - alpha)) // 255
    brightest = max(red, green, blue)
    if brightest < 120:
        scale = 120 / max(brightest, 1)
        red = min(255, round(red * scale))
        green = min(255, round(green * scale))
        blue = min(255, round(blue * scale))
    return Color.from_rgb(red, green, blue)


@lru_cache(maxsize=None)
def decode_logo_frame(width: int) -> tuple[int, bytes]:
    height, payload = LOGO_FRAMES[width]
    try:
        pixels = zlib.decompress(base64.b85decode(payload))
    except (ValueError, zlib.error) as error:
        raise ValueError(f"Invalid precomputed logo frame at width {width}") from error

    expected_size = width * height * 4
    if len(pixels) != expected_size:
        raise ValueError(
            f"Invalid precomputed logo frame: expected {expected_size} bytes, "
            f"got {len(pixels)}"
        )
    return height, pixels


@lru_cache(maxsize=16)
def render_pixel_logo(max_width: int, dark_background: bool | None = None) -> Text:
    available_widths = sorted(LOGO_FRAMES)
    if max_width < available_widths[0]:
        return Text("DSBO", style="bold #75b5de")

    width = max(frame_width for frame_width in available_widths if frame_width <= max_width)
    height, pixels = decode_logo_frame(width)
    if dark_background is None:
        dark_background = console.is_terminal
    background = (0, 0, 0) if dark_background else (255, 255, 255)

    output = Text()
    for y in range(0, height, 2):
        cells: list[tuple[str, RichStyle | None]] = []
        last_visible = -1
        for x in range(width):
            upper_offset = (y * width + x) * 4
            lower_offset = ((y + 1) * width + x) * 4
            upper = pixels[upper_offset : upper_offset + 4]
            lower = pixels[lower_offset : lower_offset + 4]
            upper_visible = upper[3] >= 48
            lower_visible = lower[3] >= 48

            if upper_visible and lower_visible:
                character = "▀"
                style = RichStyle(
                    color=terminal_color(upper, background),
                    bgcolor=terminal_color(lower, background),
                )
            elif upper_visible:
                character = "▀"
                style = RichStyle(color=terminal_color(upper, background))
            elif lower_visible:
                character = "▄"
                style = RichStyle(color=terminal_color(lower, background))
            else:
                character = " "
                style = None

            cells.append((character, style))
            if upper_visible or lower_visible:
                last_visible = x

        for character, style in cells[: last_visible + 1]:
            output.append(character, style=style)
        if y + 2 < height:
            output.append("\n")

    return output


def package_version() -> str:
    try:
        return version("dsbo")
    except PackageNotFoundError:
        return "development"


def environment_name() -> str:
    if conda_env := os.getenv("CONDA_DEFAULT_ENV"):
        return f"conda:{conda_env}"
    if virtual_env := os.getenv("VIRTUAL_ENV"):
        return f"venv:{Path(virtual_env).name}"
    return "system"


def brand_details() -> Group:
    return Group(
        Text("DSBO", style="bold #75b5de"),
        Text("Distributed surrogate-based optimization", style="dim"),
        Text(""),
        Text.assemble(
            (f"v{package_version()}", "dim"),
            ("  /  ", "dim"),
            (environment_name(), "dim"),
        ),
        Text(
            f"Python {platform.python_version()}  /  "
            f"{platform.system()} {platform.machine()}",
            style="dim",
        ),
    )


def show_header() -> None:
    logo = render_pixel_logo(30)
    if console.width >= 76:
        content = Table.grid(padding=(0, 3))
        content.add_column(width=30)
        content.add_column(vertical="middle")
        content.add_row(logo, brand_details())
    else:
        content = Group(
            render_pixel_logo(min(28, console.width)),
            Text(""),
            brand_details(),
        )

    console.print()
    console.print(content)
    console.print()


def show_status(status: str, message: str) -> None:
    styles = {
        "info": ("i", "cyan"),
        "success": ("+", "bold green"),
        "warning": ("!", "bold yellow"),
        "error": ("x", "bold red"),
    }
    symbol, style = styles[status]
    console.print(f"[{style}]{symbol}[/{style}] {message}")


def identifier_error(value: str, label: str) -> str | bool:
    if not value:
        return f"Enter a {label}."
    if not value.isidentifier() or keyword.iskeyword(value):
        return f"Use a valid Python {label}, for example detector_simulation."
    return True


def class_name_error(value: str, *, optional: bool = False) -> str | bool:
    if optional and not value:
        return True
    result = identifier_error(value, "class name")
    if result is not True:
        return result
    if not value[0].isupper():
        return "Class names should start with an uppercase letter."
    return True


def snake_case(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    return re.sub(r"[^a-z0-9_]+", "_", value).strip("_")


def pascal_case(value: str) -> str:
    return "".join(part.capitalize() for part in snake_case(value).split("_") if part)


def choose_template() -> str:
    heading(1, "Template", "Choose what you want to create.")
    return ask(
        questionary.select(
            "Project type",
            choices=TEMPLATE_CHOICES,
            pointer=">",
            qmark="",
            style=PROMPT_STYLE,
        )
    )


def choose_project_name(template: str) -> str:
    heading(2, "Project", "This becomes the directory and Python package name.")
    default = "my_module" if template == "module" else "my_workflow"

    def validate(value: str) -> str | bool:
        result = identifier_error(value, "package name")
        if result is not True:
            return result
        if (Path.cwd() / value).exists():
            return f"{value} already exists in this directory."
        return True

    return ask(
        questionary.text(
            "Project name",
            default=default,
            validate=validate,
            qmark="",
            style=PROMPT_STYLE,
        )
    ).strip()


def choose_base_type(message: str = "Base type", default: str = "Module") -> str:
    choices = []
    for name in get_registered_types():
        description = TYPE_DESCRIPTIONS.get(name, "Registered module type")
        choices.append(
            questionary.Choice(
                f"{name:<22} {description}",
                value=name,
            )
        )

    return ask(
        questionary.select(
            message,
            choices=choices,
            default=default,
            pointer=">",
            qmark="",
            style=PROMPT_STYLE,
        )
    )


def configure_module(project_name: str) -> tuple[str, str, str]:
    heading(3, "Module", "Name the class and select the interface it implements.")
    class_name = ask(
        questionary.text(
            "Class name",
            default=pascal_case(project_name),
            validate=class_name_error,
            qmark="",
            style=PROMPT_STYLE,
        )
    ).strip()
    base_type = choose_base_type()
    import_statement = get_type_import(base_type) or "from dsbo.modules import Module"
    return class_name, base_type, import_statement


def ask_module(
    existing: list[ModuleSpec],
    current: ModuleSpec | None = None,
) -> ModuleSpec:
    existing_names = {
        module.class_name
        for module in existing
        if current is None or module is not current
    }

    def validate_class_name(value: str) -> str | bool:
        result = class_name_error(value)
        if result is not True:
            return result
        if value in existing_names:
            return f"{value} is already used in this workflow."
        return True

    class_name = ask(
        questionary.text(
            "Class name",
            default=current.class_name if current else "",
            validate=validate_class_name,
            qmark="",
            style=PROMPT_STYLE,
        )
    ).strip()

    file_name = ask(
        questionary.text(
            "File name (reuse a name to add multiple classes to the same file)",
            default=current.file_name if current else snake_case(class_name),
            validate=lambda value: identifier_error(value.removesuffix(".py"), "module name"),
            qmark="",
            style=PROMPT_STYLE,
        )
    ).strip().removesuffix(".py")
    base_type = choose_base_type(default=current.base_type if current else "Module")
    import_statement = get_type_import(base_type) or "from dsbo.modules import Module"
    return ModuleSpec(class_name, file_name, base_type, import_statement)


def grouped_modules(modules: list[ModuleSpec]) -> dict[str, list[ModuleSpec]]:
    groups: dict[str, list[ModuleSpec]] = {}
    for module in modules:
        groups.setdefault(module.file_name, []).append(module)
    return groups


def module_plan_table(modules: list[ModuleSpec]) -> Table:
    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        header_style="dim",
        padding=(0, 2),
    )
    table.add_column("File", style="cyan")
    table.add_column("Classes")
    table.add_column("Base types", style="dim")
    for file_name, file_modules in grouped_modules(modules).items():
        table.add_row(
            f"{file_name}.py",
            ", ".join(module.class_name for module in file_modules),
            ", ".join(module.base_type for module in file_modules),
        )
    return table


def module_choices(modules: list[ModuleSpec]) -> list[questionary.Choice]:
    return [
        questionary.Choice(
            f"{index + 1}. {module.class_name:<24} {module.file_name}.py",
            value=index,
        )
        for index, module in enumerate(modules)
    ]


def choose_module(modules: list[ModuleSpec], message: str) -> int:
    return ask(
        questionary.select(
            message,
            choices=module_choices(modules),
            pointer=">",
            qmark="",
            style=PROMPT_STYLE,
        )
    )


def report_file_reuse(module: ModuleSpec, modules: list[ModuleSpec]) -> None:
    shared = [item for item in modules if item.file_name == module.file_name]
    if len(shared) > 1:
        classes = ", ".join(item.class_name for item in shared)
        show_status(
            "warning",
            f"{module.file_name}.py will contain {len(shared)} classes: {classes}",
        )


def configure_workflow() -> list[ModuleSpec]:
    heading(
        3,
        "Modules",
        "Add modules in the order you want to start implementing them.",
    )
    modules: list[ModuleSpec] = []
    while True:
        if modules:
            console.print(module_plan_table(modules))
            console.print()

        choices = [questionary.Choice("Add module", value="add")]
        if modules:
            choices.extend(
                [
                    questionary.Choice("Edit module", value="edit"),
                    questionary.Choice("Remove module", value="remove"),
                ]
            )
        choices.append(
            questionary.Choice(
                "Finish module setup",
                value="done",
                disabled=None if modules else "Add at least one module",
            )
        )

        action = ask(
            questionary.select(
                "Module setup",
                choices=choices,
                pointer=">",
                qmark="",
                style=PROMPT_STYLE,
            )
        )

        if action == "done":
            if not modules:
                show_status("error", "Add at least one module before continuing.")
                console.print()
                continue
            return modules
        if action == "add":
            module = ask_module(modules)
            modules.append(module)
            show_status("success", f"Added {module.class_name}.")
            report_file_reuse(module, modules)
        elif action == "edit":
            index = choose_module(modules, "Edit module")
            modules[index] = ask_module(modules, current=modules[index])
            show_status("success", f"Updated {modules[index].class_name}.")
            report_file_reuse(modules[index], modules)
        elif action == "remove":
            index = choose_module(modules, "Remove module")
            module = modules[index]
            confirmed = ask(
                questionary.confirm(
                    f"Remove {module.class_name}?",
                    default=False,
                    qmark="",
                    style=PROMPT_STYLE,
                )
            )
            if confirmed:
                modules.pop(index)
                show_status("success", f"Removed {module.class_name}.")
        console.print()


def validate_module_requirement(
    template: str,
    modules: list[ModuleSpec],
    module_class_name: str,
) -> None:
    if template == "workflow" and not modules:
        raise ValueError("Workflow projects require at least one module.")
    if template == "module" and class_name_error(module_class_name) is not True:
        raise ValueError("Module projects require a valid module class.")


def review_table(
    template: str,
    name: str,
    modules: list[ModuleSpec],
    module_class_name: str,
    base_type: str,
) -> Table:
    table = Table(box=None, show_header=False, padding=(0, 2), collapse_padding=True)
    table.add_column(style="dim", width=12)
    table.add_column(style="bold")
    table.add_row("Project", name)
    table.add_row("Template", template.capitalize())
    if template == "module":
        table.add_row("Class", module_class_name)
        table.add_row("Base type", base_type)
    else:
        module_count = len(modules)
        file_count = len(grouped_modules(modules))
        module_label = "module" if module_count == 1 else "modules"
        file_label = "file" if file_count == 1 else "files"
        table.add_row(
            "Contents",
            f"{module_count} {module_label} across {file_count} {file_label}",
        )
    table.add_row("Location", str(Path.cwd() / name))
    return table


def generated_tree(
    template: str,
    name: str,
    modules: list[ModuleSpec],
) -> Tree:
    root = Tree(f"[bold]{name}/[/bold]")
    root.add("pyproject.toml")
    package = root.add(f"{name}/")
    package.add("__init__.py")

    if template == "module":
        package.add(f"{name}.py")
        tests = root.add("tests/")
        tests.add("autotests_module.py")
    else:
        package.add("workflow.py")
        module_dir = package.add("modules/")
        module_dir.add("__init__.py")
        for file_name in grouped_modules(modules):
            module_dir.add(f"{file_name}.py")
        tests = root.add("tests/")
        tests.add("autotests_workflow.py")
        for module in modules:
            tests.add(f"autotests_{snake_case(module.class_name)}.py")

    return root


def show_shared_file_warnings(modules: list[ModuleSpec]) -> None:
    for file_name, file_modules in grouped_modules(modules).items():
        if len(file_modules) < 2:
            continue
        classes = ", ".join(module.class_name for module in file_modules)
        show_status(
            "warning",
            f"{file_name}.py will contain {len(file_modules)} classes: {classes}",
        )


def confirm_configuration(
    template: str,
    name: str,
    modules: list[ModuleSpec],
    module_class_name: str,
    base_type: str,
) -> None:
    validate_module_requirement(template, modules, module_class_name)
    heading(4, "Review")
    review_content = [
        review_table(template, name, modules, module_class_name, base_type)
    ]
    if modules:
        review_content.extend([Text(""), module_plan_table(modules)])
    review_content.extend(
        [
            Text(""),
            Text("Generated files", style="dim"),
            generated_tree(template, name, modules),
        ]
    )
    console.print(
        Panel(
            Group(*review_content),
            title="[bold]Project summary[/bold]",
            title_align="left",
            box=box.ROUNDED,
            border_style="#42667f",
            padding=(1, 2),
            width=min(console.width, 100),
        )
    )
    if modules:
        console.print()
        show_shared_file_warnings(modules)
    console.print()
    confirmed = ask(
        questionary.confirm(
            "Create project?",
            default=True,
            qmark="",
            style=PROMPT_STYLE,
        )
    )
    if not confirmed:
        raise Cancelled


def generate_project(
    template: str,
    name: str,
    modules: list[ModuleSpec],
    module_class_name: str,
    module_base_import: str,
) -> None:
    validate_module_requirement(template, modules, module_class_name)
    context = {"project_slug": name}
    if template == "module":
        context.update(
            module_class_name=module_class_name,
            module_import=module_base_import,
            module_import_path=f"{name}.{name}",
        )
    else:
        context.update(
            modules=",".join(module.cookiecutter_value() for module in modules),
            module_imports="|".join(module.import_statement for module in modules),
        )

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Creating {name}", total=None)
        with tempfile.TemporaryDirectory(
            prefix=".dsbo-cookiecutter-",
            dir=Path.cwd(),
        ) as replay_dir:
            cookiecutter(
                str(template_root / template),
                no_input=True,
                output_dir=".",
                default_config={"replay_dir": replay_dir},
                extra_context=context,
            )


def show_completion(name: str) -> None:
    console.print()
    show_status("success", f"Created [bold]{name}[/bold].")
    console.print(f"[dim]Next: cd {name} && pip install -e .[/dim]")
    console.print()


def main() -> None:
    show_header()
    try:
        template = choose_template()
        name = choose_project_name(template)

        modules = []
        module_class_name = "Module"
        module_base_type = "Module"
        module_base_import = "from dsbo.modules import Module"

        if template == "module":
            module_class_name, module_base_type, module_base_import = configure_module(
                name
            )
        else:
            modules = configure_workflow()

        confirm_configuration(
            template,
            name,
            modules,
            module_class_name,
            module_base_type,
        )
        generate_project(
            template,
            name,
            modules,
            module_class_name,
            module_base_import,
        )
    except (Cancelled, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return

    show_completion(name)


if __name__ == "__main__":
    main()
