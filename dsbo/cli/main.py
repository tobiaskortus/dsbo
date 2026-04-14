import typer
from pathlib import Path
from cookiecutter.main import cookiecutter

app = typer.Typer()
template_root = Path(__file__).parent / "templates"

templates = ["module", "workflow", "optimizer"]
autocomplete = lambda incomplete: [t for t in templates if t.startswith(incomplete.lower())]

@app.command()
def dsbo_create(template: str = typer.Argument(..., help="Name of the template to use", autocompletion=autocomplete),
                name: str = typer.Argument(..., help="Name of the project to create")):
    
    template_path = template_root / template
    context = {"project_slug": name}

    if not template_path.exists():
        raise FileNotFoundError(f"Template {template} not found in {template_root}")
    
    cookiecutter(
        str(template_path),
        no_input=True,
        extra_context=context,
        output_dir=".",
        overwrite_if_exists=False
    )


if __name__ == "__main__":
    app()