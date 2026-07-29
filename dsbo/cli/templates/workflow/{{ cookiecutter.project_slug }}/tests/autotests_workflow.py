import pytest

from dsbo.cli.testing import check_workflow
from {{ cookiecutter.project_slug }}.workflow import (
    generate_default_workflow,
    generate_fused_workflow,
)


def _generate_or_fail(factory):
    try:
        return factory()
    except NotImplementedError as error:
        pytest.fail(str(error), pytrace=False)


def test_generate_default_workflow_returns_workflow() -> None:
    workflow = _generate_or_fail(generate_default_workflow)
    check_workflow(workflow)


def test_generate_fused_workflow_returns_fused_workflow() -> None:
    workflow = _generate_or_fail(generate_fused_workflow)
    if workflow is None:
        pytest.skip("This project does not define a fused workflow.")
    check_workflow(workflow, require_fused=True)
