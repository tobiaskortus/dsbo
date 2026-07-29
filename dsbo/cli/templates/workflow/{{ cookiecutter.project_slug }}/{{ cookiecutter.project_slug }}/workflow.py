from dsbo.modules import batch_fuse
from dsbo.workflow import Workflow


def generate_default_workflow() -> Workflow:
    """Create the default workflow definition."""
    # Return a ``dsbo.workflow.Workflow`` instance from this function.
    # Compose module instances with the operators from ``dsbo.modules.dsl``.
    # For example:
    #
    # a = ModuleA()
    # b = ModuleB()
    # return Workflow(a >> b)
    raise NotImplementedError("Define and return the default Workflow.")


def generate_fused_workflow() -> Workflow | None:
    """Create the fused workflow definition."""
    # Return a ``dsbo.workflow.Workflow`` where data intensive components such as
    # reconstruction models are trained once for a batch of candidates. This 
    # enables a speedup when using workflow surrogates and variational obtimization at 
    # the cost of reduced truthfulness of objective values. Return None if this
    # not intended for the defined workflow.
    # For example:
    #
    # a = ModuleA()
    # b = ModuleB()
    # return Workflow(a >> batch_fuse(b))
    raise NotImplementedError("Define and return the fused Workflow.")
