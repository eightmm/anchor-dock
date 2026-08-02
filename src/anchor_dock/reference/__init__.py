"""Reference-ligand MCS anchoring."""

from .._compat import dock_reference


def run_pipeline(*args, **kwargs):
    from .._compat import run_reference_pipeline

    return run_reference_pipeline(*args, **kwargs)


def run_batch(*args, **kwargs):
    from .._compat import dock_reference_batch

    return dock_reference_batch(*args, **kwargs)


run_reference_pipeline = run_pipeline
dock_reference_batch = run_batch

__all__ = [
    "dock_reference",
    "dock_reference_batch",
    "run_batch",
    "run_pipeline",
    "run_reference_pipeline",
]
