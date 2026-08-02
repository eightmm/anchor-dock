"""Covalent residue-warhead docking."""

from .._compat import dock_covalent


def run_covalent_pipeline(*args, **kwargs):
    from .._compat import run_covalent_pipeline as compat

    return compat(*args, **kwargs)


def dock_covalent_batch(*args, **kwargs):
    from .._compat import dock_covalent_batch as compat

    return compat(*args, **kwargs)


run_batch_docking = dock_covalent_batch

__all__ = [
    "dock_covalent",
    "dock_covalent_batch",
    "run_batch_docking",
    "run_covalent_pipeline",
]
