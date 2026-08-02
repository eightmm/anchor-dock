"""AnchorDock public API."""

from ._compat import (
    dock_covalent,
    dock_covalent_batch,
    dock_reference,
    dock_reference_batch,
    run_covalent_pipeline,
    run_reference_pipeline,
)
from ._version import __version__
from .batch import DockingJob, LigandRecord, dock_batch
from .core.engine import DockingEngine
from .core.io import clear_receptor_cache
from .core.scoring import NeuralScorerAdapter, PairwiseScorer, resolve_scorer
from .covalent.pipeline import clear_covalent_context_cache
from .free import dock_free

run_batch_docking = dock_covalent_batch


def clear_all_caches() -> None:
    """Release cached receptor contexts, including device tensors."""
    clear_receptor_cache()
    clear_covalent_context_cache()


__all__ = [
    "DockingEngine",
    "DockingJob",
    "LigandRecord",
    "NeuralScorerAdapter",
    "PairwiseScorer",
    "clear_all_caches",
    "dock_batch",
    "dock_covalent",
    "dock_covalent_batch",
    "dock_free",
    "dock_reference",
    "dock_reference_batch",
    "resolve_scorer",
    "run_batch_docking",
    "run_covalent_pipeline",
    "run_reference_pipeline",
    "__version__",
]
