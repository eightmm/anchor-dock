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
from .core.ad4 import AD4Scorer, AD4Weights
from .core.amber import OpenMMScorer
from .core.autogrid import load_autogrid_maps, read_autogrid_map
from .core.engine import DockingEngine
from .core.io import clear_receptor_cache
from .core.native_vina import native_vina_components, native_vina_score
from .core.prepared import PreparedLigand, receptor_context_from_prepared_pdbqt
from .core.registry import (
    ScoringFunctionInfo,
    ScoringFunctionRegistration,
    create_scoring_function,
    list_scoring_functions,
    load_scoring_plugins,
    register_scoring_function,
    scoring_function_info,
)
from .core.scoring import NeuralScorerAdapter, PairwiseScorer, resolve_scorer
from .core.smina import SminaScorer
from .covalent.pipeline import clear_covalent_context_cache
from .interaction import InteractionConstraint, clear_interaction_context_cache, dock_interaction
from .preparation import load_ligand_for_refinement, prepare_ligand, prepare_receptor
from .refinement import PoseRefinementResult, refine_pose, refine_with_scorers

run_batch_docking = dock_covalent_batch


def clear_all_caches() -> None:
    """Release cached receptor contexts, including device tensors."""
    clear_receptor_cache()
    clear_covalent_context_cache()
    clear_interaction_context_cache()


__all__ = [
    "AD4Scorer",
    "AD4Weights",
    "DockingEngine",
    "DockingJob",
    "InteractionConstraint",
    "LigandRecord",
    "NeuralScorerAdapter",
    "PairwiseScorer",
    "OpenMMScorer",
    "PoseRefinementResult",
    "PreparedLigand",
    "ScoringFunctionInfo",
    "ScoringFunctionRegistration",
    "SminaScorer",
    "clear_all_caches",
    "create_scoring_function",
    "dock_batch",
    "dock_covalent",
    "dock_covalent_batch",
    "dock_interaction",
    "dock_reference",
    "dock_reference_batch",
    "load_autogrid_maps",
    "list_scoring_functions",
    "load_ligand_for_refinement",
    "load_scoring_plugins",
    "native_vina_components",
    "native_vina_score",
    "read_autogrid_map",
    "prepare_ligand",
    "prepare_receptor",
    "receptor_context_from_prepared_pdbqt",
    "refine_pose",
    "refine_with_scorers",
    "register_scoring_function",
    "resolve_scorer",
    "scoring_function_info",
    "run_batch_docking",
    "run_covalent_pipeline",
    "run_reference_pipeline",
    "__version__",
]

