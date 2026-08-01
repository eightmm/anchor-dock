"""Reference-ligand MCS anchoring strategy.

The pipeline in this package constructs the anchored pose ensemble; scoring,
masks, kinematics and the torsion optimizer come from ``anchor_dock.core``.
"""

from __future__ import annotations

from .pipeline import run_batch, run_pipeline

SCORE_SEMANTICS = "nonbonded_pose_score_conditioned_on_reference_anchor"


def dock_reference(*args, **kwargs):
    result = run_pipeline(*args, **kwargs)
    result.setdefault("mode", "reference")
    result.setdefault("score_semantics", SCORE_SEMANTICS)
    return result


def dock_reference_batch(*args, **kwargs):
    results = run_batch(*args, **kwargs)
    for result in results:
        if "error" not in result:
            result.setdefault("mode", "reference")
            result.setdefault("score_semantics", SCORE_SEMANTICS)
    return results


run_reference_pipeline = dock_reference
