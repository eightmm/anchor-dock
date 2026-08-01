"""Reference-ligand MCS anchoring strategy.

The mature LigAlign pipeline remains the strategy implementation while its
scoring, masks, kinematics and optimizer are supplied by ``anchor_dock.core``.
"""

from __future__ import annotations


def dock_reference(*args, **kwargs):
    from lig_align.pipeline import run_pipeline
    result = run_pipeline(*args, **kwargs)
    result.setdefault("mode", "reference")
    result.setdefault("score_semantics", "nonbonded_pose_score_conditioned_on_reference_anchor")
    return result


def dock_reference_batch(*args, **kwargs):
    from lig_align.pipeline import run_batch
    results = run_batch(*args, **kwargs)
    for result in results:
        if "error" not in result:
            result.setdefault("mode", "reference")
            result.setdefault("score_semantics", "nonbonded_pose_score_conditioned_on_reference_anchor")
    return results


run_reference_pipeline = dock_reference
