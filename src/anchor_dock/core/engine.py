"""Reusable Torch docking engine independent of any one scoring function."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import torch
from rdkit import Chem

from .features import compute_atom_features
from .io import ReceptorContext
from .masks import compute_intramolecular_mask
from .optimization import (
    OptimizationStats,
    OptimizerName,
    SE3PoseModel,
    optimize_pose_module,
    optimize_torsions,
)
from .prepared import PreparedLigand
from .scoring import PreparedScorer, ScoreComponents, ScorerLike, resolve_scorer


class AtomTypingWarning(UserWarning):
    """A selected scoring input has unresolved inferred atom chemistry."""


def _aggregate_optimization_stats(stats_values: list[OptimizationStats], total_poses: int) -> OptimizationStats:
    all_steps = [value.average_steps for value in stats_values for _ in range(value.num_poses)]
    return OptimizationStats(
        average_steps=float(sum(all_steps) / len(all_steps)) if all_steps else 0.0,
        minimum_steps=min((value.minimum_steps for value in stats_values), default=0),
        maximum_steps=max((value.maximum_steps for value in stats_values), default=0),
        num_poses=total_poses,
        initial_best_energy=min((value.initial_best_energy for value in stats_values), default=0.0),
        final_best_energy=min((value.final_best_energy for value in stats_values), default=0.0),
    )


def _features_to_device(features: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in features.items()}


@dataclass(frozen=True)
class PreparedDockingProblem:
    mol: Chem.Mol
    receptor: ReceptorContext
    scorer: PreparedScorer
    anchor_indices: tuple[int, ...]
    num_rotatable_bonds: float
    prepared_input_fingerprint: str | None = None


class DockingEngine:
    """Prepare scoring once and optimize anchor-constrained or guided poses."""

    def __init__(
        self,
        scorer: ScorerLike = "vina",
        *,
        device: torch.device | str | None = None,
        optimizer: OptimizerName = "adam",
        num_steps: int = 100,
        learning_rate: float = 0.05,
        batch_size: int = 128,
        early_stopping: bool = True,
        patience: int = 30,
        min_delta: float = 1e-5,
        strict_atom_typing: bool = False,
    ) -> None:
        for name, value, minimum in (("num_steps", num_steps, 0), ("batch_size", batch_size, 1), ("patience", patience, 1)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value, strict_positive in (("learning_rate", learning_rate, True), ("min_delta", min_delta, False)):
            if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                    or (value <= 0 if strict_positive else value < 0)):
                raise ValueError(f"{name} must be finite and {'positive' if strict_positive else 'nonnegative'}")
        if optimizer not in {"adam", "adamw", "lbfgs", "cartesian"}:
            raise ValueError(f"unsupported optimizer: {optimizer!r}")
        if not isinstance(strict_atom_typing, bool):
            raise ValueError("strict_atom_typing must be bool")
        self.scoring_model = resolve_scorer(scorer)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.strict_atom_typing = strict_atom_typing

    @property
    def scorer_name(self) -> str:
        return self.scoring_model.name

    @property
    def score_units(self) -> str:
        return self.scoring_model.units

    def prepare(
        self,
        mol: Chem.Mol,
        receptor: ReceptorContext,
        *,
        anchor_indices: tuple[int, ...] | list[int] = (),
        num_rotatable_bonds: int | float | None = None,
        exclude_intramolecular_atoms: set[int] | None = None,
        intermolecular_exclusion_mask: torch.Tensor | None = None,
        prepared_ligand: PreparedLigand | None = None,
    ) -> PreparedDockingProblem:
        """Prepare inferred input or explicitly mapped authoritative ligand data.

        With ``prepared_ligand``, pass ``prepared_ligand.mol``. Its types,
        charges, tree mobility and Vina rotor count override graph inference;
        use its coordinates for pose initialization. Input preparation never
        assigns missing protonation/tautomer states.
        """
        if prepared_ligand is not None:
            if not isinstance(prepared_ligand, PreparedLigand):
                raise TypeError("prepared_ligand must be PreparedLigand")
            prepared_ligand.validate_mol(mol)
            mol = prepared_ligand.mol
            features = prepared_ligand.features(self.device)
            if num_rotatable_bonds is not None and num_rotatable_bonds != prepared_ligand.vina_num_tors:
                raise ValueError("explicit rotor count conflicts with prepared Vina num_tors")
            num_rotatable_bonds = prepared_ligand.vina_num_tors
            intramolecular_mask = prepared_ligand.intramolecular_mask(self.device)
            if exclude_intramolecular_atoms:
                for index in exclude_intramolecular_atoms:
                    index = int(index)
                    if 0 <= index < mol.GetNumAtoms():
                        intramolecular_mask[index, :] = False
                        intramolecular_mask[:, index] = False
        else:
            num_rotatable_bonds = 0 if num_rotatable_bonds is None else num_rotatable_bonds
            features = compute_atom_features(mol, self.device, strict=self.strict_atom_typing)
            intramolecular_mask = compute_intramolecular_mask(
                mol, self.device, exclude_atom_indices=exclude_intramolecular_atoms,
            )
        if (isinstance(num_rotatable_bonds, bool) or not isinstance(num_rotatable_bonds, Real)
                or not math.isfinite(num_rotatable_bonds) or num_rotatable_bonds < 0):
            raise ValueError("rotor count must be finite and nonnegative")
        receptor_ambiguities = tuple(receptor.features.get("typing_ambiguities", ()))
        if self.strict_atom_typing and receptor_ambiguities:
            raise ValueError("ambiguous receptor atom typing: " + "; ".join(receptor_ambiguities))
        prepared_scorer = self.scoring_model.prepare(
            features,
            receptor.coords.to(self.device),
            _features_to_device(receptor.features, self.device),
            num_rotatable_bonds=num_rotatable_bonds,
            intramolecular_mask=intramolecular_mask,
            intermolecular_exclusion_mask=intermolecular_exclusion_mask,
        )
        diagnostics = tuple(features.get("typing_ambiguities", ())) + receptor_ambiguities
        if diagnostics:
            warnings.warn(
                f"Scoring input has {len(diagnostics)} atom-typing ambiguities: "
                + "; ".join(diagnostics[:4])
                + ". Supply prepared protonation/types or enable strict_atom_typing.",
                AtomTypingWarning,
                stacklevel=2,
            )
        return PreparedDockingProblem(
            mol=mol,
            receptor=receptor,
            scorer=prepared_scorer,
            anchor_indices=tuple(int(index) for index in anchor_indices),
            num_rotatable_bonds=num_rotatable_bonds,
            prepared_input_fingerprint=None if prepared_ligand is None else prepared_ligand.fingerprint,
        )

    def optimize_anchored(
        self,
        problem: PreparedDockingProblem,
        coords: torch.Tensor,
        *,
        freeze_anchor: bool = True,
    ) -> tuple[torch.Tensor, OptimizationStats]:
        return optimize_torsions(
            problem.mol,
            problem.anchor_indices,
            coords,
            problem.scorer,
            self.device,
            num_steps=self.num_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            early_stopping=self.early_stopping,
            patience=self.patience,
            min_delta=self.min_delta,
            freeze_anchor=freeze_anchor,
        )

    def optimize_se3(
        self,
        problem: PreparedDockingProblem,
        base_coords: torch.Tensor,
        pivot_atom_index: int,
        *,
        centers: torch.Tensor,
        rotation_vectors: torch.Tensor,
        additional_energy_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        release_steps: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, OptimizationStats, OptimizationStats]:
        """Guide SE(3)+torsion optimization around a frozen pivot atom, then release restraints.

        Both phases reuse the same live pose module, so the guide phase's rigid
        rotation state (and its gradients) carry over into the release phase
        rather than being reinitialized.
        """
        if release_steps < 0:
            raise ValueError("release_steps must be non-negative")
        base_coords = base_coords.to(self.device, dtype=torch.float32)
        centers = centers.to(self.device, dtype=torch.float32)
        rotation_vectors = rotation_vectors.to(self.device, dtype=torch.float32)
        if base_coords.ndim != 3 or centers.shape != (base_coords.shape[0], 3):
            raise ValueError("base_coords and centers must have shapes [B,N,3] and [B,3]")
        if rotation_vectors.shape != centers.shape:
            raise ValueError("rotation_vectors must match centers")

        guided_outputs: list[torch.Tensor] = []
        final_outputs: list[torch.Tensor] = []
        guide_stats_values: list[OptimizationStats] = []
        release_stats_values: list[OptimizationStats] = []
        chunk_size = 1 if self.optimizer in {"lbfgs", "cartesian"} else self.batch_size
        for start in range(0, base_coords.shape[0], chunk_size):
            stop = min(start + chunk_size, base_coords.shape[0])
            model = SE3PoseModel(
                problem.mol,
                base_coords[start:stop],
                pivot_atom_index,
                centers[start:stop],
                rotation_vectors[start:stop],
                self.device,
            )

            def guide_energy_fn(values: torch.Tensor) -> torch.Tensor:
                search = problem.scorer.search_energy(values)
                return search if additional_energy_fn is None else search + additional_energy_fn(values)

            guided, guide_stats = optimize_pose_module(
                model,
                guide_energy_fn,
                num_steps=self.num_steps,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                early_stopping=self.early_stopping,
                patience=self.patience,
                min_delta=self.min_delta,
            )
            released, release_stats = optimize_pose_module(
                model,
                problem.scorer.search_energy,
                num_steps=release_steps,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                early_stopping=self.early_stopping,
                patience=self.patience,
                min_delta=self.min_delta,
            )
            if guided.ndim == 2:
                guided = guided.unsqueeze(0)
            if released.ndim == 2:
                released = released.unsqueeze(0)
            guided_outputs.append(guided)
            final_outputs.append(released)
            guide_stats_values.append(guide_stats)
            release_stats_values.append(release_stats)

        total_poses = base_coords.shape[0]
        guide_aggregate = _aggregate_optimization_stats(guide_stats_values, total_poses)
        release_aggregate = _aggregate_optimization_stats(release_stats_values, total_poses)
        return torch.cat(guided_outputs), torch.cat(final_outputs), guide_aggregate, release_aggregate

    @staticmethod
    def report_scores(
        problem: PreparedDockingProblem,
        initial_coords: torch.Tensor,
        final_coords: torch.Tensor | None = None,
    ) -> tuple[ScoreComponents, ScoreComponents]:
        """Score initial/final poses against one common intramolecular baseline."""
        final_coords = initial_coords if final_coords is None else final_coords
        initial_raw = problem.scorer.raw_components(initial_coords)
        final_raw = problem.scorer.raw_components(final_coords)
        best_final = torch.argmin(final_raw.search_energy)
        reference = final_raw.intramolecular[best_final].detach()
        return problem.scorer.report(initial_raw, reference), problem.scorer.report(final_raw, reference)

    def metadata(self) -> dict[str, Any]:
        return {
            "scorer": self.scorer_name,
            "scorer_fingerprint": self.scoring_model.fingerprint,
            "score_units": self.score_units,
            "optimizer": self.optimizer,
            "optimization_steps": self.num_steps,
            "learning_rate": self.learning_rate,
        }

