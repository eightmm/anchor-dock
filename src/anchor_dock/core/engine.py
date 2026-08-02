"""Reusable Torch docking engine independent of any one scoring function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from rdkit import Chem

from .features import compute_atom_features
from .io import ReceptorContext
from .masks import compute_intramolecular_mask
from .optimization import FreePoseModel, OptimizationStats, OptimizerName, optimize_pose_module, optimize_torsions
from .scoring import PreparedScorer, ScorerLike, ScoreComponents, resolve_scorer


def _features_to_device(features: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }


@dataclass(frozen=True)
class PreparedDockingProblem:
    mol: Chem.Mol
    receptor: ReceptorContext
    scorer: PreparedScorer
    anchor_indices: tuple[int, ...]
    num_rotatable_bonds: int


class DockingEngine:
    """Prepare scoring once and optimize anchor-constrained or free poses."""

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
    ) -> None:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if patience <= 0:
            raise ValueError("patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
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
        num_rotatable_bonds: int = 0,
        exclude_intramolecular_atoms: set[int] | None = None,
        intermolecular_exclusion_mask: torch.Tensor | None = None,
    ) -> PreparedDockingProblem:
        features = compute_atom_features(mol, self.device)
        intramolecular_mask = compute_intramolecular_mask(
            mol,
            self.device,
            exclude_atom_indices=exclude_intramolecular_atoms,
        )
        prepared_scorer = self.scoring_model.prepare(
            features,
            receptor.coords.to(self.device),
            _features_to_device(receptor.features, self.device),
            num_rotatable_bonds=num_rotatable_bonds,
            intramolecular_mask=intramolecular_mask,
            intermolecular_exclusion_mask=intermolecular_exclusion_mask,
        )
        return PreparedDockingProblem(
            mol=mol,
            receptor=receptor,
            scorer=prepared_scorer,
            anchor_indices=tuple(int(index) for index in anchor_indices),
            num_rotatable_bonds=int(num_rotatable_bonds),
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

    def optimize_free(
        self,
        problem: PreparedDockingProblem,
        base_coords: torch.Tensor,
        *,
        centers: torch.Tensor,
        rotation_vectors: torch.Tensor,
        box_center: torch.Tensor,
        box_size: torch.Tensor,
        boundary_weight: float = 10.0,
    ) -> tuple[torch.Tensor, OptimizationStats]:
        """Optimize rigid SE(3) and torsions in memory-bounded chunks."""
        base_coords = base_coords.to(self.device, dtype=torch.float32)
        centers = centers.to(self.device, dtype=torch.float32)
        rotation_vectors = rotation_vectors.to(self.device, dtype=torch.float32)
        if base_coords.ndim != 3 or centers.shape != (base_coords.shape[0], 3):
            raise ValueError("base_coords and centers must have shapes [B,N,3] and [B,3]")
        if rotation_vectors.shape != centers.shape:
            raise ValueError("rotation_vectors must match centers")
        box_center = box_center.to(self.device, dtype=torch.float32)
        half_size = box_size.to(self.device, dtype=torch.float32) * 0.5

        outputs: list[torch.Tensor] = []
        stats_values: list[OptimizationStats] = []
        chunk_size = 1 if self.optimizer == "lbfgs" else self.batch_size
        for start in range(0, base_coords.shape[0], chunk_size):
            stop = min(start + chunk_size, base_coords.shape[0])
            model = FreePoseModel(
                problem.mol,
                base_coords[start:stop],
                centers[start:stop],
                rotation_vectors[start:stop],
                self.device,
            )

            def energy_fn(values: torch.Tensor) -> torch.Tensor:
                search = problem.scorer.search_energy(values)
                excess = torch.relu(torch.abs(values - box_center) - half_size)
                boundary = excess.square().sum(dim=(1, 2)) * boundary_weight
                return search + boundary

            optimized, stats = optimize_pose_module(
                model,
                energy_fn,
                num_steps=self.num_steps,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                early_stopping=self.early_stopping,
                patience=self.patience,
                min_delta=self.min_delta,
            )
            if optimized.ndim == 2:
                optimized = optimized.unsqueeze(0)
            outputs.append(optimized)
            stats_values.append(stats)

        all_steps = [value.average_steps for value in stats_values for _ in range(value.num_poses)]
        aggregate = OptimizationStats(
            average_steps=float(sum(all_steps) / len(all_steps)) if all_steps else 0.0,
            minimum_steps=min((value.minimum_steps for value in stats_values), default=0),
            maximum_steps=max((value.maximum_steps for value in stats_values), default=0),
            num_poses=base_coords.shape[0],
            initial_best_energy=min((value.initial_best_energy for value in stats_values), default=0.0),
            final_best_energy=min((value.final_best_energy for value in stats_values), default=0.0),
        )
        return torch.cat(outputs), aggregate

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
            "score_units": self.score_units,
            "optimizer": self.optimizer,
            "optimization_steps": self.num_steps,
            "learning_rate": self.learning_rate,
        }
