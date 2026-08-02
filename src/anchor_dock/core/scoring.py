"""Differentiable, scorer-independent pairwise docking energies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from .masks import normalize_pair_mask


@dataclass(frozen=True)
class ScoringConfig:
    name: str
    units: str
    radius_key: str
    cutoff: float
    formula: str
    weights: dict[str, float]
    torsion_weight: float = 0.0
    gauss1_width: float = 0.5
    gauss2_offset: float | None = 3.0
    gauss2_width: float = 2.0
    hydrophobic_good: float = 0.5
    hydrophobic_bad: float = 1.5
    hbond_good: float = -0.7
    hbond_bad: float = 0.0


VINA_CONFIG = ScoringConfig(
    name="vina",
    units="kcal/mol-like",
    radius_key="radius_vina",
    cutoff=8.0,
    formula="vina",
    weights={
        "gauss1": -0.035579,
        "gauss2": -0.005156,
        "repulsion": 0.840245,
        "hydrophobic": -0.035069,
        "hbond": -0.587439,
    },
    torsion_weight=0.05846,
)

VINARDO_CONFIG = ScoringConfig(
    name="vinardo",
    units="kcal/mol-like",
    radius_key="radius_vinardo",
    cutoff=8.0,
    formula="vina",
    weights={
        "gauss1": -0.045,
        "gauss2": 0.0,
        "repulsion": 0.8,
        "hydrophobic": -0.035,
        "hbond": -0.600,
    },
    torsion_weight=0.05846,
    gauss1_width=0.8,
    gauss2_offset=None,
    hydrophobic_good=0.0,
    hydrophobic_bad=2.5,
    hbond_good=-0.6,
)

SOFTDOCK_CONFIG = ScoringConfig(
    name="softdock",
    units="arbitrary",
    radius_key="radius_vina",
    cutoff=10.0,
    formula="softdock",
    weights={
        "repulsion": 1.0,
        "contact": -0.18,
        "hydrophobic": -0.32,
        "hbond": -0.75,
    },
)

SCORING_CONFIGS = {
    "vina": VINA_CONFIG,
    "vinardo": VINARDO_CONFIG,
    "softdock": SOFTDOCK_CONFIG,
}


@dataclass(frozen=True)
class RawScoreComponents:
    intermolecular: torch.Tensor
    intramolecular: torch.Tensor
    search_energy: torch.Tensor


@dataclass(frozen=True)
class ScoreComponents:
    intermolecular: torch.Tensor
    intramolecular: torch.Tensor
    search_energy: torch.Tensor
    score: torch.Tensor
    intramolecular_reference: torch.Tensor


def _tensor_features(features: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in features.items() if isinstance(value, torch.Tensor)}


def prepare_interaction_matrices(
    ligand_features: dict[str, object],
    receptor_features: dict[str, object],
    config: ScoringConfig,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Precompute coordinate-independent ligand-receptor pair attributes."""
    device = torch.device(device)
    ligand = _tensor_features(ligand_features, device)
    receptor = _tensor_features(receptor_features, device)
    radius_sum = ligand[config.radius_key][:, None] + receptor[config.radius_key][None, :]
    hydrophobic = ligand["hydrophobic"][:, None] * receptor["hydrophobic"][None, :]
    hbond = (
        ligand["donor"][:, None] * receptor["acceptor"][None, :]
        + ligand["acceptor"][:, None] * receptor["donor"][None, :]
        > 0
    ).to(torch.float32)
    active = ligand["active"][:, None] & receptor["active"][None, :]
    return {"radius_sum": radius_sum, "hydrophobic": hydrophobic, "hbond": hbond, "active": active}


def _slope_step(bad: float, good: float, value: torch.Tensor) -> torch.Tensor:
    if bad == good:
        return (value <= good).to(value.dtype)
    low = min(bad, good)
    high = max(bad, good)
    fraction = (value - bad) / (good - bad)
    return torch.where(value <= low, torch.tensor(float(good == low), device=value.device, dtype=value.dtype),
                       torch.where(value >= high, torch.tensor(float(good == high), device=value.device, dtype=value.dtype), fraction))


def pair_terms(
    distances: torch.Tensor,
    radius_sum: torch.Tensor,
    hydrophobic_match: torch.Tensor,
    hbond_match: torch.Tensor,
    config: ScoringConfig,
) -> dict[str, torch.Tensor]:
    """Return unweighted pair terms for one scorer configuration."""
    surface_distance = distances - radius_sum
    if config.formula == "softdock":
        overlap = torch.nn.functional.softplus(-surface_distance / 0.20) * 0.20
        return {
            "repulsion": overlap.square(),
            "contact": torch.exp(-((surface_distance - 0.6) / 1.6).square()),
            "hydrophobic": hydrophobic_match * torch.sigmoid((1.5 - surface_distance) / 0.35),
            "hbond": hbond_match * torch.exp(-((surface_distance + 0.25) / 0.65).square()),
        }

    gauss1 = torch.exp(-((surface_distance / config.gauss1_width) ** 2))
    if config.gauss2_offset is None:
        gauss2 = torch.zeros_like(surface_distance)
    else:
        gauss2 = torch.exp(-(((surface_distance - config.gauss2_offset) / config.gauss2_width) ** 2))
    repulsion = torch.where(surface_distance < 0.0, surface_distance.square(), torch.zeros_like(surface_distance))
    hydrophobic = hydrophobic_match * _slope_step(
        config.hydrophobic_bad, config.hydrophobic_good, surface_distance
    )
    hbond = hbond_match * _slope_step(config.hbond_bad, config.hbond_good, surface_distance)
    return {
        "gauss1": gauss1,
        "gauss2": gauss2,
        "repulsion": repulsion,
        "hydrophobic": hydrophobic,
        "hbond": hbond,
    }


def _weighted_pair_energy(
    distances: torch.Tensor,
    matrices: dict[str, torch.Tensor],
    config: ScoringConfig,
) -> torch.Tensor:
    terms = pair_terms(
        distances,
        matrices["radius_sum"].to(device=distances.device, dtype=distances.dtype),
        matrices["hydrophobic"].to(device=distances.device, dtype=distances.dtype),
        matrices["hbond"].to(device=distances.device, dtype=distances.dtype),
        config,
    )
    energy = torch.zeros_like(distances)
    for name, weight in config.weights.items():
        energy = energy + weight * terms[name]
    active = matrices["active"].to(device=distances.device)
    cutoff = distances < config.cutoff
    return energy.masked_fill(~(active & cutoff), 0.0)


@runtime_checkable
class PreparedScorer(Protocol):
    name: str
    units: str

    def search_energy(self, ligand_coords: torch.Tensor) -> torch.Tensor: ...

    def raw_components(self, ligand_coords: torch.Tensor) -> RawScoreComponents: ...

    def report(
        self,
        raw: RawScoreComponents,
        intramolecular_reference: torch.Tensor | float | None = None,
    ) -> ScoreComponents: ...


@dataclass
class PreparedPairwiseScorer:
    config: ScoringConfig
    receptor_coords: torch.Tensor
    ligand_features: dict[str, object]
    receptor_features: dict[str, object]
    num_rotatable_bonds: int
    intramolecular_mask: torch.Tensor | None
    intermolecular_exclusion_mask: torch.Tensor | None
    intermolecular_matrices: dict[str, torch.Tensor]
    intramolecular_matrices: dict[str, torch.Tensor]

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def units(self) -> str:
        return self.config.units

    def raw_components(self, ligand_coords: torch.Tensor) -> RawScoreComponents:
        if ligand_coords.ndim == 2:
            ligand_coords = ligand_coords.unsqueeze(0)
        if ligand_coords.ndim != 3 or ligand_coords.shape[-1] != 3:
            raise ValueError("ligand_coords must have shape [B,N,3] or [N,3]")
        batch_size, num_atoms, _ = ligand_coords.shape
        receptor = self.receptor_coords.to(device=ligand_coords.device, dtype=ligand_coords.dtype)
        receptor_batch = receptor.unsqueeze(0).expand(batch_size, -1, -1)
        inter_distances = torch.cdist(ligand_coords, receptor_batch)
        inter_energy = _weighted_pair_energy(inter_distances, self.intermolecular_matrices, self.config)
        exclusion = normalize_pair_mask(
            self.intermolecular_exclusion_mask,
            batch_size,
            num_atoms,
            receptor.shape[0],
            device=ligand_coords.device,
        )
        if exclusion is not None:
            inter_energy = inter_energy.masked_fill(exclusion, 0.0)
        intermolecular = inter_energy.sum(dim=(1, 2))

        if self.intramolecular_mask is None:
            intramolecular = torch.zeros_like(intermolecular)
        else:
            intra_distances = torch.cdist(ligand_coords, ligand_coords)
            intra_energy = _weighted_pair_energy(intra_distances, self.intramolecular_matrices, self.config)
            intra_mask = normalize_pair_mask(
                self.intramolecular_mask,
                batch_size,
                num_atoms,
                num_atoms,
                device=ligand_coords.device,
            )
            intramolecular = (intra_energy * intra_mask).sum(dim=(1, 2)) * 0.5
        search_energy = intermolecular + intramolecular
        return RawScoreComponents(intermolecular, intramolecular, search_energy)

    def search_energy(self, ligand_coords: torch.Tensor) -> torch.Tensor:
        return self.raw_components(ligand_coords).search_energy

    def report(
        self,
        raw: RawScoreComponents,
        intramolecular_reference: torch.Tensor | float | None = None,
    ) -> ScoreComponents:
        if self.config.formula == "softdock":
            reference = torch.zeros((), dtype=raw.search_energy.dtype, device=raw.search_energy.device)
            score = raw.search_energy
        else:
            if intramolecular_reference is None:
                best = torch.argmin(raw.search_energy)
                reference = raw.intramolecular[best].detach()
            else:
                reference = torch.as_tensor(
                    intramolecular_reference,
                    dtype=raw.search_energy.dtype,
                    device=raw.search_energy.device,
                )
            denominator = 1.0 + self.config.torsion_weight * self.num_rotatable_bonds
            score = (raw.search_energy - reference) / denominator
        return ScoreComponents(
            raw.intermolecular,
            raw.intramolecular,
            raw.search_energy,
            score,
            reference,
        )

    def score_components(
        self,
        ligand_coords: torch.Tensor,
        intramolecular_reference: torch.Tensor | float | None = None,
    ) -> ScoreComponents:
        return self.report(self.raw_components(ligand_coords), intramolecular_reference)


class PairwiseScorer:
    """Factory for prepared Vina, Vinardo and SoftDock scorers."""

    def __init__(self, config: ScoringConfig | str):
        if isinstance(config, str):
            try:
                config = SCORING_CONFIGS[config.lower()]
            except KeyError as exc:
                raise ValueError(f"unknown scorer {config!r}; choose from {sorted(SCORING_CONFIGS)}") from exc
        self.config = config
        self.name = config.name
        self.units = config.units

    def prepare(
        self,
        ligand_features: dict[str, object],
        receptor_coords: torch.Tensor,
        receptor_features: dict[str, object],
        *,
        num_rotatable_bonds: int = 0,
        intramolecular_mask: torch.Tensor | None = None,
        intermolecular_exclusion_mask: torch.Tensor | None = None,
    ) -> PreparedPairwiseScorer:
        device = receptor_coords.device
        inter = prepare_interaction_matrices(ligand_features, receptor_features, self.config, device)
        intra = prepare_interaction_matrices(ligand_features, ligand_features, self.config, device)
        return PreparedPairwiseScorer(
            config=self.config,
            receptor_coords=receptor_coords,
            ligand_features=ligand_features,
            receptor_features=receptor_features,
            num_rotatable_bonds=max(0, int(num_rotatable_bonds)),
            intramolecular_mask=intramolecular_mask,
            intermolecular_exclusion_mask=intermolecular_exclusion_mask,
            intermolecular_matrices=inter,
            intramolecular_matrices=intra,
        )


class NeuralScorerAdapter:
    """Adapter for a differentiable ``nn.Module`` custom scoring model.

    The wrapped module receives ``(ligand_coords, receptor_coords,
    ligand_features, receptor_features)`` and must return one scalar per pose.
    """

    def __init__(self, module: nn.Module, *, name: str = "neural", units: str = "arbitrary"):
        self.module = module
        self.name = name
        self.units = units

    def prepare(
        self,
        ligand_features: dict[str, object],
        receptor_coords: torch.Tensor,
        receptor_features: dict[str, object],
        **_: object,
    ) -> PreparedNeuralScorer:
        return PreparedNeuralScorer(self.module, self.name, self.units, receptor_coords, ligand_features, receptor_features)


@dataclass
class PreparedNeuralScorer:
    module: nn.Module
    name: str
    units: str
    receptor_coords: torch.Tensor
    ligand_features: dict[str, object]
    receptor_features: dict[str, object]

    def search_energy(self, ligand_coords: torch.Tensor) -> torch.Tensor:
        if ligand_coords.ndim == 2:
            ligand_coords = ligand_coords.unsqueeze(0)
        values = self.module(ligand_coords, self.receptor_coords, self.ligand_features, self.receptor_features)
        if values.ndim != 1 or values.shape[0] != ligand_coords.shape[0]:
            raise ValueError("custom scorer must return shape [B]")
        return values

    def raw_components(self, ligand_coords: torch.Tensor) -> RawScoreComponents:
        values = self.search_energy(ligand_coords)
        zeros = torch.zeros_like(values)
        return RawScoreComponents(values, zeros, values)

    def report(
        self,
        raw: RawScoreComponents,
        intramolecular_reference: torch.Tensor | float | None = None,
    ) -> ScoreComponents:
        del intramolecular_reference
        zero = torch.zeros((), device=raw.search_energy.device, dtype=raw.search_energy.dtype)
        return ScoreComponents(raw.intermolecular, raw.intramolecular, raw.search_energy, raw.search_energy, zero)

    def score_components(
        self,
        ligand_coords: torch.Tensor,
        intramolecular_reference: torch.Tensor | float | None = None,
    ) -> ScoreComponents:
        return self.report(self.raw_components(ligand_coords), intramolecular_reference)


ScorerLike = str | PairwiseScorer | NeuralScorerAdapter | nn.Module


def resolve_scorer(scorer: ScorerLike) -> PairwiseScorer | NeuralScorerAdapter:
    if isinstance(scorer, str):
        try:
            return PairwiseScorer(SCORING_CONFIGS[scorer.lower()])
        except KeyError as exc:
            raise ValueError(f"unknown scorer {scorer!r}; choose from {sorted(SCORING_CONFIGS)}") from exc
    if isinstance(scorer, nn.Module):
        return NeuralScorerAdapter(scorer)
    if hasattr(scorer, "prepare"):
        return scorer
    raise TypeError("scorer must be a scorer name, nn.Module, or object with prepare()")
