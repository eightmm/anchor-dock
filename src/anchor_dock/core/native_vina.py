"""Explicit score-only native Vina/Vinardo radial-table convention.

This is a final-energy API for ordinary authoritative prepared ligands and a
fixed rigid receptor. It does not emulate a native docking search or native
bond perception. The factor-32 FAST table is piecewise constant, so callers
must detach coordinates deliberately. Use the continuous pair scorer during
optimization. Numeric parity was checked against Vina 1.2.7, not every native
release or every chemistry supported by Vina.
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch

from .features import AUTHORITATIVE_TYPING_VERSION, _features_from_xs
from .prepared import PREPARED_INPUT_CONTRACT
from .scoring import (
    VINA_CONFIG,
    VINARDO_CONFIG,
    PreparedPairwiseScorer,
    RawScoreComponents,
    ScoreComponents,
    _weighted_pair_energy,
    positive_energy_curl,
    prepare_interaction_matrices,
)

NATIVE_VINA_SCORE_CONTRACT = "vina-1.2.7-fast-r2-factor32-curl1000-prepared-v1"
_OFFICIAL_CONFIGS = (deepcopy(VINA_CONFIG), deepcopy(VINARDO_CONFIG))
_ORDINARY_TYPES = frozenset(("C", "A", "N", "NA", "O", "OA", "S", "SA", "P", "H", "HD", "F", "Cl", "Br", "I", "Si", "At"))


def _validate_features(features: dict[str, object], count: int, label: str) -> None:
    if (features.get("typing_version") != AUTHORITATIVE_TYPING_VERSION
            or features.get("prepared_input_contract") != PREPARED_INPUT_CONTRACT):
        raise ValueError(f"native final scoring requires authoritative prepared {label} types; use PreparedLigand/PDBQT mapping")
    kinds = features.get("ad4_types", ())
    if (not isinstance(kinds, (tuple, list)) or len(kinds) != count
            or any(not isinstance(kind, str) or kind not in _ORDINARY_TYPES for kind in kinds)):
        raise ValueError(f"native final scoring supports ordinary prepared {label} types; metals and extended/dummy types are outside this API")
    if features.get("typing_ambiguities"):
        raise ValueError(f"native final scoring requires unambiguous prepared {label} types")
    xs = features.get("xs_types")
    if (not isinstance(xs, (tuple, list)) or len(xs) != count
            or any(not isinstance(kind, str) or kind in {"X", "Met_D"} for kind in xs)):
        raise ValueError(f"native final scoring requires ordinary prepared {label} XS roles")
    try:
        expected = _features_from_xs(xs, "cpu", version=AUTHORITATIVE_TYPING_VERSION, source="validation")
    except KeyError as error:
        raise ValueError(f"unsupported prepared {label} XS type") from error
    for name in ("radius_vina", "radius_vinardo", "hydrophobic", "donor", "acceptor", "active"):
        tensor = features.get(name)
        expected_dtypes = (torch.bool,) if name == "active" else (torch.float32, torch.float64)
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != (count,) or tensor.dtype not in expected_dtypes
                or not torch.isfinite(tensor).all()):
            raise ValueError(f"invalid prepared {label} feature {name}")
        if tensor.requires_grad:
            raise ValueError("native final scoring is nondifferentiable; detach feature tensors explicitly")
        reference = expected[name].to(device=tensor.device, dtype=tensor.dtype)
        if (not torch.equal(tensor, reference) if name not in {"radius_vina", "radius_vinardo"}
                else not torch.allclose(tensor, reference, atol=1e-7, rtol=0.0)):
            raise ValueError(f"modified prepared {label} feature {name} is outside native scoring")


def _validated_coordinates(scorer: PreparedPairwiseScorer, coordinates: torch.Tensor) -> torch.Tensor:
    if not isinstance(scorer, PreparedPairwiseScorer) or scorer.config not in _OFFICIAL_CONFIGS:
        raise ValueError("native final scoring requires the official ordinary Vina or Vinardo pair configuration")
    if not isinstance(coordinates, torch.Tensor) or coordinates.dtype not in (torch.float32, torch.float64):
        raise ValueError("coordinates must be a float32/float64 tensor")
    if coordinates.requires_grad:
        raise ValueError("native final scoring is nondifferentiable; call coordinates.detach() explicitly")
    if coordinates.ndim == 2:
        coordinates = coordinates.unsqueeze(0)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or min(coordinates.shape[:2]) < 1:
        raise ValueError("coordinates must be nonempty [B,N,3] or [N,3]")
    if not torch.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    receptor = scorer.receptor_coords
    if (not isinstance(receptor, torch.Tensor) or receptor.dtype not in (torch.float32, torch.float64)
            or receptor.ndim != 2 or receptor.shape[1] != 3
            or receptor.shape[0] < 1 or not torch.isfinite(receptor).all()):
        raise ValueError("prepared rigid receptor coordinates must be finite nonempty [M,3]")
    if receptor.requires_grad:
        raise ValueError("native final scoring is nondifferentiable and requires a fixed detached receptor")
    _validate_features(scorer.ligand_features, coordinates.shape[1], "ligand")
    _validate_features(scorer.receptor_features, receptor.shape[0], "receptor")
    if scorer.intermolecular_exclusion_mask is not None:
        raise ValueError("native rigid-receptor final scoring does not support custom intermolecular exclusions")
    count = scorer.ligand_features.get("vina_num_tors")
    if (isinstance(count, bool) or not isinstance(count, (int, float)) or not math.isfinite(count)
            or count < 0 or 2 * count != round(2 * count)):
        raise ValueError("prepared ligand must supply its native half-integral vina_num_tors")
    if scorer.num_rotatable_bonds != count:
        raise ValueError("prepared scorer rotor count disagrees with authoritative vina_num_tors")
    return coordinates


def _prepared_mask(scorer: PreparedPairwiseScorer, coordinates: torch.Tensor) -> torch.Tensor:
    pairs = scorer.ligand_features.get("prepared_intramolecular_pairs")
    if not isinstance(pairs, (tuple, list)):
        raise ValueError("native final scoring requires authoritative prepared intramolecular pairs")
    size = coordinates.shape[1]
    expected = torch.zeros((size, size), dtype=torch.bool, device=coordinates.device)
    for pair in pairs:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(isinstance(index, bool) or not isinstance(index, int) for index in pair)
                or not 0 <= pair[0] < pair[1] < size):
            raise ValueError("invalid authoritative intramolecular pair")
        expected[pair[0], pair[1]] = expected[pair[1], pair[0]] = True
    mask = scorer.intramolecular_mask
    if mask is None:
        if pairs:
            raise ValueError("prepared scorer is missing its authoritative intramolecular mask")
    elif (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != expected.shape
          or not torch.equal(mask.to(device=coordinates.device), expected)):
        raise ValueError("prepared scorer mask disagrees with authoritative intramolecular pairs")
    return expected


def native_vina_components(scorer: PreparedPairwiseScorer, coordinates: torch.Tensor) -> RawScoreComponents:
    """Return native-convention final inter/intra/search energies, without grad.

    Requires prepared types and the prepared rotor/mobility contract for both
    scorer inputs. Custom exclusions, modified force fields, metals, macrocycle
    and hydrated-ligand dummy types are rejected. This direct rigid-receptor
    calculation has no search-grid boundary or interpolation convention.

    The native authentic positive-energy cap is always 1000, independently of
    the continuous scorer's chosen search cap. Coordinates are in angstroms.
    """
    coordinates = _validated_coordinates(scorer, coordinates)
    mask = _prepared_mask(scorer, coordinates)

    def table(distances, matrices):
        # Only r<8 is queried by native final scoring; clamp solely the unused
        # table index to keep far-away finite inputs from overflowing r^2.
        index = torch.floor(32.0 * distances.square().clamp(max=scorer.config.cutoff**2))
        lower, upper = torch.sqrt(index / 32.0), torch.sqrt((index + 1.0) / 32.0)
        energy = 0.5 * (_weighted_pair_energy(lower, matrices, scorer.config)
                        + _weighted_pair_energy(upper, matrices, scorer.config))
        return energy.masked_fill(distances >= scorer.config.cutoff, 0.0)

    with torch.no_grad():
        # Rebuild these small coordinate-independent attributes from validated
        # features. Mutated/stale cached matrices must not bypass the explicit
        # official-force-field validation of this final-score API.
        inter_matrices = prepare_interaction_matrices(
            scorer.ligand_features, scorer.receptor_features, scorer.config, coordinates.device)
        intra_matrices = prepare_interaction_matrices(
            scorer.ligand_features, scorer.ligand_features, scorer.config, coordinates.device)
        receptor = scorer.receptor_coords.to(device=coordinates.device, dtype=coordinates.dtype)
        distances = torch.cdist(coordinates, receptor.unsqueeze(0).expand(coordinates.shape[0], -1, -1))
        inter = positive_energy_curl(table(distances, inter_matrices).sum(-1), 1000.0).sum(-1)
        distances = torch.cdist(coordinates, coordinates)
        intra = positive_energy_curl(table(distances, intra_matrices), 1000.0)
        intra = (intra * mask).sum(dim=(1, 2)) * 0.5
        return RawScoreComponents(inter, intra, inter + intra)


def native_vina_score(scorer: PreparedPairwiseScorer, coordinates: torch.Tensor, *,
                      intramolecular_reference: torch.Tensor | float | None = None) -> ScoreComponents:
    """Report native-convention energies with an explicit shared intra reference.

    ``None`` uses the intra energy of the lowest-search-energy pose in this
    batch, matching AnchorDock's common docking reference. A scalar supplies a
    reference from another final pose; zero exposes the full intra component.
    Native Vina's standalone ``score()`` uses each current pose's own reference,
    so its multi-pose use differs from this shared-reference docking convention.
    """
    raw = native_vina_components(scorer, coordinates)
    if intramolecular_reference is not None:
        if isinstance(intramolecular_reference, bool):
            raise ValueError("intramolecular_reference must be one finite detached scalar")
        reference = torch.as_tensor(intramolecular_reference, device=raw.search_energy.device,
                                    dtype=raw.search_energy.dtype)
        if reference.requires_grad or reference.ndim != 0 or not torch.isfinite(reference):
            raise ValueError("intramolecular_reference must be one finite detached scalar")
        intramolecular_reference = reference
    with torch.no_grad():
        return scorer.report(raw, intramolecular_reference)
