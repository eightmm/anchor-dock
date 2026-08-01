"""Differentiable Vina/Vinardo-style scoring shared by AnchorDock modes."""

from __future__ import annotations

import torch

from .masks import normalize_pair_mask

VINA_WEIGHTS: dict[str, dict[str, float]] = {
    "vina": {
        "gauss1": -0.035579,
        "gauss2": -0.005156,
        "repulsion": 0.840245,
        "hydrophobic": -0.035069,
        "hbond": -0.587439,
        "rot": 0.05846,
    },
    "vina_lp": {
        "gauss1": 0.003372,
        "gauss2": -0.008098,
        "repulsion": 0.014212,
        "hydrophobic": -0.008361,
        "hbond": -0.227928,
        "rot": 0.05846,
    },
    "vinardo": {
        "gauss1": -0.0356,
        "gauss2": 0.0,
        "repulsion": 0.840,
        "hydrophobic": -0.0351,
        "hbond": -0.587,
        "rot": 0.05846,
    },
}


def precompute_interaction_matrices(
    query_features: dict[str, torch.Tensor],
    pocket_features: dict[str, torch.Tensor],
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Precompute atom-type matrices that do not depend on coordinates."""
    if device is None:
        device = query_features["vdw"].device
    device = torch.device(device)
    q = {key: value.to(device) for key, value in query_features.items()}
    p = {key: value.to(device) for key, value in pocket_features.items()}

    hydrophobic = q["hydro"][:, None] * p["hydro"][None, :]
    hbond = ((q["hbd"][:, None] * p["hba"][None, :]) +
             (q["hba"][:, None] * p["hbd"][None, :]) > 0).float()
    radius_sum = q["vdw"][:, None] + p["vdw"][None, :]
    return {"is_hydrophobic": hydrophobic, "is_hbond": hbond, "R_ij": radius_sum}


def _pair_terms(
    delta: torch.Tensor,
    hydrophobic_match: torch.Tensor,
    hbond_match: torch.Tensor,
    preset: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if preset == "vinardo":
        gauss1 = torch.exp(-((delta / 0.8) ** 2))
        gauss2 = torch.zeros_like(delta)
        hydrophobic = hydrophobic_match * torch.where(
            delta <= 0.0,
            torch.ones_like(delta),
            torch.where(delta < 2.5, 1.0 - delta / 2.5, torch.zeros_like(delta)),
        )
        hbond = hbond_match * torch.where(
            delta <= -0.6,
            torch.ones_like(delta),
            torch.where(delta < 0.0, -delta / 0.6, torch.zeros_like(delta)),
        )
    else:
        gauss1 = torch.exp(-((delta / 0.5) ** 2))
        gauss2 = torch.exp(-(((delta - 3.0) / 2.0) ** 2))
        hydrophobic = hydrophobic_match * torch.where(
            delta <= 0.5,
            torch.ones_like(delta),
            torch.where(delta < 1.5, 1.5 - delta, torch.zeros_like(delta)),
        )
        hbond = hbond_match * torch.where(
            delta <= -0.7,
            torch.ones_like(delta),
            torch.where(delta < 0.0, -delta / 0.7, torch.zeros_like(delta)),
        )
    repulsion = torch.where(delta < 0.0, delta.square(), torch.zeros_like(delta))
    return gauss1, gauss2, repulsion, hydrophobic, hbond


def vina_scoring(
    aligned_query_coords: torch.Tensor,
    pocket_coords: torch.Tensor,
    query_features: dict[str, torch.Tensor],
    pocket_features: dict[str, torch.Tensor],
    num_rotatable_bonds: int | None = None,
    weight_preset: str = "vina",
    intramolecular_mask: torch.Tensor | None = None,
    precomputed_matrices: dict[str, torch.Tensor] | None = None,
    intermolecular_exclusion_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score one or more poses.

    The returned value is a Vina-like non-bonded pose score. For covalent mode,
    interactions explicitly masked around the formed bond are not included; it
    is therefore not a reaction free energy.
    """
    if weight_preset not in VINA_WEIGHTS:
        raise ValueError(f"Unknown weight preset: {weight_preset}")
    if aligned_query_coords.ndim == 2:
        aligned_query_coords = aligned_query_coords.unsqueeze(0)
    if aligned_query_coords.ndim != 3 or aligned_query_coords.shape[-1] != 3:
        raise ValueError("aligned_query_coords must have shape [B, N, 3] or [N, 3]")

    device = aligned_query_coords.device
    dtype = aligned_query_coords.dtype
    pocket_coords = pocket_coords.to(device=device, dtype=dtype)
    batch_size, num_query_atoms, _ = aligned_query_coords.shape
    num_pocket_atoms = pocket_coords.shape[0]

    if precomputed_matrices is None:
        precomputed_matrices = precompute_interaction_matrices(query_features, pocket_features, device)
    radius_sum = precomputed_matrices["R_ij"].to(device=device, dtype=dtype)
    hydrophobic_match = precomputed_matrices["is_hydrophobic"].to(device=device, dtype=dtype)
    hbond_match = precomputed_matrices["is_hbond"].to(device=device, dtype=dtype)

    distances = torch.cdist(aligned_query_coords, pocket_coords.unsqueeze(0).expand(batch_size, -1, -1))
    delta = distances - radius_sum.unsqueeze(0)
    terms = _pair_terms(delta, hydrophobic_match.unsqueeze(0), hbond_match.unsqueeze(0), weight_preset)
    weights = VINA_WEIGHTS[weight_preset]
    energy = (
        weights["gauss1"] * terms[0]
        + weights["gauss2"] * terms[1]
        + weights["repulsion"] * terms[2]
        + weights["hydrophobic"] * terms[3]
        + weights["hbond"] * terms[4]
    )

    exclusion = normalize_pair_mask(
        intermolecular_exclusion_mask,
        batch_size,
        num_query_atoms,
        num_pocket_atoms,
        device=device,
    )
    if exclusion is not None:
        energy = energy.masked_fill(exclusion, 0.0)
    total = energy.sum(dim=(1, 2))

    intra = normalize_pair_mask(
        intramolecular_mask,
        batch_size,
        num_query_atoms,
        num_query_atoms,
        device=device,
    )
    if intra is not None:
        intra_dist = torch.cdist(aligned_query_coords, aligned_query_coords)
        q_vdw = query_features["vdw"].to(device=device, dtype=dtype)
        intra_delta = intra_dist - (q_vdw[:, None] + q_vdw[None, :]).unsqueeze(0)
        zero_match = torch.zeros_like(intra_delta)
        intra_terms = _pair_terms(intra_delta, zero_match, zero_match, weight_preset)
        intra_energy = (
            weights["gauss1"] * intra_terms[0]
            + weights["gauss2"] * intra_terms[1]
            + weights["repulsion"] * intra_terms[2]
        )
        total = total + (intra_energy * intra).sum(dim=(1, 2)) / 2.0

    if num_rotatable_bonds is not None:
        total = total / (1.0 + weights["rot"] * int(num_rotatable_bonds))
    return total
