"""Continuous DKOES preset equations for ordinary Smina/GNINA atom roles.

This independent Torch implementation uses published parameter/formula facts
from the pinned GNINA source. It does not vendor native implementation code.
It follows AnchorDock's shared intramolecular-reference reporting contract;
native atom preparation, radial tables, grid interpolation and search updates
remain separate. It is not an executed native-Smina parity claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Real

import torch

from .ad4 import _AD4_TYPES
from .features import _features_from_xs
from .masks import normalize_pair_mask
from .scoring import (
    VINA_CONFIG,
    RawScoreComponents,
    ScoreComponents,
    _slope_step,
    positive_energy_curl,
    prepare_interaction_matrices,
)

SMINA_SOURCE = {
    "repository": "https://github.com/gnina/gnina",
    "commit": "01cae6aa6e014e0bf11e511f1fe7a61b17975855",
    "files": ("gninasrc/lib/builtinscoring.cpp", "gninasrc/lib/everything.h", "gninasrc/lib/atom_constants.h"),
    "contract": "continuous-dkoes-xs-v1-shared-intra-reference-v1",
    "native_execution_verified": False,
}


@dataclass(frozen=True)
class SminaPreset:
    vdw_weight: float
    hbond_weight: float
    torsion_square_weight: float
    constant: float
    solvation_weight: float = 0.
    solvation_sigma: float = 3.6
    charge_solvation: float = .01097
    cutoff: float = 8.
    repulsion_cap: float = 100.


# Positive fast/old H-bond coefficients are the actual preset facts, not
# typographical errors to be replaced by a hand-chosen attractive term.
SMINA_PRESETS = {
    "dkoes_fast": SminaPreset(.008962, .387739, .285035, -2.467357),
    "dkoes_scoring_old": SminaPreset(.010607, .197201, .285035, -2.585651),
    "dkoes_scoring": SminaPreset(.009900, -.153055, .317267, -2.469020, .048934),
}
_SUPPORTED_XS = frozenset({"H", "C_H", "C_P", "N_P", "N_D", "N_A", "N_DA", "O_P", "O_D", "O_A", "O_DA",
                           "S_P", "P_P", "F_H", "Cl_H", "Br_H", "I_H", "Met_D"})
_SOLVATION_TYPES = frozenset({"C", "A", "N", "NA", "O", "OA", "S", "SA", "P", "H", "HD", "F", "Cl", "Br", "I",
                              "Mg", "Mn", "Zn", "Ca", "Fe"})


def smina_pair_terms(distances: torch.Tensor, radius_sum: torch.Tensor, hbond_match: torch.Tensor,
                     *, cap: float = 100., solvation_amplitude: torch.Tensor | None = None,
                     sigma: float = 3.6) -> dict[str, torch.Tensor]:
    """Unweighted continuous 4-8 well, Vina H-bond ramp and DKOES solvation.

    Native 4-8 LJ uses unit well depth at the XS radius sum, not AD4 epsilon.
    Saturation before powers prevents NaN backward at coincident atoms. The
    desolvation distance factor is exp(-(r/(2*sigma))**2), not the different
    standard AD4 Gaussian. Cutoff and atom masks are applied by the scorer.
    """
    if not math.isfinite(cap) or cap <= 0 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("cap and sigma must be finite and positive")
    limit = (1. + math.sqrt(1. + cap)) ** .25
    ratio = (radius_sum / distances.clamp_min(1e-6)).clamp(max=limit)
    fourth = ratio.square().square()
    lj = fourth.square() - 2. * fourth
    lj = torch.where(lj < cap, lj, cap)
    hbond = hbond_match * _slope_step(0., -.7, distances - radius_sum)
    solvation = torch.zeros_like(distances) if solvation_amplitude is None else (
        solvation_amplitude * torch.exp(-(distances / (2. * sigma)).square()))
    return {"vdw": lj, "hbond": hbond, "solvation": solvation}


def _validate_features(features, count, label):
    types = features.get("xs_types")
    if (not isinstance(types, (tuple, list)) or len(types) != count
            or any(not isinstance(t, str) or t not in _SUPPORTED_XS for t in types)):
        raise ValueError(f"DKOES scoring requires supported ordinary {label} XS atom roles")
    expected = _features_from_xs(types, "cpu", version="validation", source="validation")
    for key in ("radius_vina", "hydrophobic", "donor", "acceptor", "active"):
        value = features.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (count,) or not torch.isfinite(value).all():
            raise ValueError(f"invalid {label} feature {key}")
        if key == "active":
            if value.dtype != torch.bool or not torch.equal(value.cpu(), expected[key]):
                raise ValueError(f"{label} activity must match its XS roles")
        elif (value.dtype not in (torch.float32, torch.float64)
                or not torch.allclose(value.cpu(), expected[key].to(dtype=value.dtype), rtol=0, atol=1e-7)):
            raise ValueError(f"{label} feature {key} must match its XS roles")


def _solvation_parameters(features, count, device, dtype, label):
    types, charges = features.get("ad4_types"), features.get("ad4_charges")
    if (not isinstance(types, (tuple, list)) or len(types) != count
            or any(not isinstance(kind, str) or kind not in _SOLVATION_TYPES for kind in types)):
        raise ValueError(f"dkoes_scoring requires ordinary prepared {label} AD4 types and charges")
    if not isinstance(charges, torch.Tensor) or charges.shape != (count,) or not torch.isfinite(charges).all():
        raise ValueError(f"dkoes_scoring requires finite prepared {label} AD4 charges")
    compatible = {"C": {"C_H", "C_P"}, "A": {"C_H", "C_P"}, "N": {"N_P", "N_D"}, "NA": {"N_A", "N_DA"},
                  "O": {"O_P", "O_D"}, "OA": {"O_A", "O_DA"}, "S": {"S_P"}, "SA": {"S_P"}, "P": {"P_P"},
                  "H": {"H"}, "HD": {"H"}, "F": {"F_H"}, "Cl": {"Cl_H"}, "Br": {"Br_H"}, "I": {"I_H"}}
    if any(xs not in compatible.get(kind, {"Met_D"}) for kind, xs in zip(types, features["xs_types"], strict=True)):
        raise ValueError(f"prepared {label} AD4 types disagree with XS roles")
    charges = charges.detach().to(device=device, dtype=dtype).clone()
    solv = torch.tensor([_AD4_TYPES[kind][4] for kind in types], device=device, dtype=dtype)
    volume = torch.tensor([_AD4_TYPES[kind][5] for kind in types], device=device, dtype=dtype)
    return solv, volume, charges


class SminaScorer:
    """Factory for continuous dkoes_fast/scoring_old/scoring presets.

    Preset coefficients and conf-independent operators follow the pinned
    GNINA source. Input XS inference/prepared chemistry is explicit. Charge-
    dependent dkoes_scoring requires supplied AD4 types/charges for both sides.
    No charge defaults or external protonation choices are introduced.
    """

    units = "kcal/mol-like"

    def __init__(self, preset: str = "dkoes_fast", *, positive_energy_cap: float | None = None):
        if not isinstance(preset, str) or preset not in SMINA_PRESETS:
            raise ValueError(f"unknown continuous DKOES preset {preset!r}; choose from {sorted(SMINA_PRESETS)}")
        if positive_energy_cap is not None and (isinstance(positive_energy_cap, bool)
                or not isinstance(positive_energy_cap, Real) or not math.isfinite(positive_energy_cap) or positive_energy_cap <= 0):
            raise ValueError("positive_energy_cap must be finite and positive or None")
        self.preset_name, self.preset = preset, SMINA_PRESETS[preset]
        self.positive_energy_cap = None if positive_energy_cap is None else float(positive_energy_cap)
        self.name = "smina-" + preset.replace("_", "-")

    @property
    def fingerprint(self):
        state = {"source": SMINA_SOURCE, "preset": self.preset_name, "parameters": asdict(self.preset),
                 "positive_energy_cap": self.positive_energy_cap}
        return "sha256:" + hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

    def prepare(self, ligand_features, receptor_coords, receptor_features, *, num_rotatable_bonds=0,
                intramolecular_mask=None, intermolecular_exclusion_mask=None):
        if (isinstance(num_rotatable_bonds, bool) or not isinstance(num_rotatable_bonds, Real)
                or not math.isfinite(num_rotatable_bonds) or num_rotatable_bonds < 0):
            raise ValueError("num_rotatable_bonds must be finite and nonnegative")
        if (not isinstance(receptor_coords, torch.Tensor) or receptor_coords.dtype not in (torch.float32, torch.float64)
                or receptor_coords.ndim != 2 or receptor_coords.shape[1] != 3 or receptor_coords.shape[0] < 1
                or not torch.isfinite(receptor_coords).all()):
            raise ValueError("receptor_coords must be finite nonempty float [M,3]")
        count = len(ligand_features.get("xs_types", ()))
        if count < 1:
            raise ValueError("ligand atom set must be nonempty")
        _validate_features(ligand_features, count, "ligand")
        _validate_features(receptor_features, receptor_coords.shape[0], "receptor")
        device, dtype = receptor_coords.device, receptor_coords.dtype
        inter = prepare_interaction_matrices(ligand_features, receptor_features, VINA_CONFIG, device)
        intra = prepare_interaction_matrices(ligand_features, ligand_features, VINA_CONFIG, device)
        if self.preset.solvation_weight:
            left = _solvation_parameters(ligand_features, count, device, dtype, "ligand")
            right = _solvation_parameters(receptor_features, receptor_coords.shape[0], device, dtype, "receptor")
            for matrix, other in ((inter, right), (intra, left)):
                ls, lv, lq = left
                rs, rv, rq = other
                matrix["solvation"] = ((ls + self.preset.charge_solvation * lq.abs())[:, None] * rv[None]
                                         + lv[:, None] * (rs + self.preset.charge_solvation * rq.abs())[None])
        for matrix in (inter, intra):
            for key, value in matrix.items():
                matrix[key] = value.detach().clone()
        return PreparedSminaScorer(self, receptor_coords.detach().clone(), count, num_rotatable_bonds,
                                    None if intramolecular_mask is None else intramolecular_mask.detach().clone(),
                                    None if intermolecular_exclusion_mask is None else intermolecular_exclusion_mask.detach().clone(),
                                    inter, intra)


@dataclass
class PreparedSminaScorer:
    factory: SminaScorer
    receptor_coords: torch.Tensor
    num_atoms: int
    num_rotatable_bonds: int | float
    intramolecular_mask: torch.Tensor | None
    intermolecular_exclusion_mask: torch.Tensor | None
    intermolecular_matrices: dict[str, torch.Tensor]
    intramolecular_matrices: dict[str, torch.Tensor]

    @property
    def name(self):
        return self.factory.name

    @property
    def units(self):
        return self.factory.units

    @property
    def fingerprint(self):
        return self.factory.fingerprint

    @property
    def effective_rotatable_bonds(self):
        return self.num_rotatable_bonds

    @property
    def torsion_penalty_applied(self):
        return self.num_rotatable_bonds > 0 and self.factory.preset.torsion_square_weight != 0

    def _pair_energy(self, distances, matrix):
        params = self.factory.preset
        matrix = {key: value.to(device=distances.device, dtype=distances.dtype if value.dtype != torch.bool else torch.bool)
                  for key, value in matrix.items()}
        terms = smina_pair_terms(distances, matrix["radius_sum"], matrix["hbond"], cap=params.repulsion_cap,
                                solvation_amplitude=matrix.get("solvation"), sigma=params.solvation_sigma)
        energy = (params.vdw_weight * terms["vdw"] + params.hbond_weight * terms["hbond"]
                  + params.solvation_weight * terms["solvation"])
        return energy.masked_fill(~(matrix["active"] & (distances < params.cutoff)), 0.)

    def raw_components(self, ligand_coords):
        if not isinstance(ligand_coords, torch.Tensor) or ligand_coords.dtype not in (torch.float32, torch.float64):
            raise ValueError("ligand coordinates must be float32/float64 tensors")
        if ligand_coords.ndim == 2:
            ligand_coords = ligand_coords.unsqueeze(0)
        if (ligand_coords.ndim != 3 or ligand_coords.shape[0] < 1 or ligand_coords.shape[1:] != (self.num_atoms, 3)
                or not torch.isfinite(ligand_coords).all()):
            raise ValueError("ligand coordinates must be finite nonempty [B,N,3] matching prepared atoms")
        batch = ligand_coords.shape[0]
        receptor = self.receptor_coords.to(device=ligand_coords.device, dtype=ligand_coords.dtype)
        inter = self._pair_energy(torch.cdist(ligand_coords, receptor[None].expand(batch, -1, -1)), self.intermolecular_matrices)
        exclusion = normalize_pair_mask(self.intermolecular_exclusion_mask, batch, self.num_atoms, receptor.shape[0],
                                        device=ligand_coords.device)
        if exclusion is not None:
            inter = inter.masked_fill(exclusion, 0.)
        cap = self.factory.positive_energy_cap
        intermolecular = inter.sum((1, 2)) if cap is None else positive_energy_curl(inter.sum(2), cap).sum(1)
        if self.intramolecular_mask is None:
            intra = torch.zeros_like(intermolecular)
        else:
            intra = self._pair_energy(torch.cdist(ligand_coords, ligand_coords), self.intramolecular_matrices)
            intra = positive_energy_curl(intra, cap)
            mask = normalize_pair_mask(self.intramolecular_mask, batch, self.num_atoms, self.num_atoms,
                                       device=ligand_coords.device)
            intra = (intra * mask).sum((1, 2)) * .5
        return RawScoreComponents(intermolecular, intra, intermolecular + intra)

    def search_energy(self, ligand_coords):
        return self.raw_components(ligand_coords).search_energy

    def report(self, raw, intramolecular_reference=None):
        reference = (raw.intramolecular[torch.argmin(raw.search_energy)].detach() if intramolecular_reference is None else
                     torch.as_tensor(intramolecular_reference, device=raw.search_energy.device, dtype=raw.search_energy.dtype))
        if reference.ndim != 0 or not torch.isfinite(reference):
            raise ValueError("intramolecular_reference must be one finite scalar")
        params = self.factory.preset
        # Source num_tors_sqr reads 0.1*w, then adds w*num_tors^2/5.
        correction = .1 * params.torsion_square_weight * self.num_rotatable_bonds**2 / 5. + params.constant
        return ScoreComponents(raw.intermolecular, raw.intramolecular, raw.search_energy,
                               raw.search_energy - reference + correction, reference)

    def score_components(self, ligand_coords, intramolecular_reference=None):
        return self.report(self.raw_components(ligand_coords), intramolecular_reference)
