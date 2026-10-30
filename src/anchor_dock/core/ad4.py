"""Analytic rigid-receptor AD4 scoring with explicit prepared molecular state.

Standard AD4 uses Mehler--Solmajer dielectric and Gaussian desolvation. The
separate adgpu formulation evaluates the inspected GPU forward approximations
with exact autograd derivatives. Native radial tables/search updates differ.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .autogrid import AutoGridMapSet, load_autogrid_maps
from .grid import AD4IntermolecularGrid
from .scoring import RawScoreComponents, ScoreComponents

# Ordinary AD4.2 parameter facts: radius, depth, signed HB depth, HB radius,
# solvation and atomic volume. Extended/dummy types require separate contracts.
_AD4_TYPES = {
    "C": (2., .150, 0., 0., -.00143, 33.5103),
    "A": (2., .150, 0., 0., -.00052, 33.5103),
    "N": (1.75, .160, 0., 0., -.00162, 22.4493),
    "O": (1.60, .200, 0., 0., -.00251, 17.1573),
    "P": (2.10, .200, 0., 0., -.00110, 38.7924),
    "S": (2., .200, 0., 0., -.00214, 33.5103),
    "H": (1., .020, 0., 0., .00051, 0.),
    "F": (1.545, .080, 0., 0., -.00110, 15.4480),
    "I": (2.360, .550, 0., 0., -.00110, 55.0585),
    "NA": (1.75, .160, -5., 1.9, -.00162, 22.4493),
    "OA": (1.60, .200, -5., 1.9, -.00251, 17.1573),
    "SA": (2., .200, -1., 2.5, -.00214, 33.5103),
    "HD": (1., .020, 1., 0., .00051, 0.),
    "Mg": (.650, .875, 0., 0., -.00110, 1.5600),
    "Mn": (.650, .875, 0., 0., -.00110, 2.1400),
    "Zn": (.740, .550, 0., 0., -.00110, 1.7000),
    "Ca": (.990, .550, 0., 0., -.00110, 2.7700),
    "Fe": (.650, .010, 0., 0., -.00110, 1.8400),
    "Cl": (2.045, .276, 0., 0., -.00110, 35.8235),
    "Br": (2.165, .389, 0., 0., -.00110, 42.5661),
    "Si": (2.30, .200, 0., 0., -.00143, 50.9650),
    "At": (2.40, .550, 0., 0., -.00110, 57.9058),
}
AD4_CONTRACT = "analytic-ad4-grid-v1"


@dataclass(frozen=True)
class AD4Weights:
    """Intra-pair and torsion weights; supplied affinity maps are already weighted.

    Changing these weights never reweights maps. Their generation settings
    must be tracked separately through input/map hashes.
    """
    vdw: float = .1662
    hbond: float = .1209
    electrostatic: float = .1406
    desolvation: float = .1322
    torsion: float = .2983

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError("AD4 weights must be finite and nonnegative")


@dataclass(frozen=True)
class AD4PairComponents:
    vdw: torch.Tensor
    hbond: torch.Tensor
    electrostatic: torch.Tensor
    desolvation: torch.Tensor
    curl_correction: torch.Tensor
    energy: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.energy


def _types(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("one prepared AD4 type is required per atom")
    result = tuple(value)
    if not result or any(t not in _AD4_TYPES for t in result):
        raise ValueError("unsupported AD4 type; only ordinary parameterized types are supported")
    return result


def _coords(value: torch.Tensor, atoms: int, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("coordinates must be a tensor")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] < 1 or value.shape[1:] != (atoms, 3):
        raise ValueError("coordinates must have shape [B,N,3] or [N,3] matching the prepared atoms")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError("coordinate dtype/device must match the prepared AD4 scorer")
    if not torch.isfinite(value).all():
        raise ValueError("coordinates must be finite")
    return value


class AD4PairEnergy(nn.Module):
    """Prepared intramolecular contributors, each supplied once as i<j.

    The caller excludes bonded/1-3/1-4 and distance-invariant pairs. Standard
    AD4 uses smoothed 12-6/12-10, electrostatics and Gaussian desolvation.
    adgpu evaluates its rational forward approximations; its approximate,
    quantized/clipped native update directions are deliberately not copied.
    """

    def __init__(self, atom_types: Sequence[str], charges, pairs: Sequence[tuple[int, int]], *,
                 formulation: str = "standard", smoothing: float = .5, weights: AD4Weights = AD4Weights(),
                 pair_curl: float | None = None,
                 dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.atom_types = _types(atom_types)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("AD4 scoring requires float32 or float64")
        if formulation not in {"standard", "adgpu"}:
            raise ValueError("formulation must be standard or adgpu")
        if not math.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be finite and nonnegative")
        self.formulation, self.smoothing, self.weights = formulation, float(smoothing), weights
        if pair_curl is not None and (not math.isfinite(pair_curl) or pair_curl <= 0):
            raise ValueError("pair_curl must be finite and positive, or None")
        self.pair_curl = pair_curl
        prepared_charges = torch.as_tensor(charges, dtype=dtype, device=device).detach().clone()
        if prepared_charges.shape != (len(self.atom_types),) or not torch.isfinite(prepared_charges).all():
            raise ValueError("one finite prepared charge is required per atom")
        contributors = tuple(tuple(p) for p in pairs)
        if any(len(p) != 2 or any(type(i) is not int for i in p) or not 0 <= p[0] < p[1] < len(self.atom_types)
               for p in contributors) or len(set(contributors)) != len(contributors):
            raise ValueError("contributor pairs must be unique integer atom indices i<j")
        self.register_buffer("charges", prepared_charges)
        self.register_buffer("pairs", torch.tensor(contributors, dtype=torch.long, device=device).reshape(-1, 2))
        self.register_buffer("parameters_by_atom", torch.tensor([_AD4_TYPES[t] for t in self.atom_types], dtype=dtype, device=device))

    def components(self, coordinates: torch.Tensor) -> AD4PairComponents:
        coords = _coords(coordinates, len(self.atom_types), self.charges)
        left, right = self.pairs.unbind(-1)
        distance = torch.linalg.vector_norm(coords[:, left] - coords[:, right], dim=-1)
        p, q = self.parameters_by_atom[left], self.parameters_by_atom[right]
        hb_product = p[:, 2] * q[:, 2]
        hydrogen_bond = hb_product < 0
        optimum = torch.where(hydrogen_bond, p[:, 3] + q[:, 3], p[:, 0] + q[:, 0])
        half = self.smoothing * .5
        smoothed = torch.where(distance < optimum - half, distance + half,
                               torch.where(distance > optimum + half, distance - half, optimum))
        # Beyond ratio=100 the capped repulsion is already constant; clipping
        # before powers keeps inactive branches and zero-distance gradients finite.
        ratio = optimum / smoothed.clamp_min(1e-6)
        if self.formulation == "standard":
            ratio = ratio.clamp_max(100.)
        vdw = (p[:, 1] * q[:, 1]).sqrt() * (ratio.pow(12) - 2 * ratio.pow(6))
        hb = -hb_product * (5 * ratio.pow(12) - 6 * ratio.pow(10))
        active = distance < 8.
        if self.formulation == "standard":
            vdw, hb = vdw.clamp_max(100000.), hb.clamp_max(100000.)
        elif not torch.isfinite(vdw).all() or not torch.isfinite(hb).all():
            raise ValueError("ADGPU pair repulsion exceeds the supported numerical range")
        vdw = torch.where(active & ~hydrogen_bond, vdw * self.weights.vdw, 0.)
        hb = torch.where(active & hydrogen_bond, hb * self.weights.hbond, 0.)
        charge_left, charge_right = self.charges[left], self.charges[right]
        if self.formulation == "adgpu":
            r = distance.clamp_min(.01)
            reciprocal_dielectric = 1.10859 / (r + 1.26366).square() + .010358
            electro = 332.06363 * charge_left * charge_right * reciprocal_dielectric / r
            r2 = r.square()
            desolv_distance = (12.96 - .1063 * r2 * (1 - .001947 * r2)) / (
                12.96 + r2 * (.4137 + r2 * (.00357 + .000112 * r2)))
        else:
            r = distance
            b = 78.4 + 8.5525
            dielectric = -8.5525 + b / (1 + 7.7839 * torch.exp(-b * .003627 * r))
            reciprocal = (1 / (r.clamp_min(1e-8) * dielectric)).clamp_max(100.)
            # Native's exact-coincidence branch places the cap before the
            # dielectric division; use a finite unselected reciprocal branch.
            reciprocal = torch.where(r < torch.finfo(torch.float64).eps, 100. / dielectric, reciprocal)
            electro = 332. * charge_left * charge_right * reciprocal
            desolv_distance = torch.exp(-.5 * (distance / 3.6).square())
        amplitude = ((p[:, 4] + .01097 * charge_left.abs()) * q[:, 5]
                     + (q[:, 4] + .01097 * charge_right.abs()) * p[:, 5])
        electro = torch.where(distance < 20.48, electro * self.weights.electrostatic, 0.)
        desolv = torch.where(distance < 20.48, amplitude * desolv_distance * self.weights.desolvation, 0.)
        total = vdw + hb + electro + desolv
        curled = total
        if self.pair_curl is not None:
            curled = total / (1 + total.clamp_min(0.) / self.pair_curl)
        return AD4PairComponents(vdw.sum(-1), hb.sum(-1), electro.sum(-1), desolv.sum(-1),
                                 (curled - total).sum(-1), curled.sum(-1))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.components(coordinates).total


@dataclass(frozen=True)
class AD4ScoreComponents:
    intermolecular: torch.Tensor
    intramolecular: torch.Tensor
    search_energy: torch.Tensor
    torsion: torch.Tensor
    unbound: torch.Tensor
    score: torch.Tensor


class AD4GridScorer(nn.Module):
    """Grid + prepared intra + torsion + explicit unbound reporting.

    Search minimizes inter+intra. same_as_bound reports inter+torsion; fixed
    reports inter+intra+torsion-unbound_energy. The unbound model is mandatory.
    Maps carry explicit raw/curled and closed/upper-exclusive conventions.
    """

    def __init__(self, maps: AD4IntermolecularGrid, atom_types: Sequence[str], charges,
                 intra_pairs: Sequence[tuple[int, int]], *, torsdof: int, unbound_model: str,
                 unbound_energy: float | None = None, formulation: str = "standard",
                 pair_curl: float | None = None,
                 weights: AD4Weights = AD4Weights()) -> None:
        super().__init__()
        self.atom_types = _types(atom_types)
        if type(torsdof) is not int or torsdof < 0:
            raise ValueError("TORSDOF must be a nonnegative prepared integer")
        _validate_unbound(unbound_model, unbound_energy)
        try:
            channels = [maps.atom_types.index(kind) for kind in self.atom_types]
        except ValueError as exc:
            raise ValueError("an affinity map is missing for a prepared ligand atom type") from exc
        self.maps = maps
        self.pair_energy = AD4PairEnergy(self.atom_types, charges, intra_pairs, formulation=formulation,
                                        weights=weights, pair_curl=pair_curl,
                                        dtype=maps.field.values.dtype, device=maps.field.values.device)
        self.register_buffer("channels", torch.tensor(channels, dtype=torch.long, device=maps.field.values.device))
        self.torsdof, self.unbound_model, self.unbound_energy = torsdof, unbound_model, unbound_energy
        self.weights = weights

    def components(self, coordinates: torch.Tensor) -> AD4ScoreComponents:
        coordinates = _coords(coordinates, len(self.atom_types), self.pair_energy.charges)
        inter = self.maps(coordinates, self.channels, self.pair_energy.charges)
        intra = self.pair_energy(coordinates)
        search = inter + intra
        torsion = torch.full_like(search, self.weights.torsion * self.torsdof)
        unbound = intra if self.unbound_model == "same_as_bound" else torch.full_like(search, self.unbound_energy)
        score = inter + torsion if self.unbound_model == "same_as_bound" else search + torsion - unbound
        return AD4ScoreComponents(inter, intra, search, torsion, unbound, score)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.components(coordinates).search_energy


def _validate_unbound(model, energy):
    if model not in {"same_as_bound", "fixed"}:
        raise ValueError("unbound_model must be same_as_bound or fixed")
    if model == "same_as_bound" and energy is not None:
        raise ValueError("same_as_bound does not accept a fixed unbound energy")
    if model == "fixed" and (energy is None or not math.isfinite(energy)):
        raise ValueError("fixed unbound model requires a finite unbound_energy")


class AD4Scorer:
    """Engine-compatible reference factory requiring authoritative prepared inputs.

    Raw maps use an upper-exclusive domain, as ADGPU does. Vina map mode curls
    each positive map sample with v=1000 before q/abs(q) multiplication and
    uses closed bounds; it also curls the summed intra energy of each pair.
    Intra energies use analytic formulas rather than
    native radial tables. Maps represent the full fixed receptor: per-atom
    receptor exclusions/covalent corrections are unsupported. Maps are the
    authoritative receptor input; this module cannot establish that a separate
    receptor coordinate tensor produced them. Their source hashes are exposed.
    """

    name = "ad4-grid"
    units = "kcal/mol-like"

    def __init__(self, maps: AutoGridMapSet, *, map_mode: str = "raw", formulation: str = "standard",
                 unbound_model: str = "same_as_bound", unbound_energy: float | None = None,
                 weights: AD4Weights = AD4Weights()) -> None:
        if map_mode not in {"raw", "vina"}:
            raise ValueError("map_mode must be raw or vina")
        if formulation not in {"standard", "adgpu"}:
            raise ValueError("formulation must be standard or adgpu")
        _validate_unbound(unbound_model, unbound_energy)
        self.map_mode, self.formulation = map_mode, formulation
        self.unbound_model, self.unbound_energy, self.weights = unbound_model, unbound_energy, weights
        self.map_source_sha256 = maps.source_sha256
        self.grid = maps.to_inter_grid(map_curl=1000. if map_mode == "vina" else None, upper_exclusive=map_mode == "raw")

    @classmethod
    def from_autogrid(cls, prefix, atom_types, *, dtype=torch.float32, device="cpu", **kwargs):
        return cls(load_autogrid_maps(prefix, atom_types, dtype=dtype, device=device), **kwargs)

    @property
    def fingerprint(self) -> str:
        metadata = dict(contract=AD4_CONTRACT, map_types=self.grid.atom_types, map_mode=self.map_mode,
                        formulation=self.formulation, weights=vars(self.weights),
                        unbound_model=self.unbound_model, unbound_energy=self.unbound_energy,
                        atom_parameters=_AD4_TYPES)
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        for field in (self.grid.field.values, self.grid.field.origin, self.grid.field.spacing):
            digest.update(str((str(field.dtype), tuple(field.shape))).encode())
            digest.update(field.detach().contiguous().cpu().numpy().tobytes())
        return "sha256:" + digest.hexdigest()

    def prepare(self, ligand_features, receptor_coords, receptor_features, *, num_rotatable_bonds=0,
                intramolecular_mask=None, intermolecular_exclusion_mask=None):
        del receptor_features, num_rotatable_bonds
        if intermolecular_exclusion_mask is not None:
            raise ValueError("AD4 maps cannot apply per-receptor-atom exclusions or covalent corrections")
        if receptor_coords.device != self.grid.field.values.device:
            raise ValueError("AD4 map and receptor device must match")
        required = ("ad4_types", "ad4_charges", "ad4_torsdof", "prepared_intramolecular_pairs")
        if any(key not in ligand_features for key in required):
            raise ValueError("AD4 requires authoritative prepared atom types, charges, TORSDOF and mobility pairs")
        kinds, charges, torsdof, pairs = (ligand_features[key] for key in required)
        if intramolecular_mask is not None:
            expected = torch.zeros((len(kinds), len(kinds)), dtype=torch.bool, device=intramolecular_mask.device)
            for a, b in pairs:
                expected[a, b] = expected[b, a] = True
            if intramolecular_mask.shape != expected.shape or not torch.equal(intramolecular_mask, expected):
                raise ValueError("AD4 intramolecular mask must match the authoritative prepared contributor pairs")
        count = len(self.grid.atom_types)
        snapshot = AD4IntermolecularGrid(self.grid.field.values[:count], self.grid.field.values[count],
                                         self.grid.field.values[count + 1], atom_types=self.grid.atom_types,
                                         origin=self.grid.field.origin, spacing=self.grid.field.spacing,
                                         map_curl=self.grid.map_curl, upper_exclusive=self.grid.upper_exclusive)
        module = AD4GridScorer(snapshot, kinds, charges, pairs, torsdof=torsdof,
                               unbound_model=self.unbound_model, unbound_energy=self.unbound_energy,
                               formulation=self.formulation, pair_curl=1000. if self.map_mode == "vina" else None,
                               weights=self.weights)
        state = dict(factory=self.fingerprint, types=kinds, charges=module.pair_energy.charges.detach().cpu().tolist(),
                     pairs=module.pair_energy.pairs.cpu().tolist(), torsdof=torsdof)
        fingerprint = "sha256:" + hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        return PreparedAD4Scorer(module, fingerprint)


@dataclass
class PreparedAD4Scorer:
    module: AD4GridScorer
    fingerprint: str
    name: str = "ad4-grid"
    units: str = "kcal/mol-like"

    @property
    def effective_rotatable_bonds(self) -> int:
        return self.module.torsdof

    @property
    def torsion_penalty_applied(self) -> bool:
        return self.module.torsdof > 0 and self.module.weights.torsion > 0

    def search_energy(self, ligand_coords):
        return self.module(ligand_coords)

    def raw_components(self, ligand_coords):
        value = self.module.components(ligand_coords)
        return RawScoreComponents(value.intermolecular, value.intramolecular, value.search_energy)

    def report(self, raw, intramolecular_reference=None):
        del intramolecular_reference
        unbound = raw.intramolecular if self.module.unbound_model == "same_as_bound" else raw.search_energy.new_tensor(self.module.unbound_energy)
        base = raw.intermolecular if self.module.unbound_model == "same_as_bound" else raw.search_energy - unbound
        score = base + self.module.weights.torsion * self.module.torsdof
        return ScoreComponents(raw.intermolecular, raw.intramolecular, raw.search_energy, score, unbound)

    def score_components(self, ligand_coords, intramolecular_reference=None):
        return self.report(self.raw_components(ligand_coords), intramolecular_reference)
