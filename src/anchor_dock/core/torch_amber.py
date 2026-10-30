"""Pure Torch fixed-charge AMBER/OpenMM vacuum potential.

OpenMM is used only to import a fully parameterized topology. Evaluation and
automatic derivatives use Torch on the selected device. This is the supplied
complex's potential energy, not binding affinity or an MM/GBSA delta.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral
from pathlib import Path

import torch

from .amber import PreparedOpenMMScorer, _fixed_positions, _openmm

TORCH_AMBER_CONTRACT = "torch-fixed-charge-vacuum-nocutoff-fp64-v2"
# OpenMM's CODATA-derived ONE_4PI_EPS0 in kJ*nm/(mol*e^2), converted to kcal*A.
COULOMB_KCAL_ANGSTROM = 138.93545764438198 * 10. / 4.184


class TorchAmberGeometryError(ValueError, FloatingPointError):
    """Undefined physical geometry; an optimizer may reject this trial pose."""


def _finite(value, name, *, nonnegative=False):
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


class TorchAmberScorer:
    """Snapshot a supported all-atom System, then evaluate entirely in Torch.

    Coordinates are angstroms; energy is FP64 kcal/mol, with gradients through
    FP32 or FP64 input coordinates. Parameters are fixed buffers, never learned
    weights. Use ``to(device)`` before optimization; implicit per-call parameter
    transfers are deliberately avoided. Only vacuum NoCutoff fixed-charge
    Coulomb, Lorentz--Berthelot LJ, harmonic bonds/angles and periodic torsions
    are supported. Explicit exceptions replace ordinary nonbonded pairs.

    Active zero-distance pairs, zero-length bonds, collinear angles and
    degenerate torsion planes raise an error instead of modifying the physical
    potential with a soft core. Zero-strength terms/excluded pairs are omitted
    before geometry is evaluated. Constraints do not remove energy terms and
    are not enforced by this potential; the optimizer chooses its coordinates.

    Static moving-pair indices and exception-adjusted coefficients are cached
    when the candidate pair count is at most ``pair_cache_max_pairs`` (200000
    by default). The bound applies before exclusions, so even highly excluded
    large systems cannot create an unbounded cache. Zero disables the cache.
    Larger systems use bounded atom tiles. Cached evaluation also chunks pairs
    to at most ``pair_chunk_size**2`` per batch operation.
    """

    name = "amber-torch"
    units = "kcal/mol"
    direction = "lower"
    differentiable = True
    requires_all_atoms = True
    preparation_requirement = "amber-topology"
    fixed_receptor = True
    objective = "full_complex_potential"
    component_semantics = "undecomposed-full-complex-potential"

    def __init__(self, system, positions, ligand_atom_indices: Sequence[int], *,
                 atomic_numbers: Sequence[int] | None = None, device="cpu", dtype=torch.float64,
                 pair_chunk_size=128, source_metadata: Mapping | None = None,
                 topology_bonds: Sequence[tuple[int, int]] | None = None,
                 pair_cache_max_pairs=200000):
        openmm, _, unit = _openmm()
        if not isinstance(system, openmm.System):
            raise TypeError("system must be an OpenMM System")
        count = system.getNumParticles()
        if count < 1 or any(system.isVirtualSite(i) for i in range(count)):
            raise ValueError("explicit nonempty particle Systems without virtual sites are required")
        if isinstance(ligand_atom_indices, (str, bytes)):
            raise ValueError("ligand_atom_indices must be an explicit ordered particle mapping")
        try:
            indices = tuple(ligand_atom_indices)
        except TypeError as error:
            raise ValueError("ligand_atom_indices must be an explicit ordered particle mapping") from error
        if (not indices or any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < count for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("ligand_atom_indices must be unique valid integer System indices")
        if atomic_numbers is not None:
            atomic_numbers = tuple(atomic_numbers)
            if len(atomic_numbers) != count or any(
                isinstance(z, bool) or not isinstance(z, Integral) or not 1 <= z <= 118 for z in atomic_numbers
            ):
                raise ValueError("atomic_numbers must contain one valid element per System particle")
            atomic_numbers = tuple(int(z) for z in atomic_numbers)
        if isinstance(pair_chunk_size, bool) or not isinstance(pair_chunk_size, Integral) or pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be a positive integer atom tile width")
        if (isinstance(pair_cache_max_pairs, bool) or not isinstance(pair_cache_max_pairs, Integral)
                or pair_cache_max_pairs < 0):
            raise ValueError("pair_cache_max_pairs must be a nonnegative integer; zero disables caching")
        self.pair_chunk_size = int(pair_chunk_size)
        self.pair_cache_max_pairs = int(pair_cache_max_pairs)
        self.ligand_atom_indices = tuple(int(i) for i in indices)
        selected = set(indices)
        self.receptor_atom_indices = tuple(i for i in range(count) if i not in selected)
        self.atomic_numbers = atomic_numbers
        self._ligand_edges = None
        if topology_bonds is not None:
            local = {particle: i for i, particle in enumerate(indices)}
            edges = set()
            for edge in topology_bonds:
                try:
                    edge = tuple(edge)
                except TypeError as error:
                    raise ValueError("topology_bonds must contain explicit particle-index pairs") from error
                if (len(edge) != 2 or any(isinstance(i, bool) or not isinstance(i, Integral)
                                          or not 0 <= i < count for i in edge) or edge[0] == edge[1]):
                    raise ValueError("topology_bonds must be pairs of distinct valid System particle indices")
                a, b = edge
                if (atomic_numbers is not None and (a in selected) != (b in selected)
                        and (atomic_numbers[a] == 1 or atomic_numbers[b] == 1)):
                    raise ValueError("ligand mapping must include bonded hydrogens with their parent atom")
                if a in selected and b in selected:
                    edges.add(tuple(sorted((local[a], local[b]))))
            self._ligand_edges = frozenset(edges)
        positions = _fixed_positions(positions, count, unit)
        self._source_metadata = deepcopy(dict(source_metadata or {}))
        json.dumps(self._source_metadata, sort_keys=True, allow_nan=False)
        self._openmm_version = openmm.__version__
        self._system_xml = openmm.XmlSerializer.serialize(system)
        self._buffers = {"positions": torch.tensor(positions.copy(), dtype=torch.float64),
                         "ligand": torch.tensor(indices, dtype=torch.long),
                         "receptor": torch.tensor(self.receptor_atom_indices, dtype=torch.long)}
        self._terms = {kind: {False: [], True: []} for kind in ("bond", "angle", "torsion")}
        nonbonded = []
        counts = {}
        for force in system.getForces():
            kind = type(force).__name__
            counts[kind] = counts.get(kind, 0) + 1
            if isinstance(force, openmm.CMMotionRemover):
                continue  # Exact zero potential; never run a dynamics step.
            if isinstance(force, (openmm.HarmonicBondForce, openmm.HarmonicAngleForce, openmm.PeriodicTorsionForce)):
                if force.usesPeriodicBoundaryConditions():
                    raise ValueError("periodic bonded forces are unsupported by Torch AMBER")
                if isinstance(force, openmm.HarmonicBondForce):
                    for i in range(force.getNumBonds()):
                        a, b, length, k = force.getBondParameters(i)
                        params = (_finite(length.value_in_unit(unit.angstrom), "bond length", nonnegative=True),
                                  _finite(k.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom**2), "bond k", nonnegative=True))
                        self._add_term("bond", (a, b), params, selected)
                elif isinstance(force, openmm.HarmonicAngleForce):
                    for i in range(force.getNumAngles()):
                        a, b, c, theta, k = force.getAngleParameters(i)
                        params = (_finite(theta.value_in_unit(unit.radian), "angle"),
                                  _finite(k.value_in_unit(unit.kilocalorie_per_mole / unit.radian**2), "angle k", nonnegative=True))
                        if not 0 <= params[0] <= math.pi:
                            raise ValueError("equilibrium angle must lie in [0,pi]")
                        self._add_term("angle", (a, b, c), params, selected)
                else:
                    for i in range(force.getNumTorsions()):
                        a, b, c, d, n, phase, k = force.getTorsionParameters(i)
                        if n < 1:
                            raise ValueError("torsion periodicity must be positive")
                        params = (int(n), _finite(phase.value_in_unit(unit.radian), "torsion phase"),
                                  _finite(k.value_in_unit(unit.kilocalorie_per_mole), "torsion k"))
                        self._add_term("torsion", (a, b, c, d), params, selected)
            elif isinstance(force, openmm.NonbondedForce):
                nonbonded.append(force)
            else:
                raise ValueError(f"unsupported Torch AMBER force: {kind}")
        if len(nonbonded) > 1:
            raise ValueError("multiple NonbondedForce objects are unsupported")
        self._has_nonbonded = bool(nonbonded)
        if nonbonded:
            self._import_nonbonded(nonbonded[0], count, unit)
        moving = len(self.ligand_atom_indices)
        fixed = len(self.receptor_atom_indices)
        self._moving_candidate_pairs = moving * fixed + moving * (moving - 1) // 2
        self._pair_cache_enabled = (self._has_nonbonded and self.pair_cache_max_pairs > 0
                                    and self._moving_candidate_pairs <= self.pair_cache_max_pairs)
        self._cached_active_pairs = 0
        if self._pair_cache_enabled:
            self._cache_moving_pairs()
        for kind, groups in self._terms.items():
            width = {"bond": 2, "angle": 3, "torsion": 4}[kind]
            for moving, terms in groups.items():
                key = f"{kind}_{int(moving)}"
                self._buffers[key + "_indices"] = torch.tensor([t[0] for t in terms], dtype=torch.long).reshape(-1, width)
                self._buffers[key + "_parameters"] = torch.tensor([t[1] for t in terms], dtype=torch.float64).reshape(-1, 3 if kind == "torsion" else 2)
        del self._terms
        self._force_counts = counts
        identity = {"contract": TORCH_AMBER_CONTRACT, "openmm_import_version": self._openmm_version,
                    "ligand_atom_indices": indices, "atomic_numbers": atomic_numbers,
                    "ligand_topology_edges": None if self._ligand_edges is None else sorted(self._ligand_edges),
                    "source": self._source_metadata, "pair_chunk_size": self.pair_chunk_size,
                    "pair_cache_max_pairs": self.pair_cache_max_pairs,
                    "compute_dtype": "float64"}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
        digest.update(self._system_xml.encode())
        digest.update(positions.tobytes())
        self._fingerprint = "sha256:" + digest.hexdigest()
        self.device, self.dtype = torch.device("cpu"), torch.float64
        # Fixed--fixed energy is computed once; it never enters the search
        # objective and cannot obscure small moving-coordinate improvements.
        with torch.no_grad():
            fixed = self._bonded(self._buffers["positions"][None], False)
            if self._has_nonbonded:
                fixed = fixed + self._nonbonded(self._buffers["positions"][None], self._buffers["receptor"], self._buffers["receptor"], True)
            self._fixed_energy = float(fixed[0])
            self._reference_moving = float(self._moving_energy(self._buffers["positions"][None])[0])
        if not all(math.isfinite(value) for value in (self._fixed_energy, self._reference_moving, self.reference_energy)):
            raise FloatingPointError("Torch AMBER reference potential is nonfinite")
        self.to(device, dtype=dtype)

    def _add_term(self, kind, indices, parameters, selected):
        count = self._buffers["positions"].shape[0]
        if any(not 0 <= i < count for i in indices):
            raise ValueError(f"{kind} contains an invalid System particle index")
        if parameters[-1] == 0:
            return
        if len(set(indices)) != len(indices):
            raise ValueError(f"{kind} requires distinct atom indices")
        self._terms[kind][any(i in selected for i in indices)].append((tuple(indices), parameters))

    def _import_nonbonded(self, force, count, unit):
        if force.getNonbondedMethod() != force.NoCutoff:
            raise ValueError("Torch AMBER supports only vacuum NoCutoff; PME/PBC/cutoffs are unsupported")
        if (force.getNumGlobalParameters() or force.getNumParticleParameterOffsets()
                or force.getNumExceptionParameterOffsets()):
            raise ValueError("NonbondedForce global parameters/offsets are unsupported")
        if not force.getIncludeDirectSpace() or force.getExceptionsUsePeriodicBoundaryConditions():
            raise ValueError("non-direct-space or periodic exceptions are unsupported")
        if force.getUseSwitchingFunction():
            raise ValueError("nonbonded switching functions are unsupported")
        if force.getNumParticles() != count:
            raise ValueError("NonbondedForce particle count must match System")
        values = []
        for i in range(count):
            q, sigma, epsilon = force.getParticleParameters(i)
            values.append((_finite(q.value_in_unit(unit.elementary_charge), "charge"),
                           _finite(sigma.value_in_unit(unit.angstrom), "sigma", nonnegative=True),
                           _finite(epsilon.value_in_unit(unit.kilocalorie_per_mole), "epsilon", nonnegative=True)))
        self._buffers["nonbonded"] = torch.tensor(values, dtype=torch.float64)
        exceptions = {}
        for i in range(force.getNumExceptions()):
            a, b, q, sigma, epsilon = force.getExceptionParameters(i)
            if a == b or not 0 <= a < count or not 0 <= b < count:
                raise ValueError("nonbonded exception requires two distinct valid particles")
            key = min(a, b) * count + max(a, b)
            if key in exceptions:
                raise ValueError("duplicate nonbonded exception")
            exceptions[key] = (_finite(q.value_in_unit(unit.elementary_charge**2), "exception charge product"),
                               _finite(sigma.value_in_unit(unit.angstrom), "exception sigma", nonnegative=True),
                               _finite(epsilon.value_in_unit(unit.kilocalorie_per_mole), "exception epsilon", nonnegative=True))
        keys = sorted(exceptions)
        self._buffers["exception_keys"] = torch.tensor(keys, dtype=torch.long)
        self._buffers["exception_parameters"] = torch.tensor([exceptions[k] for k in keys], dtype=torch.float64).reshape(-1, 3)

    @classmethod
    def from_system(cls, system, positions, ligand_atom_indices, **kwargs):
        return cls(system, positions, ligand_atom_indices, **kwargs)

    @classmethod
    def from_amber(cls, prmtop, inpcrd, ligand_atom_indices, *, implicit_solvent=None,
                   nonbonded_method="NoCutoff", device="cpu", dtype=torch.float64,
                   pair_chunk_size=128, pair_cache_max_pairs=200000):
        """Import an AMBER vacuum topology once; no Context is constructed."""
        if implicit_solvent is not None or nonbonded_method != "NoCutoff":
            raise ValueError("Torch AMBER supports vacuum NoCutoff only; implicit solvent/PME/PBC are unsupported")
        _, app, _ = _openmm()
        prmtop, inpcrd = Path(prmtop), Path(inpcrd)
        parameters, coordinates = prmtop.read_bytes(), inpcrd.read_bytes()
        coordinate_file = app.AmberInpcrdFile(str(inpcrd))
        if coordinate_file.boxVectors is not None:
            raise ValueError("periodic AMBER coordinate boxes are unsupported by the vacuum Torch potential")
        topology = app.AmberPrmtopFile(str(prmtop))
        system = topology.createSystem(nonbondedMethod=app.NoCutoff, implicitSolvent=None,
                                       constraints=None, rigidWater=False, removeCMMotion=False)
        if parameters != prmtop.read_bytes() or coordinates != inpcrd.read_bytes():
            raise ValueError("AMBER input files changed while being imported")
        elements = [atom.element.atomic_number if atom.element is not None else 0 for atom in topology.topology.atoms()]
        bonds = [(a.index, b.index) for a, b in topology.topology.bonds()]
        return cls(system, coordinate_file.positions, ligand_atom_indices, atomic_numbers=elements,
                   device=device, dtype=dtype, pair_chunk_size=pair_chunk_size, topology_bonds=bonds,
                   pair_cache_max_pairs=pair_cache_max_pairs,
                   source_metadata={"prmtop_sha256": hashlib.sha256(parameters).hexdigest(),
                                    "inpcrd_sha256": hashlib.sha256(coordinates).hexdigest(),
                                    "implicit_solvent": None, "nonbonded_method": "NoCutoff"})

    def to(self, device=None, dtype=None):
        """Move fixed parameter buffers before optimization (FP64 only)."""
        if dtype is not None and dtype != torch.float64:
            raise ValueError("Torch AMBER requires float64 compute parameters for physical energy precision")
        device = self.device if device is None else torch.device(device)
        self._buffers = {key: value.to(device=device) for key, value in self._buffers.items()}
        self.device = self._buffers["positions"].device
        return self

    def _bonded(self, xyz, moving):
        total = xyz[:, 0, 0] * 0.
        for kind in ("bond", "angle", "torsion"):
            key = f"{kind}_{int(moving)}"
            indices, parameters = self._buffers[key + "_indices"], self._buffers[key + "_parameters"]
            if indices.shape[0] == 0:
                continue
            points = xyz[:, indices]
            first = points[:, :, 1] - points[:, :, 0]
            if kind == "bond":
                length = torch.linalg.vector_norm(first, dim=-1)
                if bool((length == 0).any()):
                    raise TorchAmberGeometryError("active harmonic bond has zero length")
                value = .5 * parameters[:, 1] * (length - parameters[:, 0]).square()
            else:
                second = points[:, :, 2] - points[:, :, 1]
                if kind == "angle":
                    left, right = -first, second
                    cross = torch.linalg.cross(left, right, dim=-1)
                    cross_norm = torch.linalg.vector_norm(cross, dim=-1)
                    scale = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
                    if bool(((scale == 0) | (cross_norm <= 1e-12 * scale)).any()):
                        raise TorchAmberGeometryError("active harmonic angle is collinear or has a zero-length bond")
                    angle = torch.atan2(cross_norm, (left * right).sum(-1))
                    value = .5 * parameters[:, 1] * (angle - parameters[:, 0]).square()
                else:
                    third = points[:, :, 3] - points[:, :, 2]
                    n1 = torch.linalg.cross(first, second, dim=-1)
                    n2 = torch.linalg.cross(second, third, dim=-1)
                    n1_norm, n2_norm = torch.linalg.vector_norm(n1, dim=-1), torch.linalg.vector_norm(n2, dim=-1)
                    central = torch.linalg.vector_norm(second, dim=-1)
                    scale1 = torch.linalg.vector_norm(first, dim=-1) * central
                    scale2 = torch.linalg.vector_norm(third, dim=-1) * central
                    if bool(((central == 0) | (n1_norm <= 1e-12 * scale1) | (n2_norm <= 1e-12 * scale2)).any()):
                        raise TorchAmberGeometryError("active periodic torsion has a degenerate plane or zero central bond")
                    sine = (torch.linalg.cross(n1, n2, dim=-1) * (second / central[..., None])).sum(-1)
                    angle = torch.atan2(sine, (n1 * n2).sum(-1))
                    value = parameters[:, 2] * (1. + torch.cos(parameters[:, 0] * angle - parameters[:, 1]))
            total = total + value.sum(-1)
        return total

    def _pair_tiles(self, first, second, triangular):
        """Yield bounded index tiles, preserving explicit unsorted atom maps."""
        tile = self.pair_chunk_size
        for start in range(0, len(first), tile):
            for other in range(start if triangular else 0, len(second), tile):
                a, b = torch.meshgrid(first[start:start+tile], second[other:other+tile], indexing="ij")
                a, b = a.flatten(), b.flatten()
                if triangular and start == other:
                    # Ordering belongs to the explicit input map, which need
                    # not be sorted by System particle number.
                    row, column = torch.meshgrid(torch.arange(len(first[start:start+tile]), device=first.device),
                                                 torch.arange(len(second[other:other+tile]), device=first.device), indexing="ij")
                    keep = (row < column).flatten()
                    a, b = a[keep], b[keep]
                if a.numel():
                    yield a, b

    def _pair_parameters(self, a, b):
        params = self._buffers["nonbonded"]
        keys, exceptions = self._buffers["exception_keys"], self._buffers["exception_parameters"]
        count = self._buffers["positions"].shape[0]
        product = params[a, 0] * params[b, 0]
        sigma = .5 * (params[a, 1] + params[b, 1])
        epsilon = torch.sqrt(params[a, 2]) * torch.sqrt(params[b, 2])
        if keys.numel():
            pair_keys = torch.minimum(a, b) * count + torch.maximum(a, b)
            location = torch.searchsorted(keys, pair_keys).clamp(max=len(keys)-1)
            match = keys[location] == pair_keys
            replacement = exceptions[location]
            product = torch.where(match, replacement[:, 0], product)
            sigma = torch.where(match, replacement[:, 1], sigma)
            epsilon = torch.where(match, replacement[:, 2], epsilon)
        active = (product != 0) | ((epsilon != 0) & (sigma != 0))
        return a[active], b[active], torch.stack((product[active], sigma[active], epsilon[active]), dim=-1)

    def _cache_moving_pairs(self):
        pairs, parameters = [], []
        movable, fixed = self._buffers["ligand"], self._buffers["receptor"]
        for first, second, triangular in ((movable, fixed, False), (movable, movable, True)):
            for a, b in self._pair_tiles(first, second, triangular):
                a, b, coefficients = self._pair_parameters(a, b)
                if a.numel():
                    pairs.append(torch.stack((a, b), dim=-1))
                    parameters.append(coefficients)
        self._buffers["moving_pair_indices"] = torch.cat(pairs) if pairs else torch.empty((0, 2), dtype=torch.long)
        self._buffers["moving_pair_parameters"] = torch.cat(parameters) if parameters else torch.empty((0, 3), dtype=torch.float64)
        self._cached_active_pairs = len(self._buffers["moving_pair_indices"])

    @staticmethod
    def _pair_energy(xyz, a, b, parameters):
        distance = torch.linalg.vector_norm(xyz[:, a] - xyz[:, b], dim=-1)
        if bool((distance == 0).any()):
            raise TorchAmberGeometryError("active nonbonded pair has zero separation")
        product, sigma, epsilon = parameters.unbind(dim=-1)
        ratio6 = (sigma / distance).pow(6)
        return (COULOMB_KCAL_ANGSTROM * product / distance
                + 4. * epsilon * (ratio6.square() - ratio6)).sum(-1)

    def _nonbonded(self, xyz, first, second, triangular):
        total = xyz[:, 0, 0] * 0.
        for a, b in self._pair_tiles(first, second, triangular):
            a, b, parameters = self._pair_parameters(a, b)
            if a.numel():
                total = total + self._pair_energy(xyz, a, b, parameters)
        return total

    def _cached_nonbonded(self, xyz):
        total = xyz[:, 0, 0] * 0.
        pairs, parameters = self._buffers["moving_pair_indices"], self._buffers["moving_pair_parameters"]
        chunk = self.pair_chunk_size**2
        for start in range(0, self._cached_active_pairs, chunk):
            block = pairs[start:start+chunk]
            total = total + self._pair_energy(xyz, block[:, 0], block[:, 1], parameters[start:start+chunk])
        return total

    def _moving_energy(self, xyz):
        total = self._bonded(xyz, True)
        if self._has_nonbonded:
            if self._pair_cache_enabled:
                total = total + self._cached_nonbonded(xyz)
            else:
                movable, fixed = self._buffers["ligand"], self._buffers["receptor"]
                total = total + self._nonbonded(xyz, movable, fixed, False)
                total = total + self._nonbonded(xyz, movable, movable, True)
        return total

    def _coordinates(self, coordinates):
        if not isinstance(coordinates, torch.Tensor) or coordinates.dtype not in (torch.float32, torch.float64):
            raise ValueError("ligand coordinates must be a float32/float64 tensor")
        if coordinates.ndim == 2:
            coordinates = coordinates[None]
        if (coordinates.ndim != 3 or coordinates.shape[0] < 1
                or coordinates.shape[1:] != (len(self.ligand_atom_indices), 3)
                or not bool(torch.isfinite(coordinates).all())):
            raise ValueError("ligand coordinates must be finite nonempty [B,N,3] matching every mapped atom")
        if coordinates.device != self.device:
            raise ValueError("ligand coordinates and prepared parameter device differ; call scorer.to(device) first")
        reference = self._buffers["positions"][None].expand(len(coordinates), -1, -1)
        return torch.index_copy(reference, 1, self._buffers["ligand"], coordinates.double())

    def _energy(self, coordinates, relative):
        value = self._moving_energy(self._coordinates(coordinates))
        value = value - self._reference_moving if relative else value + self._fixed_energy
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Torch AMBER returned nonfinite potential energy")
        return value

    def energy(self, coordinates):
        """Full supplied complex potential in FP64 kcal/mol."""
        return self._energy(coordinates, False)

    __call__ = energy

    def relative_energy(self, coordinates):
        """Pose energy minus input pose energy, excluding fixed constants exactly."""
        return self._energy(coordinates, True)

    def components(self, coordinates):
        return {"complex_potential": self.energy(coordinates)}

    @property
    def reference_energy(self):
        return self._fixed_energy + self._reference_moving

    @property
    def fingerprint(self):
        return self._fingerprint

    @property
    def metadata(self):
        return {"contract": TORCH_AMBER_CONTRACT, "objective": self.objective,
                "component_semantics": self.component_semantics, "units": self.units,
                "direction": self.direction, "runtime": "pure-torch", "compute_dtype": "float64",
                "device": str(self.device), "openmm_import_version": self._openmm_version,
                "force_counts": dict(self._force_counts), "source": deepcopy(self._source_metadata),
                "pair_cache": {"enabled": self._pair_cache_enabled,
                               "max_candidate_pairs": self.pair_cache_max_pairs,
                               "moving_candidate_pairs": self._moving_candidate_pairs,
                               "cached_active_pairs": self._cached_active_pairs,
                               "evaluation_pair_chunk_size": self.pair_chunk_size**2,
                               "bound_policy": "candidate count before exclusions; zero disables caching"},
                "reference_energy_kcal_mol": self.reference_energy, "fixed_energy_kcal_mol": self._fixed_energy,
                "search_reference": "input full-complex potential; fixed-fixed terms removed analytically",
                "constraints": "not enforced; supplied potential terms retained",
                "ligand_graph_validation": ("count/elements and induced topology edges; bond orders not validated"
                                             if self._ligand_edges is not None else
                                             "count and supplied elements only; no chemical topology available"),
                "atom_mapping": "explicit caller-supplied ordered map; never inferred",
                "singular_geometry": "reject active zero separations/collinear angles/degenerate torsions",
                "fingerprint": self.fingerprint, "fixed_receptor": True}

    def ligand_coordinates(self, *, device=None, dtype=torch.float32):
        return self._buffers["positions"][self._buffers["ligand"]].to(device=self.device if device is None else device, dtype=dtype).clone()

    def receptor_coordinates(self, *, device=None, dtype=torch.float32):
        return self._buffers["positions"][self._buffers["receptor"]].to(device=self.device if device is None else device, dtype=dtype).clone()

    def validate_ligand(self, mol):
        if mol.GetNumAtoms() != len(self.ligand_atom_indices):
            raise ValueError("AMBER ligand graph must include every mapped atom, including hydrogens")
        if self.atomic_numbers is not None:
            expected = tuple(self.atomic_numbers[i] for i in self.ligand_atom_indices)
            if tuple(atom.GetAtomicNum() for atom in mol.GetAtoms()) != expected:
                raise ValueError("AMBER ligand graph elements/order disagree with the explicit particle mapping")
        if self._ligand_edges is not None:
            edges = frozenset(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in mol.GetBonds())
            if edges != self._ligand_edges:
                raise ValueError("AMBER ligand graph connectivity disagrees with the mapped topology edges")

    def prepare(self, ligand_features, receptor_coords, receptor_features, *, num_rotatable_bonds=0,
                intramolecular_mask=None, intermolecular_exclusion_mask=None):
        del receptor_features, num_rotatable_bonds, intramolecular_mask
        if intermolecular_exclusion_mask is not None:
            raise ValueError("AMBER exclusions must be encoded in the parameterized System, not a docking mask")
        active = ligand_features.get("active")
        if not isinstance(active, torch.Tensor) or active.shape != (len(self.ligand_atom_indices),):
            raise ValueError("AMBER scoring requires all mapped ligand atoms, including hydrogens")
        expected = self.receptor_coordinates(device="cpu", dtype=torch.float64)
        if (not isinstance(receptor_coords, torch.Tensor) or receptor_coords.requires_grad
                or receptor_coords.shape != expected.shape or not bool(torch.isfinite(receptor_coords).all())
                or not torch.allclose(receptor_coords.detach().cpu().double(), expected, atol=1e-4, rtol=0.)):
            raise ValueError("receptor coordinates/order must match fixed nonligand particles in the AMBER System")
        return PreparedOpenMMScorer(self)
