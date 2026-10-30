"""Optional all-atom AMBER/OpenMM potential for fixed-receptor refinement.

The objective is the complete supplied complex's potential energy, including
its receptor constant. It is neither binding affinity nor MM/GBSA subtraction.
Only the explicitly indexed ligand particles move. Charges, Lennard-Jones,
bonded terms and exclusions come from the supplied OpenMM System/prmtop; PQR
charge/radius records alone do not define this force field.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from .scoring import RawScoreComponents, ScoreComponents

AMBER_SCORING_CONTRACT = "openmm-fixed-receptor-kcal-ang-relative-search-v1"


def _openmm():
    try:
        import openmm
        from openmm import app, unit
    except ImportError as error:
        raise ImportError("AMBER scoring requires the optional 'openmm' package; install anchor-dock[amber]") from error
    return openmm, app, unit


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _fixed_positions(value, count, unit):
    if hasattr(value, "value_in_unit"):
        value = value.value_in_unit(unit.angstrom)
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("reference positions must be fixed detached coordinates")
        value = value.detach().cpu().numpy()
    positions = np.array(value, dtype=np.float64, copy=True)
    if positions.shape != (count, 3) or not np.isfinite(positions).all():
        raise ValueError("reference positions must be finite [system particles,3] in angstroms")
    positions.flags.writeable = False
    return positions


class _OpenMMPotential(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coordinates, owner, subtract_reference):
        energy, gradient = owner._evaluate(coordinates.detach().cpu().double().numpy())
        if subtract_reference:
            # The receptor constant may dwarf pose changes. Subtract while
            # still in native FP64, before the caller's possible FP32 cast.
            energy = energy - owner.reference_energy
        gradient = torch.as_tensor(gradient, dtype=coordinates.dtype, device=coordinates.device)
        result = torch.as_tensor(energy, dtype=coordinates.dtype, device=coordinates.device)
        if not torch.isfinite(gradient).all() or not torch.isfinite(result).all():
            raise FloatingPointError("OpenMM energy/gradient is nonfinite in the requested Torch dtype")
        ctx.save_for_backward(gradient)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, energy_gradient):
        gradient, = ctx.saved_tensors
        return energy_gradient[:, None, None] * gradient, None, None


class OpenMMScorer:
    """Optional differentiable full-complex energy with explicit particle map.

    ``ligand_atom_indices`` is the ordered correspondence from ligand graph
    atoms to particles in the combined, parameterized System. All ligand atoms,
    including H, must be supplied. Remaining particles stay at the supplied
    reference positions, in original System order. No atom mapping is guessed.

    Torch coordinates are angstroms and outputs are kcal/mol. OpenMM computes
    the forces; the bridge returns dE/dx=-force in kcal/mol/angstrom. Only first
    derivatives are supported. Torch CUDA tensors can be transferred to the
    selected OpenMM platform, but this does not imply a fused GPU scorer.
    """

    name = "amber-mm"
    units = "kcal/mol"
    direction = "lower"
    differentiable = True
    requires_all_atoms = True
    preparation_requirement = "amber-topology"
    fixed_receptor = True
    objective = "full_complex_potential"
    component_semantics = "inter/intra unavailable; search is complex minus fixed reference; score is full complex potential"

    def __init__(self, system, positions, ligand_atom_indices: Sequence[int], *,
                 atomic_numbers: Sequence[int] | None = None, platform: str = "Reference",
                 platform_properties: Mapping[str, str] | None = None, source_metadata: Mapping | None = None):
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
        indices = tuple(int(i) for i in indices)
        if atomic_numbers is not None:
            atomic_numbers = tuple(atomic_numbers)
            if len(atomic_numbers) != count or any(
                isinstance(z, bool) or not isinstance(z, Integral) or not 1 <= z <= 118 for z in atomic_numbers
            ):
                raise ValueError("atomic_numbers must contain one valid element per System particle")
            atomic_numbers = tuple(int(z) for z in atomic_numbers)
        if not isinstance(platform, str) or not platform:
            raise ValueError("platform must name an installed OpenMM platform")
        selected_platform = openmm.Platform.getPlatformByName(platform)
        properties = dict(platform_properties or {})
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in properties.items()):
            raise ValueError("OpenMM platform properties must be string names and values")
        if set(properties) - set(selected_platform.getPropertyNames()):
            raise ValueError("unsupported OpenMM platform property")
        self._system_xml = openmm.XmlSerializer.serialize(system)
        self._system = openmm.XmlSerializer.deserialize(self._system_xml)
        self._positions = _fixed_positions(positions, count, unit)
        self.ligand_atom_indices = indices
        self.receptor_atom_indices = tuple(i for i in range(count) if i not in set(indices))
        self.atomic_numbers = atomic_numbers
        self.platform = platform
        self._openmm_version = openmm.__version__
        self._platform_properties = properties
        self._source_metadata = deepcopy(dict(source_metadata or {}))
        # Ensure caller-owned metadata cannot mutate identity or hide opaque state.
        json.dumps(self._source_metadata, sort_keys=True, allow_nan=False)
        self._lock = threading.RLock()
        self._context = self._integrator = None
        self._reference_energy = None
        self._closed = False
        identity = {
            "contract": AMBER_SCORING_CONTRACT, "openmm_version": openmm.__version__,
            "search_reference": "input full-complex potential",
            "platform": platform, "platform_properties": properties, "ligand_atom_indices": indices,
            "atomic_numbers": atomic_numbers, "source": self._source_metadata,
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
        digest.update(self._system_xml.encode())
        digest.update(self._positions.tobytes())
        self._fingerprint = "sha256:" + digest.hexdigest()

    @classmethod
    def from_system(cls, system, positions, ligand_atom_indices, **kwargs):
        """Snapshot an explicitly parameterized System and angstrom coordinates."""
        return cls(system, positions, ligand_atom_indices, **kwargs)

    @classmethod
    def from_amber(cls, prmtop: str | Path, inpcrd: str | Path, ligand_atom_indices, *,
                   implicit_solvent: str | None = None, nonbonded_method: str = "NoCutoff",
                   cutoff_angstrom: float = 10., solute_dielectric: float = 1., solvent_dielectric: float = 78.5,
                   platform: str = "Reference", platform_properties: Mapping[str, str] | None = None):
        """Load full AMBER parameters/coordinates; no preparation is inferred.

        Supported implicit models are HCT, OBC1, OBC2, GBn and GBn2. This is a
        single full-complex potential, even with implicit solvent; it does not
        subtract isolated receptor/ligand energies or estimate entropy.
        """
        _, app, unit = _openmm()
        methods = {name: getattr(app, name) for name in ("NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "Ewald", "PME", "LJPME")}
        models = {name: getattr(app, name) for name in ("HCT", "OBC1", "OBC2", "GBn", "GBn2")}
        if nonbonded_method not in methods:
            raise ValueError(f"nonbonded_method must be one of {tuple(methods)}")
        if implicit_solvent is not None and implicit_solvent not in models:
            raise ValueError(f"implicit_solvent must be None or one of {tuple(models)}")
        cutoff = _positive(cutoff_angstrom, "cutoff_angstrom")
        solute = _positive(solute_dielectric, "solute_dielectric")
        solvent = _positive(solvent_dielectric, "solvent_dielectric")
        prmtop, inpcrd = Path(prmtop), Path(inpcrd)
        parameter_bytes, coordinate_bytes = prmtop.read_bytes(), inpcrd.read_bytes()
        coordinates = app.AmberInpcrdFile(str(inpcrd))
        topology = app.AmberPrmtopFile(str(prmtop), periodicBoxVectors=coordinates.boxVectors)
        system = topology.createSystem(
            nonbondedMethod=methods[nonbonded_method], nonbondedCutoff=cutoff * unit.angstrom,
            constraints=None, rigidWater=False, removeCMMotion=False,
            implicitSolvent=models.get(implicit_solvent), soluteDielectric=solute, solventDielectric=solvent,
        )
        atoms = tuple(topology.topology.atoms())
        if any(atom.element is None for atom in atoms):
            raise ValueError("AMBER topology must provide explicit physical atom elements")
        metadata = {
            "prmtop_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
            "inpcrd_sha256": hashlib.sha256(coordinate_bytes).hexdigest(),
            "implicit_solvent": implicit_solvent, "nonbonded_method": nonbonded_method,
            "cutoff_angstrom": cutoff, "solute_dielectric": solute, "solvent_dielectric": solvent,
        }
        if parameter_bytes != prmtop.read_bytes() or coordinate_bytes != inpcrd.read_bytes():
            raise ValueError("AMBER parameter/coordinate file changed during preparation")
        result = cls(system, coordinates.positions, ligand_atom_indices,
                     atomic_numbers=[atom.element.atomic_number for atom in atoms], platform=platform,
                     platform_properties=platform_properties, source_metadata=metadata)
        selected = set(result.ligand_atom_indices)
        for left, right in topology.topology.bonds():
            if ((left.index in selected) != (right.index in selected)
                    and (left.element.atomic_number == 1 or right.element.atomic_number == 1)):
                result.close()
                raise ValueError("AMBER ligand mapping must include bonded hydrogens with their parent atoms")
        return result

    @property
    def fingerprint(self):
        return self._fingerprint

    @property
    def metadata(self):
        return {"contract": AMBER_SCORING_CONTRACT, "objective": self.objective,
                "openmm_version": self._openmm_version, "system_particles": self._system.getNumParticles(),
                "periodic": self._system.usesPeriodicBoundaryConditions(),
                "units": self.units, "direction": self.direction, "platform": self.platform,
                "platform_properties": dict(self._platform_properties),
                "source": deepcopy(self._source_metadata), "component_semantics": self.component_semantics,
                "search_reference": "input full-complex potential",
                "reference_energy_kcal_mol": self._reference_energy,
                "ligand_atom_indices": list(self.ligand_atom_indices), "fingerprint": self.fingerprint}

    def ligand_coordinates(self, device="cpu", *, dtype=torch.float32):
        return torch.tensor(self._positions[list(self.ligand_atom_indices)], dtype=dtype, device=device)

    def receptor_coordinates(self, device="cpu", *, dtype=torch.float32):
        return torch.tensor(self._positions[list(self.receptor_atom_indices)], dtype=dtype, device=device)

    def validate_ligand(self, mol):
        """Check graph count/elements against the explicit particle ordering."""
        if mol.GetNumAtoms() != len(self.ligand_atom_indices):
            raise ValueError("AMBER ligand graph must include every mapped atom, including hydrogens")
        if self.atomic_numbers is not None:
            expected = tuple(self.atomic_numbers[i] for i in self.ligand_atom_indices)
            if tuple(atom.GetAtomicNum() for atom in mol.GetAtoms()) != expected:
                raise ValueError("AMBER ligand graph elements/order disagree with the explicit particle mapping")

    def _ensure_context(self):
        if self._closed:
            raise RuntimeError("OpenMM scorer is closed")
        if self._context is None:
            openmm, _, unit = _openmm()
            self._integrator = openmm.VerletIntegrator(.001 * unit.picosecond)
            self._context = openmm.Context(self._system, self._integrator,
                                           openmm.Platform.getPlatformByName(self.platform), self._platform_properties)

    def _evaluate(self, ligand_coordinates):
        _, _, unit = _openmm()
        energy, gradients = [], []
        with self._lock:
            self._ensure_context()
            try:
                for pose in ligand_coordinates:
                    positions = self._positions.copy()
                    positions[list(self.ligand_atom_indices)] = pose
                    self._context.setPositions(positions * unit.angstrom)
                    state = self._context.getState(getEnergy=True, getForces=True)
                    value = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
                    force = state.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)
                    gradient = -np.asarray(force)[list(self.ligand_atom_indices)]
                    if not math.isfinite(value) or not np.isfinite(gradient).all():
                        raise FloatingPointError("OpenMM returned nonfinite potential energy or ligand forces")
                    energy.append(value)
                    gradients.append(gradient.copy())
            finally:
                # Each call starts from the same receptor state even after an
                # exception; no candidate pose becomes a hidden reference.
                self._context.setPositions(self._positions * unit.angstrom)
        return np.asarray(energy), np.asarray(gradients)

    @property
    def reference_energy(self):
        """Fixed full-complex input energy, not isolated receptor/ligand subtraction."""
        with self._lock:
            if self._reference_energy is None:
                value, _ = self._evaluate(self._positions[list(self.ligand_atom_indices)][None])
                self._reference_energy = float(value[0])
            return self._reference_energy

    def energy(self, ligand_coordinates: torch.Tensor):
        """Return full-complex potential per pose; first coordinate derivative only."""
        return self._energy(ligand_coordinates, False)

    def relative_energy(self, ligand_coordinates: torch.Tensor):
        """Search energy relative to the fixed input complex, with the same force."""
        return self._energy(ligand_coordinates, True)

    def _energy(self, ligand_coordinates, subtract_reference):
        if not isinstance(ligand_coordinates, torch.Tensor) or ligand_coordinates.dtype not in (torch.float32, torch.float64):
            raise ValueError("ligand coordinates must be a float32/float64 tensor")
        if ligand_coordinates.ndim == 2:
            ligand_coordinates = ligand_coordinates.unsqueeze(0)
        if (ligand_coordinates.ndim != 3 or ligand_coordinates.shape[0] < 1
                or ligand_coordinates.shape[1:] != (len(self.ligand_atom_indices), 3)
                or not torch.isfinite(ligand_coordinates).all()):
            raise ValueError("ligand coordinates must be finite nonempty [B,N,3] matching every mapped atom")
        return _OpenMMPotential.apply(ligand_coordinates, self, subtract_reference)

    __call__ = energy

    def components(self, coordinates):
        # Reporting keeps a large full-system constant in FP64 even when the
        # pose optimizer itself uses FP32 coordinates.
        return {"complex_potential": self.energy(coordinates.to(dtype=torch.float64))}

    def prepare(self, ligand_features, receptor_coords, receptor_features, *, num_rotatable_bonds=0,
                intramolecular_mask=None, intermolecular_exclusion_mask=None):
        del receptor_features, num_rotatable_bonds, intramolecular_mask
        if intermolecular_exclusion_mask is not None:
            raise ValueError("AMBER exclusions must be encoded in the parameterized System, not a docking mask")
        active = ligand_features.get("active")
        if not isinstance(active, torch.Tensor) or active.shape != (len(self.ligand_atom_indices),):
            raise ValueError("AMBER scoring requires all mapped ligand atoms, including hydrogens")
        expected = self.receptor_coordinates(dtype=torch.float64)
        if (not isinstance(receptor_coords, torch.Tensor) or receptor_coords.requires_grad
                or receptor_coords.shape != expected.shape or not torch.isfinite(receptor_coords).all()
                or not torch.allclose(receptor_coords.detach().cpu().double(), expected, atol=1e-4, rtol=0.)):
            raise ValueError("receptor coordinates/order must match fixed nonligand particles in the AMBER System")
        return PreparedOpenMMScorer(self)

    def close(self):
        with self._lock:
            self._context = self._integrator = None
            self._closed = True

    def __del__(self):
        # Partially constructed instances may fail before the lock exists.
        if hasattr(self, "_lock"):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass
class PreparedOpenMMScorer:
    factory: OpenMMScorer
    component_semantics = "undecomposed-full-complex-potential"

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
    def metadata(self):
        return self.factory.metadata

    @property
    def effective_rotatable_bonds(self):
        return 0

    @property
    def torsion_penalty_applied(self):
        return False

    def search_energy(self, coordinates):
        return self.factory.relative_energy(coordinates)

    def components(self, coordinates):
        return self.factory.components(coordinates)

    def raw_components(self, coordinates):
        value = self.search_energy(coordinates)
        zero = torch.zeros_like(value)
        # Zero is an unavailable-decomposition placeholder, not a physical
        # assertion. Named components()/metadata expose the actual objective.
        return RawScoreComponents(zero, zero, value)

    def report(self, raw, intramolecular_reference=None):
        del intramolecular_reference
        zero = raw.search_energy.new_zeros(())
        full_potential = raw.search_energy.double() + self.factory.reference_energy
        return ScoreComponents(raw.intermolecular, raw.intramolecular, raw.search_energy, full_potential, zero)

    def score_components(self, coordinates, intramolecular_reference=None):
        return self.report(self.raw_components(coordinates), intramolecular_reference)
