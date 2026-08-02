"""Differentiable, scorer-independent pairwise docking energies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import partial
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    CodeType,
    GetSetDescriptorType,
    MemberDescriptorType,
    ModuleType,
)
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
    return torch.where(
        value <= low,
        torch.tensor(float(good == low), device=value.device, dtype=value.dtype),
        torch.where(value >= high, torch.tensor(float(good == high), device=value.device, dtype=value.dtype), fraction),
    )


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
    hydrophobic = hydrophobic_match * _slope_step(config.hydrophobic_bad, config.hydrophobic_good, surface_distance)
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
    fingerprint: str
    effective_rotatable_bonds: int
    torsion_penalty_applied: bool

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

    @property
    def fingerprint(self) -> str:
        return scoring_config_fingerprint(self.config)

    @property
    def effective_rotatable_bonds(self) -> int:
        if self.config.formula != "vina" or self.config.torsion_weight <= 0:
            return 0
        return self.num_rotatable_bonds

    @property
    def torsion_penalty_applied(self) -> bool:
        return self.effective_rotatable_bonds > 0

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
        self.fingerprint = scoring_config_fingerprint(config)

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
        if self.config.formula == "vina":
            unknown_ligand = [
                index for index, atom_type in enumerate(ligand_features.get("xs_types", ())) if atom_type == "X"
            ]
            unknown_receptor = [
                index for index, atom_type in enumerate(receptor_features.get("xs_types", ())) if atom_type == "X"
            ]
            if unknown_ligand or unknown_receptor:
                raise ValueError(
                    "Vina/Vinardo scoring cannot assign validated XS-like types to "
                    f"ligand atoms {unknown_ligand} or receptor atoms {unknown_receptor}; "
                    "use a calibrated custom scorer instead"
                )
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

    def __init__(
        self,
        module: nn.Module,
        *,
        name: str = "neural",
        units: str = "arbitrary",
        fingerprint: str | None = None,
    ):
        self.module = module
        self.name = name
        self.units = units
        self._explicit_fingerprint = fingerprint

    @property
    def fingerprint(self) -> str:
        return self._explicit_fingerprint or neural_module_fingerprint(self.module)

    def prepare(
        self,
        ligand_features: dict[str, object],
        receptor_coords: torch.Tensor,
        receptor_features: dict[str, object],
        **_: object,
    ) -> PreparedNeuralScorer:
        target_device = receptor_coords.device
        module_devices = {value.device for value in (*self.module.parameters(), *self.module.buffers())}
        if module_devices and module_devices != {target_device}:
            raise ValueError(
                f"custom scorer parameters and buffers must be on {target_device}; "
                f"found {sorted(map(str, module_devices))}"
            )
        return PreparedNeuralScorer(
            self.module,
            self.name,
            self.units,
            self.fingerprint,
            receptor_coords,
            ligand_features,
            receptor_features,
        )


@dataclass
class PreparedNeuralScorer:
    module: nn.Module
    name: str
    units: str
    fingerprint: str
    receptor_coords: torch.Tensor
    ligand_features: dict[str, object]
    receptor_features: dict[str, object]

    @property
    def effective_rotatable_bonds(self) -> int:
        return 0

    @property
    def torsion_penalty_applied(self) -> bool:
        return False

    def search_energy(self, ligand_coords: torch.Tensor) -> torch.Tensor:
        if ligand_coords.ndim == 2:
            ligand_coords = ligand_coords.unsqueeze(0)
        training_states = [(child, child.training) for child in self.module.modules()]
        try:
            self.module.eval()
            values = self.module(
                ligand_coords,
                self.receptor_coords,
                self.ligand_features,
                self.receptor_features,
            )
        finally:
            for child, was_training in training_states:
                child.training = was_training
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


def scoring_config_fingerprint(config: ScoringConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


_UNSERIALIZABLE = object()
_MODULE_INTERNAL_ATTRIBUTES = {
    "_backward_hooks",
    "_backward_pre_hooks",
    "_buffers",
    "_forward_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_is_full_backward_hook",
    "_load_state_dict_post_hooks",
    "_load_state_dict_pre_hooks",
    "_modules",
    "_non_persistent_buffers_set",
    "_parameters",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "training",
}
_BEHAVIOR_HOOK_REGISTRIES = ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")
_GLOBAL_BEHAVIOR_HOOK_REGISTRIES = (
    "_global_forward_pre_hooks",
    "_global_forward_hooks",
    "_global_backward_pre_hooks",
    "_global_backward_hooks",
)


def _tensor_identity(value: torch.Tensor) -> dict[str, object]:
    """Return a device-independent identity for tensor-backed scorer state."""
    tensor = value.detach()
    result: dict[str, object] = {
        "dtype": str(tensor.dtype),
        "layout": str(tensor.layout),
        "shape": list(tensor.shape),
    }
    if tensor.device.type == "meta":
        result["content"] = "meta"
        return result
    if tensor.layout != torch.strided:
        tensor = tensor.to_dense()
    if tensor.is_quantized:
        tensor = tensor.dequantize()
    tensor = tensor.resolve_conj().resolve_neg().contiguous().cpu()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    result["content_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def _stable_module_attribute(value: object, seen: set[int] | None = None) -> object:
    """Convert plain scorer configuration into deterministic JSON data."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {"torch_tensor": _tensor_identity(value)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"bytes_sha256": hashlib.sha256(raw).hexdigest(), "length": len(raw)}
    if isinstance(value, range):
        return {"range": [value.start, value.stop, value.step]}
    if isinstance(value, slice):
        return {
            "slice": [
                _stable_module_attribute(value.start),
                _stable_module_attribute(value.stop),
                _stable_module_attribute(value.step),
            ]
        }

    seen = set() if seen is None else seen
    object_id = id(value)
    if object_id in seen:
        return _UNSERIALIZABLE
    seen.add(object_id)
    try:
        if isinstance(value, list):
            items = [_stable_module_attribute(item, seen) for item in value]
            return _UNSERIALIZABLE if any(item is _UNSERIALIZABLE for item in items) else items
        if isinstance(value, tuple):
            items = [_stable_module_attribute(item, seen) for item in value]
            if any(item is _UNSERIALIZABLE for item in items):
                return _UNSERIALIZABLE
            return {"tuple": items}
        if isinstance(value, dict):
            items: list[list[object]] = []
            for key, item in value.items():
                stable_key = _stable_module_attribute(key, seen)
                stable_value = _stable_module_attribute(item, seen)
                if stable_key is _UNSERIALIZABLE or stable_value is _UNSERIALIZABLE:
                    return _UNSERIALIZABLE
                items.append([stable_key, stable_value])
            items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":")))
            return {"mapping": items}
        if isinstance(value, (set, frozenset)):
            items = [_stable_module_attribute(item, seen) for item in value]
            if any(item is _UNSERIALIZABLE for item in items):
                return _UNSERIALIZABLE
            items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
            return {"frozenset" if isinstance(value, frozenset) else "set": items}
        if type(value).__module__.startswith("numpy"):
            if all(hasattr(value, name) for name in ("dtype", "shape", "tobytes")):
                if getattr(value.dtype, "hasobject", False):
                    stable_values = _stable_module_attribute(value.tolist(), seen)
                    if stable_values is _UNSERIALIZABLE:
                        return _UNSERIALIZABLE
                    return {
                        "numpy_object_array": {
                            "dtype": str(value.dtype),
                            "shape": list(value.shape),
                            "values": stable_values,
                        }
                    }
                raw = value.tobytes(order="C")
                return {
                    "numpy_array": {
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "content_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                }
            if hasattr(value, "item"):
                try:
                    return _stable_module_attribute(value.item(), seen)
                except ValueError:
                    return _UNSERIALIZABLE
    finally:
        seen.remove(object_id)
    return _UNSERIALIZABLE


def _code_identity(code: CodeType) -> dict[str, object]:
    constants: list[object] = []
    for value in code.co_consts:
        if isinstance(value, CodeType):
            constants.append({"code": _code_identity(value)})
            continue
        stable = _stable_module_attribute(value)
        constants.append(
            stable
            if stable is not _UNSERIALIZABLE
            else {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
        )
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
        "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "constants": constants,
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _global_reference_identity(
    name: str,
    value: object,
    seen: set[int],
    covered_module_ids: frozenset[int],
    reference_module: str | None = None,
) -> object:
    if isinstance(value, ModuleType):
        return {
            "module": value.__name__,
            "version": getattr(value, "__version__", None),
        }
    if isinstance(value, type):
        identity: dict[str, object] = {"class": f"{value.__module__}.{value.__qualname__}"}
        # Walk mutable class behavior only for types defined alongside the
        # referring function. Dependency/framework types are identified by
        # their qualified name; recursively traversing their implementation
        # drags in opaque C descriptors and makes ordinary nn.Modules unusable.
        if value.__module__ != "builtins" and value.__module__ == reference_module:
            stop_types = frozenset({nn.Module, object}) if issubclass(value, nn.Module) else frozenset({object})
            identity["methods"] = _callable_type_method_identities(
                value,
                seen,
                covered_module_ids,
                stop_types=stop_types,
            )
            identity["class_attributes"] = _type_class_attribute_identities(
                value,
                stop_types,
            )
        return identity
    if callable(value):
        return {
            "callable": _callable_implementation_identity(
                value,
                seen,
                covered_module_ids=covered_module_ids,
            )
        }
    stable = _stable_module_attribute(value)
    if stable is not _UNSERIALIZABLE:
        return stable
    raise ValueError(
        f"automatic custom-scorer fingerprinting cannot safely identify global {name!r} "
        f"of type {type(value).__module__}.{type(value).__qualname__}; "
        "use NeuralScorerAdapter(..., fingerprint='...')"
    )


def _require_stable_callable_state(value: object, description: str) -> object:
    stable = _stable_module_attribute(value)
    if stable is _UNSERIALIZABLE:
        raise ValueError(
            f"automatic custom-scorer fingerprinting cannot safely identify {description}; "
            "use NeuralScorerAdapter(..., fingerprint='...')"
        )
    return stable


def _callable_object_state(value: object) -> object:
    """Return deterministic Python state without invoking user serialization hooks."""
    state: dict[str, object] = {}
    try:
        state.update(vars(value))
    except TypeError:
        pass

    for owner in reversed(type(value).__mro__):
        raw_slots = vars(owner).get("__slots__", ())
        slots = (raw_slots,) if isinstance(raw_slots, str) else tuple(raw_slots)
        for slot in slots:
            if slot in {"__dict__", "__weakref__"}:
                continue
            attribute_name = slot
            if slot.startswith("__") and not slot.endswith("__"):
                attribute_name = f"_{owner.__name__.lstrip('_')}{slot}"
            try:
                slot_value = getattr(value, attribute_name)
            except AttributeError:
                continue
            except Exception as exc:
                raise ValueError(
                    "automatic custom-scorer fingerprinting could not inspect callable-object "
                    f"slot {owner.__module__}.{owner.__qualname__}.{slot}; "
                    "use NeuralScorerAdapter(..., fingerprint='...')"
                ) from exc
            state[f"{owner.__module__}.{owner.__qualname__}.{slot}"] = slot_value
    return _require_stable_callable_state(
        state,
        f"callable-object state for {type(value).__module__}.{type(value).__qualname__}",
    )


def _callable_type_method_identities(
    owner_type: type,
    seen: set[int],
    covered_module_ids: frozenset[int],
    *,
    stop_types: frozenset[type] = frozenset({object}),
) -> dict[str, object]:
    """Fingerprint Python behavior reachable through a callable's class."""
    result: dict[str, object] = {}
    for owner in owner_type.__mro__:
        if owner in stop_types:
            break
        methods: dict[str, object] = {}
        for name, value in sorted(vars(owner).items()):
            if isinstance(value, (staticmethod, classmethod)):
                methods[name] = _callable_implementation_identity(
                    value.__func__,
                    seen,
                    covered_module_ids=covered_module_ids,
                )
            elif isinstance(value, property):
                accessors = {
                    accessor_name: _callable_implementation_identity(
                        accessor,
                        seen,
                        covered_module_ids=covered_module_ids,
                    )
                    for accessor_name, accessor in (("get", value.fget), ("set", value.fset), ("delete", value.fdel))
                    if accessor is not None
                }
                if accessors:
                    methods[name] = {"property": accessors}
            elif callable(value):
                methods[name] = _callable_implementation_identity(
                    value,
                    seen,
                    covered_module_ids=covered_module_ids,
                )
        result[f"{owner.__module__}.{owner.__qualname__}"] = methods
    return result


def _type_class_attribute_identities(
    owner_type: type,
    stop_types: frozenset[type],
) -> dict[str, object]:
    """Fingerprint stable non-method class state that behavior may read."""
    result: dict[str, object] = {}
    for owner in owner_type.__mro__:
        if owner in stop_types:
            break
        attributes: dict[str, object] = {}
        for name, value in sorted(vars(owner).items()):
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(
                value,
                (staticmethod, classmethod, property, MemberDescriptorType, GetSetDescriptorType),
            ) or callable(value):
                continue
            stable = _stable_module_attribute(value)
            if stable is _UNSERIALIZABLE:
                raise ValueError(
                    "automatic custom-scorer fingerprinting cannot safely identify class attribute "
                    f"{owner.__module__}.{owner.__qualname__}.{name}; "
                    "use NeuralScorerAdapter(..., fingerprint='...')"
                )
            attributes[name] = stable
        result[f"{owner.__module__}.{owner.__qualname__}"] = attributes
    return result


def _callable_implementation_identity(
    value: object,
    seen: set[int] | None = None,
    *,
    covered_module_ids: frozenset[int] = frozenset(),
) -> dict[str, object]:
    python_bound_self = getattr(value, "__self__", None)
    function = getattr(value, "__func__", value)
    result: dict[str, object] = {
        "type": f"{type(function).__module__}.{type(function).__qualname__}",
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", None),
    }
    seen = set() if seen is None else seen
    function_id = id(function)
    if function_id in seen:
        result["recursive_reference"] = True
        return result
    seen.add(function_id)
    code = getattr(function, "__code__", None)
    try:
        if function is not value and python_bound_self is not None:
            if isinstance(python_bound_self, nn.Module):
                if id(python_bound_self) not in covered_module_ids:
                    raise ValueError(
                        "automatic custom-scorer fingerprinting cannot safely identify an unregistered "
                        f"nn.Module bound to callable {type(python_bound_self).__module__}."
                        f"{type(python_bound_self).__qualname__}; register it as a child module or use "
                        "NeuralScorerAdapter(..., fingerprint='...')"
                    )
                # ``child.forward`` and registered module-local bound helpers
                # are covered by the surrounding named-module code/state walk;
                # recursively serializing the module here would cycle.
                result["bound_self"] = {
                    "module_state": "fingerprinted_by_named_module_walk",
                    "type": f"{type(python_bound_self).__module__}.{type(python_bound_self).__qualname__}",
                }
            elif isinstance(python_bound_self, type):
                result["bound_self"] = {"class": f"{python_bound_self.__module__}.{python_bound_self.__qualname__}"}
            else:
                result["bound_self"] = {
                    "type": f"{type(python_bound_self).__module__}.{type(python_bound_self).__qualname__}",
                    "state": _callable_object_state(python_bound_self),
                    "methods": _callable_type_method_identities(
                        type(python_bound_self),
                        seen,
                        covered_module_ids,
                    ),
                    "class_attributes": _type_class_attribute_identities(
                        type(python_bound_self),
                        frozenset({object}),
                    ),
                }
        if isinstance(function, partial):
            result["partial"] = {
                "func": _callable_implementation_identity(
                    function.func,
                    seen,
                    covered_module_ids=covered_module_ids,
                ),
                "args": _require_stable_callable_state(function.args, "functools.partial positional arguments"),
                "keywords": _require_stable_callable_state(
                    function.keywords or {},
                    "functools.partial keyword arguments",
                ),
                "attributes": _require_stable_callable_state(
                    vars(function),
                    "functools.partial instance attributes",
                ),
            }
            return result
        if isinstance(code, CodeType):
            result["code"] = _code_identity(code)
            function_globals = getattr(function, "__globals__", {})
            result["globals"] = {
                name: _global_reference_identity(
                    name,
                    function_globals[name],
                    seen,
                    covered_module_ids,
                    getattr(function, "__module__", None),
                )
                for name in sorted(set(code.co_names))
                if name in function_globals
            }
        for name, default_value in (
            ("defaults", getattr(function, "__defaults__", None)),
            ("kwdefaults", getattr(function, "__kwdefaults__", None)),
        ):
            stable = _stable_module_attribute(default_value)
            if stable is _UNSERIALIZABLE:
                raise ValueError(
                    f"automatic custom-scorer fingerprinting cannot safely identify callable {name}; "
                    "use NeuralScorerAdapter(..., fingerprint='...')"
                )
            result[name] = stable
        closure = getattr(function, "__closure__", None)
        if closure:
            closure_values: list[object] = []
            for index, cell in enumerate(closure):
                try:
                    closure_value = cell.cell_contents
                except ValueError:
                    closure_values.append({"empty_cell": True})
                    continue
                closure_values.append(
                    _global_reference_identity(
                        f"<closure:{index}>",
                        closure_value,
                        seen,
                        covered_module_ids,
                        getattr(function, "__module__", None),
                    )
                )
            result["closure"] = closure_values
        if isinstance(function, (BuiltinFunctionType, BuiltinMethodType)):
            bound_self = getattr(function, "__self__", None)
            if bound_self is not None:
                result["bound_self"] = _global_reference_identity(
                    "<builtin-bound-self>",
                    bound_self,
                    seen,
                    covered_module_ids,
                    getattr(function, "__module__", None),
                )
        elif not isinstance(code, CodeType):
            call_method = getattr(type(function), "__call__", None)
            call_function = getattr(call_method, "__func__", call_method)
            if not isinstance(getattr(call_function, "__code__", None), CodeType):
                raise ValueError(
                    "automatic custom-scorer fingerprinting cannot inspect callable object "
                    f"{type(function).__module__}.{type(function).__qualname__}; "
                    "use NeuralScorerAdapter(..., fingerprint='...')"
                )
            result["callable_object"] = {
                "call": _callable_implementation_identity(
                    call_method,
                    seen,
                    covered_module_ids=covered_module_ids,
                ),
                "state": _callable_object_state(function),
                "methods": _callable_type_method_identities(
                    type(function),
                    seen,
                    covered_module_ids,
                ),
                "class_attributes": _type_class_attribute_identities(
                    type(function),
                    frozenset({object}),
                ),
            }
        return result
    finally:
        seen.remove(function_id)


def _class_method_identities(
    module: nn.Module,
    covered_module_ids: frozenset[int],
) -> dict[str, object]:
    """Fingerprint custom callables that ``forward`` may dispatch through."""
    result: dict[str, object] = {}
    for owner in type(module).__mro__:
        if owner in {nn.Module, object}:
            break
        owner_methods: dict[str, object] = {}
        for name, value in sorted(vars(owner).items()):
            if isinstance(value, (staticmethod, classmethod)):
                owner_methods[name] = _callable_implementation_identity(
                    value.__func__,
                    covered_module_ids=covered_module_ids,
                )
            elif isinstance(value, property):
                accessors = {
                    accessor_name: _callable_implementation_identity(
                        accessor,
                        covered_module_ids=covered_module_ids,
                    )
                    for accessor_name, accessor in (("get", value.fget), ("set", value.fset), ("delete", value.fdel))
                    if accessor is not None
                }
                if accessors:
                    owner_methods[name] = {"property": accessors}
            elif callable(value):
                owner_methods[name] = _callable_implementation_identity(
                    value,
                    covered_module_ids=covered_module_ids,
                )
        result[f"{owner.__module__}.{owner.__qualname__}"] = owner_methods
    return result


def neural_module_fingerprint(module: nn.Module) -> str:
    named_modules = list(module.named_modules())
    covered_module_ids = frozenset(id(child) for _, child in named_modules)
    hooked_modules = [
        module_name or "<root>"
        for module_name, child in named_modules
        if any(bool(getattr(child, registry, None)) for registry in _BEHAVIOR_HOOK_REGISTRIES)
    ]
    global_hooks = [
        registry for registry in _GLOBAL_BEHAVIOR_HOOK_REGISTRIES if bool(getattr(nn.modules.module, registry, None))
    ]
    if hooked_modules or global_hooks:
        details = []
        if hooked_modules:
            details.append(f"module hooks on {hooked_modules}")
        if global_hooks:
            details.append(f"global hooks in {global_hooks}")
        raise ValueError(
            "automatic custom-scorer fingerprinting cannot safely identify behavior with "
            + " and ".join(details)
            + "; wrap the module in NeuralScorerAdapter(..., fingerprint='...')"
        )
    digest = hashlib.sha256()
    digest.update(f"{type(module).__module__}.{type(module).__qualname__}".encode())
    module_config: dict[str, object] = {}
    for module_name, child in named_modules:
        attributes: dict[str, object] = {}
        instance_callables: dict[str, object] = {}
        for name, value in sorted(vars(child).items()):
            if name in _MODULE_INTERNAL_ATTRIBUTES:
                continue
            if callable(value):
                instance_callables[name] = _callable_implementation_identity(
                    value,
                    covered_module_ids=covered_module_ids,
                )
                continue
            stable = _stable_module_attribute(value)
            if stable is not _UNSERIALIZABLE:
                attributes[name] = stable
        registered_parameters = {
            name: None if value is None else _tensor_identity(value)
            for name, value in sorted(child._parameters.items())
        }
        registered_buffers = {
            name: {
                "persistent": name not in child._non_persistent_buffers_set,
                "value": None if value is None else _tensor_identity(value),
            }
            for name, value in sorted(child._buffers.items())
        }
        module_config[module_name] = {
            "type": f"{type(child).__module__}.{type(child).__qualname__}",
            "forward": _callable_implementation_identity(
                child.forward,
                covered_module_ids=covered_module_ids,
            ),
            "methods": _class_method_identities(child, covered_module_ids),
            "class_attributes": _type_class_attribute_identities(
                type(child),
                frozenset({nn.Module, object}),
            ),
            "instance_callables": instance_callables,
            "extra_repr": child.extra_repr(),
            "attributes": attributes,
            "registered_parameters": registered_parameters,
            "registered_buffers": registered_buffers,
        }
    digest.update(json.dumps(module_config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        if not isinstance(value, torch.Tensor):
            stable = _stable_module_attribute(value)
            serializable = (
                stable
                if stable is not _UNSERIALIZABLE
                else {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
            )
            digest.update(json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            continue
        digest.update(json.dumps(_tensor_identity(value), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    # state_dict intentionally omits non-persistent registered buffers, but
    # scorers may still read them in forward(). Include every live buffer.
    for name, value in sorted(module.named_buffers()):
        digest.update(b"buffer\0" + name.encode("utf-8") + b"\0")
        digest.update(json.dumps(_tensor_identity(value), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def scorer_fingerprint(scorer: object) -> str:
    if isinstance(scorer, str):
        try:
            return scoring_config_fingerprint(SCORING_CONFIGS[scorer.lower()])
        except KeyError:
            return f"unresolved:{scorer}"
    fingerprint = getattr(scorer, "fingerprint", None)
    if isinstance(fingerprint, str) and fingerprint:
        return fingerprint
    if isinstance(scorer, nn.Module):
        return neural_module_fingerprint(scorer)
    return f"type:{type(scorer).__module__}.{type(scorer).__qualname__}"


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
