"""Named scoring functions with explicit preparation and optimization routes.

An entry describes an implemented adapter, not a promise that an optional
executable, parameter set or trained model is installed on this machine.
"""
from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import entry_points


@dataclass(frozen=True)
class ScoringFunctionInfo:
    name: str
    family: str
    backend: str
    optimization: tuple[str, ...]
    preparation: str
    units: str
    direction: str = "lower"
    required_options: tuple[str, ...] = ()
    optional_modules: tuple[str, ...] = ()
    description: str = ""

    def as_dict(self, *, check_available: bool = True) -> dict:
        value = asdict(self)
        missing = []
        if check_available:
            for name in self.optional_modules:
                try:
                    found = importlib.util.find_spec(name) is not None
                except (ModuleNotFoundError, ImportError, ValueError):
                    found = False
                if not found:
                    missing.append(f"Python module {name}")
        value.update(dependencies_available=not missing, missing_dependencies=missing,
                     requires_configuration=bool(self.required_options))
        return value


@dataclass(frozen=True)
class ScoringFunctionRegistration:
    info: ScoringFunctionInfo
    factory: Callable[..., object]


_REGISTRY: dict[str, ScoringFunctionRegistration] = {}


def _removed_name(name: str) -> bool:
    return name in {"softdock", "dkoes_fast", "dkoes_scoring", "dkoes_scoring_old"} or name.startswith(
        ("smina:", "gnina:", "vina-native:"))


def register_scoring_function(info: ScoringFunctionInfo, factory: Callable[..., object], *, replace: bool = False) -> None:
    """Register a factory. Search energies must always be lower-is-better.

    ``direction`` describes reported scores. The factory must supply the
    corresponding search objective and a deterministic content fingerprint.
    Registration never silently substitutes another scoring function.
    """
    if (not isinstance(info, ScoringFunctionInfo) or not re.fullmatch(r"[a-z][a-z0-9_.:-]*", info.name)
            or _removed_name(info.name) or not callable(factory)):
        raise ValueError("a valid named scoring-function description and callable factory are required")
    if info.direction not in {"lower", "higher"} or not info.units:
        raise ValueError("score direction must be lower/higher and units must be explicit")
    if info.backend not in {"torch", "openmm"} or info.preparation not in {"inferred", "prepared-ad4", "amber-topology"}:
        raise ValueError("unsupported backend or preparation profile")
    if not info.optimization or set(info.optimization) - {"autograd", "pattern"}:
        raise ValueError("optimization routes must be autograd and/or pattern")
    if info.name in _REGISTRY and not replace:
        raise ValueError(f"scoring function {info.name!r} is already registered")
    _REGISTRY[info.name] = ScoringFunctionRegistration(info, factory)


def scoring_function_info(name: str) -> ScoringFunctionInfo:
    if not isinstance(name, str):
        raise TypeError("scoring-function name must be a string")
    if _removed_name(name.lower()):
        raise ValueError(f"{name.lower()} was removed; choose a documented Torch scoring function")
    try:
        return _REGISTRY[name.lower()].info
    except KeyError as exc:
        raise ValueError(f"unknown scoring function {name!r}; choose from {sorted(_REGISTRY)}") from exc


def list_scoring_functions(*, check_available: bool = True) -> list[dict]:
    """List adapters and dependency/parameter requirements without loading models."""
    return [value.info.as_dict(check_available=check_available) for _, value in sorted(_REGISTRY.items())]


def create_scoring_function(name: str, **options):
    info = scoring_function_info(name)
    missing = [key for key in info.required_options if key not in options]
    if missing:
        raise ValueError(f"{name} requires explicit scorer options: {', '.join(missing)}")
    result = _REGISTRY[info.name].factory(**options)
    method = "prepare"
    if not callable(getattr(result, method, None)):
        raise TypeError(f"{name} factory must return a scorer with {method}(); wrap neural modules with NeuralScorerAdapter")
    if getattr(result, "units", None) != info.units:
        raise ValueError(f"{name} factory units disagree with its registered units")
    return result


def load_scoring_plugins() -> list[str]:
    """Explicitly load installed ``anchor_dock.scorers`` entry points.

    Each entry point resolves to a ScoringFunctionRegistration or a no-argument
    provider returning one. Loading is explicit since plugin code may load
    models or have optional dependencies.
    """
    names = []
    for entry in sorted(entry_points(group="anchor_dock.scorers"), key=lambda e: e.name):
        value = entry.load()
        value = value() if callable(value) and not isinstance(value, ScoringFunctionRegistration) else value
        if not isinstance(value, ScoringFunctionRegistration):
            raise TypeError(f"plugin {entry.name!r} must provide ScoringFunctionRegistration")
        register_scoring_function(value.info, value.factory)
        names.append(value.info.name)
    return names


def _pairwise(name, **options):
    from .scoring import PairwiseScorer
    return PairwiseScorer(name, **options)


def _ad4(*, maps_prefix, atom_types, **options):
    from .ad4 import AD4Scorer
    return AD4Scorer.from_autogrid(maps_prefix, atom_types, **options)


def _adgpu(*, maps_prefix, atom_types, **options):
    if options.get("map_mode", "raw") != "raw" or options.get("formulation", "adgpu") != "adgpu":
        raise ValueError("adgpu uses raw maps and the explicit adgpu formulation")
    options.update(map_mode="raw", formulation="adgpu")
    return _ad4(maps_prefix=maps_prefix, atom_types=atom_types, **options)


def _amber(*, prmtop, inpcrd, ligand_atom_indices, **options):
    from .torch_amber import TorchAmberScorer
    return TorchAmberScorer.from_amber(prmtop, inpcrd, ligand_atom_indices, **options)


class NativeFinalScorer:
    """Native-table final energies as an explicit nondifferentiable objective."""
    supports_autograd = False
    preparation_requirement = "prepared-ad4"

    def __init__(self, name: str):
        from .scoring import PairwiseScorer
        self.base = PairwiseScorer(name)
        self.name, self.units = name + "-final", self.base.units

    @property
    def fingerprint(self):
        import hashlib

        from .native_vina import NATIVE_VINA_SCORE_CONTRACT
        return "sha256:" + hashlib.sha256(
            (NATIVE_VINA_SCORE_CONTRACT + self.base.fingerprint).encode()).hexdigest()

    def prepare(self, *args, **kwargs):
        return _PreparedNativeFinal(self.base.prepare(*args, **kwargs), self.name, self.fingerprint)


@dataclass
class _PreparedNativeFinal:
    base: object
    name: str
    fingerprint: str
    units: str = "kcal/mol-like"
    supports_autograd: bool = False

    @property
    def effective_rotatable_bonds(self):
        return self.base.effective_rotatable_bonds

    @property
    def torsion_penalty_applied(self):
        return self.base.torsion_penalty_applied

    def raw_components(self, coordinates):
        from .native_vina import native_vina_components
        return native_vina_components(self.base, coordinates.detach())

    def search_energy(self, coordinates):
        return self.raw_components(coordinates).search_energy

    def report(self, raw, intramolecular_reference=None):
        return self.base.report(raw, intramolecular_reference)


def _install_builtins():
    from functools import partial
    for name in ("vina", "vinardo"):
        register_scoring_function(ScoringFunctionInfo(name, name, "torch", ("autograd", "pattern"),
                                  "inferred", "kcal/mol-like"), partial(_pairwise, name))
        register_scoring_function(ScoringFunctionInfo(name + "-final", name, "torch", ("pattern",),
                                  "prepared-ad4", "kcal/mol-like", optional_modules=("meeko",),
                                  description="Nondifferentiable native FAST-table energy; pattern optimization only"),
                                  partial(NativeFinalScorer, name))
    for name, factory in (("ad4", _ad4), ("adgpu", _adgpu)):
        register_scoring_function(ScoringFunctionInfo(name, "ad4", "torch", ("autograd", "pattern"),
                                  "prepared-ad4", "kcal/mol-like", required_options=("maps_prefix", "atom_types"),
                                  optional_modules=("meeko",),
                                  description="Preweighted AutoGrid maps; Torch refinement, not a native engine"), factory)
    register_scoring_function(ScoringFunctionInfo("amber", "amber", "torch", ("autograd", "pattern"),
                              "amber-topology", "kcal/mol", required_options=("prmtop", "inpcrd", "ligand_atom_indices"),
                              optional_modules=("openmm",),
                              description="Torch fixed-charge vacuum potential; OpenMM only imports parameters"), _amber)


_install_builtins()
