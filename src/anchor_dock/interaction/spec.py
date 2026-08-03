"""Validated public specifications for interaction-guided docking."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real

MAX_INTERACTIONS = 8
_REQUIRED_FIELDS = (
    "receptor_residue",
    "receptor_atom",
    "ligand_smarts",
    "target_distance",
    "distance_tolerance",
)
_ALLOWED_FIELDS = frozenset((*_REQUIRED_FIELDS, "restraint_weight"))


def _nonempty_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_number(name: str, value: object, *, allow_zero: bool = False) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or (numeric == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} finite number")
    return numeric


@dataclass(frozen=True)
class InteractionConstraint:
    """One explicit receptor-to-ligand atom-pair distance hypothesis."""

    receptor_residue: str
    receptor_atom: str
    ligand_smarts: str
    target_distance: float
    distance_tolerance: float
    restraint_weight: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receptor_residue",
            _nonempty_text("receptor_residue", self.receptor_residue),
        )
        object.__setattr__(
            self,
            "receptor_atom",
            _nonempty_text("receptor_atom", self.receptor_atom).upper(),
        )
        object.__setattr__(
            self,
            "ligand_smarts",
            _nonempty_text("ligand_smarts", self.ligand_smarts),
        )
        target = _finite_number("target_distance", self.target_distance)
        tolerance = _finite_number("distance_tolerance", self.distance_tolerance)
        weight = _finite_number("restraint_weight", self.restraint_weight, allow_zero=True)
        if tolerance >= target:
            raise ValueError("distance_tolerance must satisfy 0 < tolerance < target_distance")
        object.__setattr__(self, "target_distance", target)
        object.__setattr__(self, "distance_tolerance", tolerance)
        object.__setattr__(self, "restraint_weight", weight)

    def as_dict(self) -> dict[str, str | float]:
        """Return a stable JSON-friendly representation."""
        return asdict(self)


type InteractionInput = InteractionConstraint | Mapping[str, object]


def normalize_interactions(
    *,
    interactions: Sequence[InteractionInput] | None,
    receptor_residue: object = None,
    receptor_atom: object = None,
    ligand_smarts: object = None,
    target_distance: object = None,
    distance_tolerance: object = None,
    default_restraint_weight: object = 10.0,
) -> tuple[InteractionConstraint, ...]:
    """Normalize canonical multi input or the legacy five-field single form."""
    default_weight = _finite_number(
        "restraint_weight", default_restraint_weight, allow_zero=True
    )
    legacy_values = {
        "receptor_residue": receptor_residue,
        "receptor_atom": receptor_atom,
        "ligand_smarts": ligand_smarts,
        "target_distance": target_distance,
        "distance_tolerance": distance_tolerance,
    }
    supplied_legacy = [name for name, value in legacy_values.items() if value is not None]

    if interactions is not None:
        if supplied_legacy:
            raise ValueError(
                "interactions cannot be combined with legacy single-interaction fields: "
                + ", ".join(supplied_legacy)
            )
        if isinstance(interactions, (str, bytes, bytearray, Mapping)) or not isinstance(
            interactions, Sequence
        ):
            raise TypeError("interactions must be a non-empty sequence")
        if not interactions:
            raise ValueError("interactions must be a non-empty sequence")
        if len(interactions) > MAX_INTERACTIONS:
            raise ValueError(f"interactions supports at most {MAX_INTERACTIONS} items")

        normalized: list[InteractionConstraint] = []
        for index, value in enumerate(interactions):
            if isinstance(value, InteractionConstraint):
                item = value
            elif isinstance(value, Mapping):
                keys = set(value)
                unknown = sorted(str(key) for key in keys - _ALLOWED_FIELDS)
                missing = [name for name in _REQUIRED_FIELDS if name not in value]
                if unknown:
                    raise ValueError(
                        f"interactions[{index}] contains unsupported fields: {', '.join(unknown)}"
                    )
                if missing:
                    raise ValueError(
                        f"interactions[{index}] requires fields: {', '.join(missing)}"
                    )
                item = InteractionConstraint(
                    receptor_residue=value["receptor_residue"],
                    receptor_atom=value["receptor_atom"],
                    ligand_smarts=value["ligand_smarts"],
                    target_distance=value["target_distance"],
                    distance_tolerance=value["distance_tolerance"],
                    restraint_weight=value.get("restraint_weight", default_weight),
                )
            else:
                raise TypeError(
                    f"interactions[{index}] must be an InteractionConstraint or mapping"
                )
            if item in normalized:
                raise ValueError(f"interactions[{index}] duplicates an earlier interaction")
            normalized.append(item)
        return tuple(normalized)

    missing_legacy = [name for name, value in legacy_values.items() if value is None]
    if missing_legacy:
        raise ValueError(
            "single interaction requires explicit fields: " + ", ".join(missing_legacy)
        )
    return (
        InteractionConstraint(
            receptor_residue=receptor_residue,
            receptor_atom=receptor_atom,
            ligand_smarts=ligand_smarts,
            target_distance=target_distance,
            distance_tolerance=distance_tolerance,
            restraint_weight=default_weight,
        ),
    )


def interaction_dicts(
    interactions: Sequence[InteractionConstraint],
) -> list[dict[str, str | float]]:
    """Return ordered canonical dictionaries for signatures and provenance."""
    return [item.as_dict() for item in interactions]
