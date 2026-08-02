"""Unified heterogeneous batch input and execution."""

from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import io
import json
import lzma
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from rdkit import Chem, rdBase

from ._version import __version__
from .core.io import file_content_fingerprint


@dataclass
class LigandRecord:
    """One ligand plus optional stable identity and metadata."""

    ligand: str | os.PathLike[str] | Chem.Mol
    id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    options: dict[str, object] = field(default_factory=dict)


@dataclass
class DockingJob:
    """One fully or partially specified docking job."""

    mode: Literal["reference", "covalent", "free"]
    ligand: str | os.PathLike[str] | Chem.Mol
    protein_pdb: str | os.PathLike[str] | None = None
    id: str | None = None
    reference_ligand: str | os.PathLike[str] | None = None
    reactive_residue: str | None = None
    options: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def reference(
        cls,
        ligand: str | os.PathLike[str] | Chem.Mol,
        *,
        protein_pdb: str | os.PathLike[str],
        reference_ligand: str | os.PathLike[str],
        id: str | None = None,
        metadata: Mapping[str, object] | None = None,
        **options: object,
    ) -> DockingJob:
        return cls(
            "reference",
            ligand,
            protein_pdb,
            id,
            reference_ligand,
            None,
            dict(options),
            dict(metadata or {}),
        )

    @classmethod
    def covalent(
        cls,
        ligand: str | os.PathLike[str] | Chem.Mol,
        *,
        protein_pdb: str | os.PathLike[str],
        reactive_residue: str | None = None,
        id: str | None = None,
        metadata: Mapping[str, object] | None = None,
        **options: object,
    ) -> DockingJob:
        return cls(
            "covalent",
            ligand,
            protein_pdb,
            id,
            None,
            reactive_residue,
            dict(options),
            dict(metadata or {}),
        )

    @classmethod
    def free(
        cls,
        ligand: str | os.PathLike[str] | Chem.Mol,
        *,
        protein_pdb: str | os.PathLike[str],
        id: str | None = None,
        metadata: Mapping[str, object] | None = None,
        **options: object,
    ) -> DockingJob:
        return cls(
            "free",
            ligand,
            protein_pdb,
            id,
            None,
            None,
            dict(options),
            dict(metadata or {}),
        )


@dataclass(frozen=True)
class _InvalidBatchItem:
    source: str
    index: int
    error: str


@dataclass
class _PreparedBatchEntry:
    index: int
    item: DockingJob | LigandRecord | _InvalidBatchItem
    job: DockingJob | None
    job_id: str | None
    job_dir: Path | None
    result_path: Path | None
    options: dict[str, object] | None
    runtime_identity: dict[str, str] | None
    signature: str | None
    error: Exception | None


_LIGAND_FIELDS = ("ligand", "query_ligand", "smiles", "inchi", "structure")
_SUPPORTED_SUFFIXES = {
    ".sdf",
    ".mol",
    ".mol2",
    ".pdb",
    ".ent",
    ".smi",
    ".smiles",
    ".inchi",
    ".txt",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ndjson",
}
_COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".lzma"}
_BATCH_SCHEMA_VERSION = "2"
_BATCH_MANIFEST_NAMES = ("results.jsonl", "summary.json")


def _coerce_value(value: object) -> object:
    """Coerce scalar values read from text tables into stable Python types."""
    if not isinstance(value, str):
        if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                return value
        return value
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    if text[:1] in {"[", "{"} or (text[:1] == text[-1:] and text[:1] in {'"', "'"}):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    if re.fullmatch(r"[-+]?(?:0|[1-9]\d*)", text):
        try:
            return int(text)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            pass
    return text


def _coerce_options(values: Mapping[str, object] | None) -> dict[str, object]:
    result: dict[str, object] = {}
    if values is None:
        return result
    for key, raw in values.items():
        if _is_missing(raw):
            continue
        value = _coerce_value(raw)
        if key in {"center", "box_size"} and isinstance(value, str) and "," in value:
            try:
                value = tuple(float(part.strip()) for part in value.split(","))
            except ValueError:
                pass
        result[str(key)] = value
    return result


def _mapping_object(value: object) -> Mapping[str, object] | None:
    coerced = _coerce_value(value)
    return coerced if isinstance(coerced, Mapping) else None


def _file_identity(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Chem.Mol):
        return {"mol": Chem.MolToSmiles(Chem.RemoveHs(value), canonical=True, isomericSmiles=True)}
    text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    path = Path(text)
    if path.is_file():
        return {
            "content_fingerprint": file_content_fingerprint(path),
            "size": path.stat().st_size,
        }
    return text


def _stable_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _stable_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_stable_value(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Chem.Mol):
        return {"mol": Chem.MolToSmiles(Chem.RemoveHs(value), canonical=True, isomericSmiles=True)}
    if hasattr(value, "state_dict") or hasattr(value, "fingerprint"):
        from .core.scoring import scorer_fingerprint

        return {
            "scorer_fingerprint": scorer_fingerprint(value),
            "scorer_name": getattr(value, "name", None),
            "score_units": getattr(value, "units", None),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def _runtime_identity(options: Mapping[str, object]) -> dict[str, str]:
    requested_device = options.get("device")
    resolved_device = torch.device(
        requested_device if requested_device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return {
        "device": str(resolved_device),
        "numpy": str(np.__version__),
        "rdkit": str(rdBase.rdkitVersion),
        "torch": str(torch.__version__),
    }


def _job_signature(
    job: DockingJob,
    options: Mapping[str, object],
    runtime_identity: Mapping[str, str],
) -> str:
    payload = {
        "schema": _BATCH_SCHEMA_VERSION,
        "anchor_dock_version": __version__,
        "runtime": dict(runtime_identity),
        "mode": job.mode,
        "ligand": _file_identity(job.ligand),
        "protein": _file_identity(job.protein_pdb),
        "reference": _file_identity(job.reference_ligand),
        "reactive_residue": job.reactive_residue,
        "metadata": _stable_value(job.metadata),
        "options": _stable_value(options),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        comparison = value != value
        return bool(comparison) if isinstance(comparison, bool) else False
    except (TypeError, ValueError):
        return False


def _canonical_ligand_text(ligand: object) -> str:
    if isinstance(ligand, Chem.Mol):
        return Chem.MolToSmiles(Chem.RemoveHs(ligand), canonical=True, isomericSmiles=True)
    return os.fspath(ligand) if isinstance(ligand, os.PathLike) else str(ligand)


def _safe_id(value: str, fallback: str, salt: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._") or fallback
    slug = slug[:64]
    digest = hashlib.sha1(salt.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def _mapping_to_item(values: Mapping[str, object]) -> DockingJob | LigandRecord:
    ligand_field = next(
        (name for name in _LIGAND_FIELDS if name in values and not _is_missing(values[name])),
        None,
    )
    if ligand_field is None:
        raise ValueError(f"mapping does not contain a ligand field ({', '.join(_LIGAND_FIELDS)})")
    ligand = values[ligand_field]
    assert ligand is not None
    item_id = next(
        (values.get(name) for name in ("id", "name", "ligand_id") if not _is_missing(values.get(name))),
        None,
    )
    metadata_mapping = _mapping_object(values.get("metadata"))
    metadata = dict(metadata_mapping) if metadata_mapping is not None else {}
    known = {
        *_LIGAND_FIELDS,
        "id",
        "name",
        "ligand_id",
        "mode",
        "protein_pdb",
        "protein",
        "receptor",
        "reference_ligand",
        "reference",
        "ref_ligand",
        "reactive_residue",
        "residue",
        "options",
        "metadata",
    }
    options_mapping = _mapping_object(values.get("options"))
    options = _coerce_options(options_mapping)
    for key, value in values.items():
        if key not in known and not _is_missing(value):
            options[key] = _coerce_value(value)
    mode = values.get("mode")
    if _is_missing(mode):
        return LigandRecord(
            ligand,
            str(item_id) if item_id is not None else None,
            metadata,
            options,
        )
    mode_text = str(mode).strip().lower()
    if mode_text not in {"reference", "covalent", "free"}:
        raise ValueError(f"unknown docking mode {mode!r}")
    protein = next(
        (values.get(name) for name in ("protein_pdb", "protein", "receptor") if not _is_missing(values.get(name))),
        None,
    )
    reference = next(
        (
            values.get(name)
            for name in ("reference_ligand", "reference", "ref_ligand")
            if not _is_missing(values.get(name))
        ),
        None,
    )
    residue = next(
        (values.get(name) for name in ("reactive_residue", "residue") if not _is_missing(values.get(name))),
        None,
    )
    return DockingJob(
        mode_text,  # type: ignore[arg-type]
        ligand,  # type: ignore[arg-type]
        protein,  # type: ignore[arg-type]
        str(item_id) if item_id is not None else None,
        reference,  # type: ignore[arg-type]
        str(residue) if residue is not None else None,
        options,
        metadata,
    )


def _decompress(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] in _COMPRESSED_SUFFIXES:
        compression = suffixes.pop()
        if compression == ".gz":
            data = gzip.decompress(data)
        elif compression == ".bz2":
            data = bz2.decompress(data)
        else:
            data = lzma.decompress(data)
    logical_suffix = suffixes[-1] if suffixes else path.suffix.lower()
    return data, logical_suffix


def _decompress_stream(data: bytes, source_name: str) -> tuple[bytes, str]:
    suffixes = [suffix.lower() for suffix in Path(source_name).suffixes]
    if suffixes and suffixes[-1] in _COMPRESSED_SUFFIXES:
        compression = suffixes.pop()
        if compression == ".gz":
            data = gzip.decompress(data)
        elif compression == ".bz2":
            data = bz2.decompress(data)
        else:
            data = lzma.decompress(data)
    logical_suffix = suffixes[-1] if suffixes else Path(source_name).suffix.lower() or ".smi"
    return data, logical_suffix


def _records_from_sdf(data: bytes, source_name: str) -> Iterator[LigandRecord | _InvalidBatchItem]:
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(data), removeHs=True)
    for index, mol in enumerate(supplier, start=1):
        if mol is None:
            yield _InvalidBatchItem(source_name, index, "failed to parse SDF record")
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        yield LigandRecord(mol, name or f"{Path(source_name).stem}_{index:05d}")


def _resolve_relative_path(value: object, base_dir: Path | None) -> object:
    if base_dir is None or not isinstance(value, (str, os.PathLike)):
        return value
    path = Path(os.fspath(value))
    if path.is_absolute():
        return path
    candidate = base_dir / path
    return candidate if candidate.exists() else value


def _resolve_item_paths(
    item: DockingJob | LigandRecord,
    base_dir: Path | None,
) -> DockingJob | LigandRecord:
    if base_dir is None:
        return item
    if isinstance(item, LigandRecord):
        return replace(item, ligand=_resolve_relative_path(item.ligand, base_dir))
    return replace(
        item,
        ligand=_resolve_relative_path(item.ligand, base_dir),
        protein_pdb=_resolve_relative_path(item.protein_pdb, base_dir),
        reference_ligand=_resolve_relative_path(item.reference_ligand, base_dir),
    )


def _items_from_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    base_dir: Path | None = None,
) -> Iterator[DockingJob | LigandRecord | _InvalidBatchItem]:
    for index, row in enumerate(rows, start=1):
        try:
            yield _resolve_item_paths(_mapping_to_item(row), base_dir)
        except (TypeError, ValueError) as exc:
            yield _InvalidBatchItem(str(base_dir or "table"), index, str(exc))


def _items_from_text(
    data: bytes,
    suffix: str,
    source_name: str,
    *,
    base_dir: Path | None = None,
) -> Iterator[DockingJob | LigandRecord | _InvalidBatchItem]:
    text = data.decode("utf-8-sig")
    if suffix in {".jsonl", ".ndjson"}:
        for index, line in enumerate(text.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield _InvalidBatchItem(source_name, index, f"invalid JSON: {exc.msg}")
                continue
            if not isinstance(row, Mapping):
                yield _InvalidBatchItem(source_name, index, "JSONL record must be an object")
                continue
            try:
                yield _resolve_item_paths(_mapping_to_item(row), base_dir)
            except (TypeError, ValueError) as exc:
                yield _InvalidBatchItem(source_name, index, str(exc))
        return
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            container_keys = [key for key in ("jobs", "ligands") if key in payload]
            if len(container_keys) > 1:
                raise ValueError("JSON batch input cannot contain both 'jobs' and 'ligands'")
            payload = payload[container_keys[0]] if container_keys else [payload]
        if not isinstance(payload, list):
            raise ValueError("JSON batch input must be an object, list, or contain jobs/ligands")
        for index, item in enumerate(payload, start=1):
            if isinstance(item, Mapping):
                try:
                    yield _resolve_item_paths(_mapping_to_item(item), base_dir)
                except (TypeError, ValueError) as exc:
                    yield _InvalidBatchItem(source_name, index, str(exc))
            else:
                yield _InvalidBatchItem(source_name, index, "JSON batch records must be objects")
        return
    if suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        yield from _items_from_rows(csv.DictReader(io.StringIO(text), dialect=dialect), base_dir=base_dir)
        return

    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        ligand = fields[0]
        name = fields[1] if len(fields) > 1 else f"{Path(source_name).stem}_{index:05d}"
        yield _resolve_item_paths(LigandRecord(ligand, name), base_dir)


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory)
    except (OSError, ValueError):
        return False
    return True


def _items_from_path(
    path: Path,
    *,
    excluded_directory: Path | None = None,
) -> Iterator[DockingJob | LigandRecord | _InvalidBatchItem]:
    if path.is_dir():
        if excluded_directory is not None and path.resolve() == excluded_directory:
            raise ValueError("output_dir must not be the same directory as a batch directory source")
        if excluded_directory is not None and _is_within_directory(path, excluded_directory):
            return
        candidates: list[Path] = []
        for current, directory_names, file_names in os.walk(path, topdown=True, followlinks=False):
            current_path = Path(current)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if excluded_directory is None or not _is_within_directory(current_path / name, excluded_directory)
            )
            candidates.extend(current_path / name for name in sorted(file_names))
        for child in sorted(candidate for candidate in candidates if candidate.is_file()):
            suffixes = [suffix.lower() for suffix in child.suffixes]
            logical_suffix = (
                suffixes[-2]
                if suffixes and suffixes[-1] in _COMPRESSED_SUFFIXES and len(suffixes) > 1
                else child.suffix.lower()
            )
            if logical_suffix in _SUPPORTED_SUFFIXES:
                yield from _items_from_path(child, excluded_directory=excluded_directory)
        return
    data, suffix = _decompress(path)
    if suffix == ".sdf":
        yield from _records_from_sdf(data, path.name)
    elif suffix in {".mol", ".mol2", ".pdb", ".ent"}:
        yield LigandRecord(path, path.stem)
    elif suffix in _SUPPORTED_SUFFIXES:
        yield from _items_from_text(data, suffix, path.name, base_dir=path.parent)
    else:
        raise ValueError(f"unsupported batch input file: {path}")


def _iter_batch_items(
    source: object,
    *,
    excluded_directory: Path | None,
) -> Iterator[DockingJob | LigandRecord | _InvalidBatchItem]:
    """Normalize Python collections, tables, files, directories, and streams."""
    if isinstance(source, (DockingJob, LigandRecord, _InvalidBatchItem)):
        yield source
        return
    if isinstance(source, Chem.Mol):
        yield LigandRecord(source)
        return
    if (
        isinstance(source, tuple)
        and len(source) == 2
        and isinstance(source[1], str)
        and isinstance(source[0], (str, os.PathLike, Chem.Mol))
    ):
        yield LigandRecord(source[0], source[1])
        return
    if isinstance(source, Mapping):
        if any(field in source for field in _LIGAND_FIELDS):
            try:
                yield _mapping_to_item(source)
            except (TypeError, ValueError) as exc:
                yield _InvalidBatchItem("mapping", 1, str(exc))
        else:
            for key, value in source.items():
                yield LigandRecord(value, str(key))  # type: ignore[arg-type]
        return
    if hasattr(source, "to_dict") and not isinstance(source, (str, bytes, os.PathLike)):
        try:
            records = source.to_dict(orient="records")  # type: ignore[attr-defined]
        except TypeError:
            records = source.to_dict("records")  # type: ignore[attr-defined]
        yield from _items_from_rows(records)
        return
    if hasattr(source, "read") and not isinstance(source, (str, bytes, os.PathLike)):
        content = source.read()  # type: ignore[attr-defined]
        data = content.encode() if isinstance(content, str) else bytes(content)
        name = getattr(source, "name", "batch.smi")
        data, suffix = _decompress_stream(data, str(name))
        if suffix == ".sdf":
            yield from _records_from_sdf(data, str(name))
        else:
            yield from _items_from_text(data, suffix, str(name))
        return
    if isinstance(source, (str, os.PathLike)):
        path = Path(os.fspath(source))
        if path.exists():
            # Explicit files remain valid inputs even when they are stored
            # below the output root. Only recursive directory discovery omits
            # generated output trees.
            yield from _items_from_path(
                path,
                excluded_directory=excluded_directory if path.is_dir() else None,
            )
        else:
            yield LigandRecord(source)
        return
    if isinstance(source, bytes):
        yield from _items_from_text(source, ".smi", "memory.smi")
        return
    if isinstance(source, Iterable):
        for item in source:
            yield from _iter_batch_items(item, excluded_directory=excluded_directory)
        return
    raise TypeError(f"unsupported batch input type: {type(source).__name__}")


def iter_batch_items(source: object) -> Iterator[DockingJob | LigandRecord | _InvalidBatchItem]:
    """Normalize Python collections, tables, files, directories, and streams."""
    yield from _iter_batch_items(source, excluded_directory=None)


def _buffer_batch_stream(source: object) -> io.BytesIO:
    content = source.read()  # type: ignore[attr-defined]
    data = content.encode() if isinstance(content, str) else bytes(content)
    buffered = io.BytesIO(data)
    buffered.name = str(getattr(source, "name", "batch.smi"))
    return buffered


def _materialize_preflight_source(
    source: object,
    active: set[int] | None = None,
) -> object:
    """Freeze finite iterable inputs before output publication."""
    if isinstance(source, (DockingJob, LigandRecord, _InvalidBatchItem, Chem.Mol, str, bytes, os.PathLike, Mapping)):
        return source
    if hasattr(source, "to_dict"):
        try:
            records = source.to_dict(orient="records")  # type: ignore[attr-defined]
        except TypeError:
            records = source.to_dict("records")  # type: ignore[attr-defined]
        return _materialize_preflight_source(records, active)
    if hasattr(source, "read"):
        return _buffer_batch_stream(source)
    if (
        isinstance(source, tuple)
        and len(source) == 2
        and isinstance(source[1], str)
        and isinstance(source[0], (str, os.PathLike, Chem.Mol))
    ):
        return source

    active = set() if active is None else active
    source_id = id(source)
    if source_id in active:
        raise ValueError("batch input contains a circular iterable")

    if isinstance(source, (list, tuple, set, frozenset)):
        active.add(source_id)
        try:
            values = [_materialize_preflight_source(item, active) for item in source]
        finally:
            active.remove(source_id)
        if isinstance(source, tuple):
            return tuple(values)
        return values

    if isinstance(source, Iterable):
        active.add(source_id)
        try:
            return [_materialize_preflight_source(item, active) for item in source]
        finally:
            active.remove(source_id)
    return source


def _validate_reserved_input_path(value: object, output_directory: Path) -> None:
    if not isinstance(value, (str, os.PathLike)):
        return
    path = Path(os.fspath(value))
    manifests = tuple(output_directory / name for name in _BATCH_MANIFEST_NAMES)
    lexical = Path(os.path.abspath(os.fspath(path)))
    if any(lexical == Path(os.path.abspath(os.fspath(manifest))) for manifest in manifests):
        raise ValueError("batch input file conflicts with an output manifest path")
    try:
        resolved = path.resolve()
    except OSError:
        return
    if any(resolved == manifest.resolve() for manifest in manifests):
        raise ValueError("batch input file conflicts with an output manifest path")


def _validate_source_output_separation(
    source: object,
    output_directory: Path,
    seen: set[int] | None = None,
) -> None:
    """Reject reusable source/output collisions before publishing manifests."""
    if isinstance(source, DockingJob):
        for value in (source.ligand, source.protein_pdb, source.reference_ligand):
            _validate_reserved_input_path(value, output_directory)
        return
    if isinstance(source, LigandRecord):
        _validate_reserved_input_path(source.ligand, output_directory)
        return
    if isinstance(source, (str, os.PathLike)):
        source_path = Path(os.fspath(source))
        try:
            if source_path.is_dir() and source_path.resolve() == output_directory:
                raise ValueError("output_dir must not be the same directory as a batch directory source")
        except OSError:
            pass
        _validate_reserved_input_path(source, output_directory)
        return
    if hasattr(source, "read"):
        _validate_reserved_input_path(getattr(source, "name", None), output_directory)
        return
    if isinstance(source, Mapping):
        path_fields = {
            *_LIGAND_FIELDS,
            "protein_pdb",
            "protein",
            "receptor",
            "reference_ligand",
            "reference",
            "ref_ligand",
        }
        if any(field in source for field in _LIGAND_FIELDS):
            values = (source[field] for field in path_fields if field in source)
        else:
            values = source.values()
        for value in values:
            _validate_reserved_input_path(value, output_directory)
        return
    if (
        isinstance(source, tuple)
        and len(source) == 2
        and isinstance(source[1], str)
        and isinstance(source[0], (str, os.PathLike, Chem.Mol))
    ):
        _validate_source_output_separation(source[0], output_directory, seen)
        return
    if not isinstance(source, (list, tuple, set, frozenset)):
        return
    seen = set() if seen is None else seen
    source_id = id(source)
    if source_id in seen:
        return
    seen.add(source_id)
    try:
        for item in source:
            _validate_source_output_separation(item, output_directory, seen)
    finally:
        seen.remove(source_id)


def _add_existing_input_path(value: object, paths: set[Path]) -> None:
    if not isinstance(value, (str, os.PathLike)):
        return
    try:
        path = Path(os.fspath(value))
        if path.exists() or path.is_symlink():
            paths.add(path)
    except (OSError, TypeError, ValueError):
        return


def _collect_existing_input_paths(
    source: object,
    paths: set[Path] | None = None,
    seen: set[int] | None = None,
) -> set[Path]:
    paths = set() if paths is None else paths
    if isinstance(source, DockingJob):
        for value in (source.ligand, source.protein_pdb, source.reference_ligand):
            _add_existing_input_path(value, paths)
        return paths
    if isinstance(source, LigandRecord):
        _add_existing_input_path(source.ligand, paths)
        return paths
    if isinstance(source, (str, os.PathLike)):
        _add_existing_input_path(source, paths)
        return paths
    if hasattr(source, "read"):
        _add_existing_input_path(getattr(source, "name", None), paths)
        return paths
    if isinstance(source, Mapping):
        path_fields = {
            *_LIGAND_FIELDS,
            "protein_pdb",
            "protein",
            "receptor",
            "reference_ligand",
            "reference",
            "ref_ligand",
        }
        if any(field in source for field in _LIGAND_FIELDS):
            values = (source[field] for field in path_fields if field in source)
        else:
            values = source.values()
        for value in values:
            _add_existing_input_path(value, paths)
        return paths
    if (
        isinstance(source, tuple)
        and len(source) == 2
        and isinstance(source[1], str)
        and isinstance(source[0], (str, os.PathLike, Chem.Mol))
    ):
        _add_existing_input_path(source[0], paths)
        return paths
    if not isinstance(source, (list, tuple, set, frozenset)):
        return paths
    seen = set() if seen is None else seen
    source_id = id(source)
    if source_id in seen:
        return paths
    seen.add(source_id)
    try:
        for item in source:
            _collect_existing_input_paths(item, paths, seen)
    finally:
        seen.remove(source_id)
    return paths


def _path_is_within_output_subtree(path: Path, directory: Path) -> bool:
    lexical_path = Path(os.path.abspath(os.fspath(path)))
    lexical_directory = Path(os.path.abspath(os.fspath(directory)))
    for candidate, root in (
        (lexical_path, lexical_directory),
        (path.resolve(), directory.resolve()),
    ):
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _validate_input_output_ownership(
    input_paths: set[Path],
    output_directory: Path,
    owned_directories: list[Path],
) -> None:
    for path in input_paths:
        _validate_reserved_input_path(path, output_directory)
    for owned_directory in owned_directories:
        if owned_directory.is_symlink() or (owned_directory.exists() and not owned_directory.is_dir()):
            raise ValueError("batch job output path must be a real directory, not a file or symlink")
        if not _is_within_directory(owned_directory, output_directory):
            raise ValueError("batch job output directory resolves outside output_dir")
        for path in input_paths:
            if _path_is_within_output_subtree(path, owned_directory):
                raise ValueError("batch input path conflicts with a batch job output directory")


def _materialize_job(
    item: DockingJob | LigandRecord,
    *,
    mode: str | None,
    protein_pdb: str | os.PathLike[str] | None,
    reference_ligand: str | os.PathLike[str] | None,
    reactive_residue: str | None,
) -> DockingJob:
    if isinstance(item, DockingJob):
        if item.mode not in {"reference", "covalent", "free"}:
            raise ValueError("mode must be reference, covalent or free")
        job = item
    else:
        if mode is None:
            raise ValueError("mode is required for ligand-only batch inputs")
        if mode not in {"reference", "covalent", "free"}:
            raise ValueError("mode must be reference, covalent or free")
        job = DockingJob(
            mode,  # type: ignore[arg-type]
            item.ligand,
            protein_pdb,
            item.id,
            reference_ligand,
            reactive_residue,
            dict(item.options),
            dict(item.metadata),
        )
    job = replace(
        job,
        protein_pdb=job.protein_pdb if job.protein_pdb is not None else protein_pdb,
        reference_ligand=(job.reference_ligand if job.reference_ligand is not None else reference_ligand),
        reactive_residue=(job.reactive_residue if job.reactive_residue is not None else reactive_residue),
    )
    if job.protein_pdb is None:
        raise ValueError("protein_pdb is required for every docking job")
    if job.mode == "reference" and job.reference_ligand is None:
        raise ValueError("reference_ligand is required for reference jobs")
    return job


def _prepare_batch_entries(
    items: list[DockingJob | LigandRecord | _InvalidBatchItem],
    *,
    root: Path,
    mode: Literal["reference", "covalent", "free"] | None,
    protein_pdb: str | os.PathLike[str] | None,
    reference_ligand: str | os.PathLike[str] | None,
    reactive_residue: str | None,
    defaults: Mapping[str, object],
) -> list[_PreparedBatchEntry]:
    entries: list[_PreparedBatchEntry] = []
    used_job_ids: dict[str, int] = {}
    for index, item in enumerate(items, start=1):
        try:
            if isinstance(item, _InvalidBatchItem):
                raise ValueError(f"{item.source} record {item.index}: {item.error}")
            job = _materialize_job(
                item,
                mode=mode,
                protein_pdb=protein_pdb,
                reference_ligand=reference_ligand,
                reactive_residue=reactive_residue,
            )
            ligand_text = _canonical_ligand_text(job.ligand)
            raw_id = job.id or "ligand"
            identity = (
                f"{job.mode}|{ligand_text}|{job.protein_pdb}|{job.reference_ligand}|{job.reactive_residue}|{raw_id}"
            )
            base_job_id = _safe_id(raw_id, "ligand", identity)
            occurrence = used_job_ids.get(base_job_id, 0) + 1
            used_job_ids[base_job_id] = occurrence
            job_id = base_job_id if occurrence == 1 else f"{base_job_id}-{occurrence}"
            job_dir = root / job_id
            options = _coerce_options(defaults)
            options.update(_coerce_options(job.options))
            options.pop("output_dir", None)
            options.setdefault("verbose", False)
            runtime_identity = _runtime_identity(options)
            signature = _job_signature(job, options, runtime_identity)
            entries.append(
                _PreparedBatchEntry(
                    index,
                    item,
                    job,
                    job_id,
                    job_dir,
                    job_dir / "result.json",
                    options,
                    runtime_identity,
                    signature,
                    None,
                )
            )
        except Exception as exc:
            entries.append(
                _PreparedBatchEntry(
                    index,
                    item,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    exc,
                )
            )
    return entries


def _dispatch(job: DockingJob, output_dir: Path, options: dict[str, object]) -> dict[str, object]:
    if job.mode == "reference":
        from .reference.pipeline import dock_reference

        assert job.reference_ligand is not None
        return dock_reference(
            job.protein_pdb,  # type: ignore[arg-type]
            job.reference_ligand,
            job.ligand,
            output_dir,
            **options,
        )
    if job.mode == "covalent":
        from .covalent.pipeline import dock_covalent

        return dock_covalent(
            job.protein_pdb,  # type: ignore[arg-type]
            job.ligand,
            job.reactive_residue,
            output_dir,
            **options,
        )
    if job.mode == "free":
        from .free import dock_free

        return dock_free(job.protein_pdb, job.ligand, output_dir, **options)  # type: ignore[arg-type]
    raise ValueError(f"unsupported docking mode: {job.mode!r}")


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str),
    )


def _validated_output_artifact(value: object, job_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw]
    if not raw.is_absolute() and raw.parent == Path("."):
        candidates.append(job_dir / raw)
    try:
        job_root = job_dir.resolve(strict=True)
    except OSError:
        return None
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(job_root)
            if resolved.is_file() and resolved.stat().st_size > 0:
                return resolved
        except (OSError, ValueError):
            continue
    return None


def _load_resume_result(result_path: Path, job_dir: Path, signature: str) -> dict[str, object] | None:
    try:
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    if loaded.get("job_signature") != signature or loaded.get("success") is not True:
        return None
    artifact = _validated_output_artifact(loaded.get("output_file"), job_dir)
    if artifact is None:
        return None
    expected_fingerprint = loaded.get("output_artifact_fingerprint")
    expected_size = loaded.get("output_artifact_size")
    if (
        not isinstance(expected_fingerprint, str)
        or not isinstance(expected_size, int)
        or artifact.stat().st_size != expected_size
        or file_content_fingerprint(artifact) != expected_fingerprint
    ):
        return None
    loaded["output_file"] = str(artifact)
    loaded["resumed"] = True
    return loaded


def _preflight_failure(
    root: Path,
    index: int,
    item: DockingJob | LigandRecord | _InvalidBatchItem,
    error: Exception,
    mode: str | None,
) -> dict[str, object]:
    job_id = f"invalid-{index:05d}"
    item_mode = item.mode if isinstance(item, DockingJob) else mode
    metadata = (
        item.metadata
        if isinstance(item, (DockingJob, LigandRecord))
        else {
            "source": item.source,
            "source_index": item.index,
        }
    )
    result: dict[str, object] = {
        "job_id": job_id,
        "mode": item_mode,
        "success": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "metadata": metadata,
    }
    _atomic_write_json(root / job_id / "result.json", result)
    return result


def dock_batch(
    source: object,
    *,
    mode: Literal["reference", "covalent", "free"] | None = None,
    protein_pdb: str | os.PathLike[str] | None = None,
    reference_ligand: str | os.PathLike[str] | None = None,
    reactive_residue: str | None = None,
    output_dir: str | os.PathLike[str] = "anchor_dock_batch",
    on_error: Literal["record", "raise", "skip"] = "record",
    resume: bool = False,
    **defaults: object,
) -> list[dict[str, object]]:
    """Execute homogeneous or mixed-mode jobs through one API.

    Inputs may be individual ligands, RDKit molecules, iterables, generators,
    mappings, DataFrame-like objects, file-like objects, directories, SDF/SMI,
    CSV/TSV, JSON/JSONL, or gzip/bzip2/xz-compressed text/SDF files.
    Finite iterator and generic iterable inputs are fully materialized for
    collision preflight before output publication. Non-terminating iterables
    cannot form a completable batch input.
    Execution is sequential at the molecule level; each job still uses batched
    pose scoring and optimization and receptor contexts are cached globally.
    """
    if on_error not in {"record", "raise", "skip"}:
        raise ValueError("on_error must be record, raise or skip")
    root = Path(output_dir)
    resolved_root = root.resolve()
    _validate_source_output_separation(source, resolved_root)
    _validate_reserved_input_path(protein_pdb, resolved_root)
    _validate_reserved_input_path(reference_ligand, resolved_root)
    input_paths = _collect_existing_input_paths(source)
    _add_existing_input_path(protein_pdb, input_paths)
    _add_existing_input_path(reference_ligand, input_paths)
    source = _materialize_preflight_source(source)
    _validate_source_output_separation(source, resolved_root)
    _collect_existing_input_paths(source, input_paths)
    items = list(_iter_batch_items(source, excluded_directory=resolved_root))
    _validate_source_output_separation(items, resolved_root)
    _collect_existing_input_paths(items, input_paths)
    entries = _prepare_batch_entries(
        items,
        root=root,
        mode=mode,
        protein_pdb=protein_pdb,
        reference_ligand=reference_ligand,
        reactive_residue=reactive_residue,
        defaults=defaults,
    )
    owned_directories = [
        entry.job_dir if entry.job_dir is not None else root / f"invalid-{entry.index:05d}" for entry in entries
    ]
    _validate_input_output_ownership(input_paths, resolved_root, owned_directories)
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    result_log = root / "results.jsonl"
    summary_path = root / "summary.json"
    # Publish the non-complete state before invalidating the previous result
    # list. A crash at any later point can therefore never leave an old
    # `status=complete` manifest describing the new attempt.
    _atomic_write_json(
        summary_path,
        {
            "status": "running",
            "total_items": 0,
            "results": 0,
            "successful": 0,
            "errors": 0,
            "skipped": 0,
        },
    )
    _atomic_write_text(result_log, "")
    total_items = 0
    error_count = 0
    skipped_count = 0

    for entry in entries:
        total_items = entry.index
        if entry.error is not None:
            error_count += 1
            failure = _preflight_failure(root, entry.index, entry.item, entry.error, mode)
            if on_error == "raise":
                raise entry.error
            if on_error == "skip":
                skipped_count += 1
            else:
                results.append(failure)
            continue
        job = entry.job
        job_id = entry.job_id
        job_dir = entry.job_dir
        result_path = entry.result_path
        options = entry.options
        batch_runtime_identity = entry.runtime_identity
        signature = entry.signature
        assert (
            job is not None
            and job_id is not None
            and job_dir is not None
            and result_path is not None
            and options is not None
            and batch_runtime_identity is not None
            and signature is not None
        )
        if resume and result_path.is_file():
            loaded = _load_resume_result(result_path, job_dir, signature)
            if loaded is not None:
                results.append(loaded)
                continue

        running_result: dict[str, object] = {
            "job_id": job_id,
            "job_signature": signature,
            "batch_runtime_identity": batch_runtime_identity,
            "success": False,
            "status": "running",
        }
        _atomic_write_json(result_path, running_result)

        try:
            result = _dispatch(job, job_dir, options)
            if not isinstance(result, dict):
                raise TypeError("docking dispatch must return a result mapping")
            result.update(
                {
                    "job_id": job_id,
                    "job_signature": signature,
                    "batch_runtime_identity": batch_runtime_identity,
                    "success": True,
                    "metadata": job.metadata,
                }
            )
            artifact = _validated_output_artifact(result.get("output_file"), job_dir)
            if artifact is None:
                raise RuntimeError("docking reported success without a non-empty output artifact")
            result["output_file"] = str(artifact)
            result["output_artifact_fingerprint"] = file_content_fingerprint(artifact)
            result["output_artifact_size"] = artifact.stat().st_size
            _atomic_write_json(result_path, result)
        except Exception as exc:
            error_count += 1
            result = {
                "job_id": job_id,
                "mode": job.mode,
                "job_signature": signature,
                "batch_runtime_identity": batch_runtime_identity,
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "metadata": job.metadata,
            }
            _atomic_write_json(result_path, result)
            if on_error == "raise":
                raise
            if on_error == "skip":
                skipped_count += 1
                continue
        results.append(result)

    result_log_payload = "".join(json.dumps(result, sort_keys=True, default=str) + "\n" for result in results)
    _atomic_write_text(result_log, result_log_payload)
    _atomic_write_json(
        summary_path,
        {
            "status": "complete",
            "total_items": total_items,
            "results": len(results),
            "successful": sum(item.get("success") is True for item in results),
            "errors": error_count,
            "skipped": skipped_count,
        },
    )
    return results
