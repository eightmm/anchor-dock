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
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from rdkit import Chem


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
            "reference", ligand, protein_pdb, id, reference_ligand, None,
            dict(options), dict(metadata or {}),
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
            "covalent", ligand, protein_pdb, id, None, reactive_residue,
            dict(options), dict(metadata or {}),
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
            "free", ligand, protein_pdb, id, None, None,
            dict(options), dict(metadata or {}),
        )


_LIGAND_FIELDS = ("ligand", "query_ligand", "smiles", "inchi", "structure")
_SUPPORTED_SUFFIXES = {
    ".sdf", ".mol", ".mol2", ".pdb", ".ent", ".smi", ".smiles", ".inchi", ".txt",
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson",
}
_COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".lzma"}
_BATCH_SCHEMA_VERSION = "1"


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
    if text[:1] in {"[", "{"} or (text[:1] == text[-1:] and text[:1] in {"\"", "'"}):
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
        stat = path.stat()
        return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def _job_signature(job: DockingJob, options: Mapping[str, object]) -> str:
    payload = {
        "schema": _BATCH_SCHEMA_VERSION,
        "mode": job.mode,
        "ligand": _file_identity(job.ligand),
        "protein": _file_identity(job.protein_pdb),
        "reference": _file_identity(job.reference_ligand),
        "reactive_residue": job.reactive_residue,
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
        "id", "name", "ligand_id", "mode", "protein_pdb", "protein", "receptor",
        "reference_ligand", "reference", "ref_ligand", "reactive_residue", "residue",
        "options", "metadata",
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


def _records_from_sdf(data: bytes, source_name: str) -> Iterator[LigandRecord]:
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(data), removeHs=True)
    for index, mol in enumerate(supplier, start=1):
        if mol is None:
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
) -> Iterator[DockingJob | LigandRecord]:
    for row in rows:
        yield _resolve_item_paths(_mapping_to_item(row), base_dir)


def _items_from_text(
    data: bytes,
    suffix: str,
    source_name: str,
    *,
    base_dir: Path | None = None,
) -> Iterator[DockingJob | LigandRecord]:
    text = data.decode("utf-8-sig")
    if suffix in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        yield from _items_from_rows((row for row in rows if isinstance(row, Mapping)), base_dir=base_dir)
        return
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            payload = payload.get("jobs") or payload.get("ligands") or [payload]
        if not isinstance(payload, list):
            raise ValueError("JSON batch input must be an object, list, or contain jobs/ligands")
        for item in payload:
            if isinstance(item, Mapping):
                yield _resolve_item_paths(_mapping_to_item(item), base_dir)
            else:
                yield _resolve_item_paths(LigandRecord(str(item)), base_dir)
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


def _items_from_path(path: Path) -> Iterator[DockingJob | LigandRecord]:
    if path.is_dir():
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            suffixes = [suffix.lower() for suffix in child.suffixes]
            logical_suffix = suffixes[-2] if suffixes and suffixes[-1] in _COMPRESSED_SUFFIXES and len(suffixes) > 1 else child.suffix.lower()
            if logical_suffix in _SUPPORTED_SUFFIXES:
                yield from _items_from_path(child)
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


def iter_batch_items(source: object) -> Iterator[DockingJob | LigandRecord]:
    """Normalize Python collections, tables, files, directories, and streams."""
    if isinstance(source, (DockingJob, LigandRecord)):
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
            yield _mapping_to_item(source)
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
        suffix = Path(str(name)).suffix.lower() or ".smi"
        if suffix == ".sdf":
            yield from _records_from_sdf(data, str(name))
        else:
            yield from _items_from_text(data, suffix, str(name))
        return
    if isinstance(source, (str, os.PathLike)):
        path = Path(os.fspath(source))
        if path.exists():
            yield from _items_from_path(path)
        else:
            yield LigandRecord(source)
        return
    if isinstance(source, bytes):
        yield from _items_from_text(source, ".smi", "memory.smi")
        return
    if isinstance(source, Iterable):
        for item in source:
            yield from iter_batch_items(item)
        return
    raise TypeError(f"unsupported batch input type: {type(source).__name__}")


def _materialize_job(
    item: DockingJob | LigandRecord,
    *,
    mode: str | None,
    protein_pdb: str | os.PathLike[str] | None,
    reference_ligand: str | os.PathLike[str] | None,
    reactive_residue: str | None,
) -> DockingJob:
    if isinstance(item, DockingJob):
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
        reference_ligand=(
            job.reference_ligand if job.reference_ligand is not None else reference_ligand
        ),
        reactive_residue=(
            job.reactive_residue if job.reactive_residue is not None else reactive_residue
        ),
    )
    if job.protein_pdb is None:
        raise ValueError("protein_pdb is required for every docking job")
    if job.mode == "reference" and job.reference_ligand is None:
        raise ValueError("reference_ligand is required for reference jobs")
    return job


def _dispatch(job: DockingJob, output_dir: Path, options: dict[str, object]) -> dict[str, object]:
    if job.mode == "reference":
        from .reference import dock_reference

        assert job.reference_ligand is not None
        return dock_reference(
            job.protein_pdb,  # type: ignore[arg-type]
            job.reference_ligand,
            job.ligand,
            output_dir,
            **options,
        )
    if job.mode == "covalent":
        from .covalent import dock_covalent

        return dock_covalent(
            job.protein_pdb,  # type: ignore[arg-type]
            job.ligand,
            job.reactive_residue,
            output_dir,
            **options,
        )
    from .free import dock_free

    return dock_free(job.protein_pdb, job.ligand, output_dir, **options)  # type: ignore[arg-type]


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
    Execution is sequential at the molecule level; each job still uses batched
    pose scoring and optimization and receptor contexts are cached globally.
    """
    if on_error not in {"record", "raise", "skip"}:
        raise ValueError("on_error must be record, raise or skip")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    result_log = root / "results.jsonl"

    used_job_ids: dict[str, int] = {}
    for index, item in enumerate(iter_batch_items(source), start=1):
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
            f"{job.mode}|{ligand_text}|{job.protein_pdb}|"
            f"{job.reference_ligand}|{job.reactive_residue}|{raw_id}"
        )
        base_job_id = _safe_id(raw_id, "ligand", identity)
        occurrence = used_job_ids.get(base_job_id, 0) + 1
        used_job_ids[base_job_id] = occurrence
        job_id = base_job_id if occurrence == 1 else f"{base_job_id}-{occurrence}"
        job_dir = root / job_id
        result_path = job_dir / "result.json"
        options = _coerce_options(defaults)
        options.update(_coerce_options(job.options))
        options.pop("output_dir", None)
        options.setdefault("verbose", False)
        signature = _job_signature(job, options)
        if resume and result_path.is_file():
            loaded = json.loads(result_path.read_text())
            if loaded.get("job_signature") == signature:
                loaded["resumed"] = True
                results.append(loaded)
                continue

        try:
            result = _dispatch(job, job_dir, options)
            result.update({
                "job_id": job_id,
                "job_signature": signature,
                "success": True,
                "metadata": job.metadata,
            })
            job_dir.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
        except Exception as exc:
            if on_error == "raise":
                raise
            if on_error == "skip":
                continue
            result = {
                "job_id": job_id,
                "mode": job.mode,
                "job_signature": signature,
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "metadata": job.metadata,
            }
            job_dir.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
        results.append(result)

    with result_log.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True, default=str) + "\n")
    return results
