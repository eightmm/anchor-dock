"""Batch covalent docking with receptor feature reuse."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .pipeline import dock_covalent, load_pocket_for_caching


def _parse_ligands(ligands: str | Iterable[str]) -> list[tuple[str, str]]:
    if not isinstance(ligands, str):
        return [(value, f"ligand_{idx:04d}") for idx, value in enumerate(ligands, start=1)]
    path = Path(ligands)
    if path.is_file() and path.suffix.lower() in {".smi", ".smiles", ".txt"}:
        result: list[tuple[str, str]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            result.append((fields[0], fields[1] if len(fields) > 1 else f"ligand_{len(result)+1:04d}"))
        return result
    return [(ligands, "ligand_0001")]


def dock_covalent_batch(
    protein_pdb: str,
    ligands: str | Iterable[str],
    reactive_residue: str | None = None,
    output_dir: str = "results",
    *,
    pocket_cutoff: float = 12.0,
    device=None,
    verbose: bool = True,
    **kwargs,
) -> list[dict[str, object]]:
    """Dock one or many ligands using one cached pocket."""
    parsed = _parse_ligands(ligands)
    cache = load_pocket_for_caching(
        protein_pdb, reactive_residue, pocket_cutoff, device, verbose=verbose
    )
    results: list[dict[str, object]] = []
    for index, (smiles, name) in enumerate(parsed, start=1):
        try:
            result = dock_covalent(
                protein_pdb,
                smiles,
                reactive_residue=reactive_residue,
                output_dir=str(Path(output_dir) / name),
                pocket_cutoff=pocket_cutoff,
                _cached_pocket=cache,
                device=device,
                verbose=False,
                **kwargs,
            )
            result.update({"ligand": smiles, "name": name, "success": True})
        except Exception as exc:
            result = {"ligand": smiles, "name": name, "success": False, "error": str(exc)}
        results.append(result)
        if verbose:
            status = "ok" if result["success"] else f"failed: {result['error']}"
            print(f"[{index}/{len(parsed)}] {name}: {status}")
    return results


run_batch_docking = dock_covalent_batch
