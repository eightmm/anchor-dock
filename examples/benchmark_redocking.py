"""Known-pocket local pose-recovery benchmark; no ligand alignment.

Example: uv run --frozen python examples/benchmark_redocking.py --output report.json

Uses the bundled 10gs crystal pose and seeded rigid/torsional perturbations.
This is a local-refinement sanity check, not global docking or cross-docking.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolAlign

ROOT = Path(__file__).resolve().parents[1]


def fixed_frame_symmetry_rmsd(mol: Chem.Mol, reference, coordinates, *, max_matches: int = 100_000) -> list[float]:
    """Heavy-atom symmetry RMSDs in angstroms, without superposition.

    Arrays are reference[N,3] and coordinates[B,N,3] (or one [N,3] pose),
    matching the same heavy-atom graph/order. Explicit H is rejected instead
    of silently changing this mapping. Conjugated terminal groups are treated
    symmetrically by RDKit. A finite graph-automorphism cap is checked.
    """
    if isinstance(max_matches, bool) or not isinstance(max_matches, int) or max_matches < 1:
        raise ValueError("max_matches must be a positive integer")
    if mol.GetNumAtoms() < 1 or any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        raise ValueError("mol must contain nonempty heavy atoms only")
    reference = np.asarray(reference, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim == 2:
        coordinates = coordinates[None]
    expected = (mol.GetNumAtoms(), 3)
    if reference.shape != expected or coordinates.ndim != 3 or coordinates.shape[1:] != expected:
        raise ValueError("RMSD coordinates must match the molecule heavy-atom order")
    if coordinates.shape[0] < 1 or not np.isfinite(reference).all() or not np.isfinite(coordinates).all():
        raise ValueError("RMSD coordinates must be nonempty and finite")
    matches = len(mol.GetSubstructMatches(mol, uniquify=False, maxMatches=max_matches + 1))
    # CalcRMS additionally symmetrizes conjugated terminal groups. Bound their
    # extra permutations so its cap cannot silently truncate a terminal swap.
    symmetry_upper_bound = matches
    for atom in mol.GetAtoms():
        terminal_elements = [neighbor.GetAtomicNum() for neighbor in atom.GetNeighbors() if neighbor.GetDegree() == 1]
        for element in set(terminal_elements):
            symmetry_upper_bound *= math.factorial(terminal_elements.count(element))
    if symmetry_upper_bound > max_matches:
        raise ValueError("molecular symmetry exceeds max_matches; truncated RMSD is not reported")
    ref = Chem.Mol(mol)
    ref.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, point in enumerate(reference):
        conformer.SetAtomPosition(index, point)
    ref.AddConformer(conformer)
    probe = Chem.Mol(ref)
    result = []
    for pose in coordinates:
        for index, point in enumerate(pose):
            probe.GetConformer().SetAtomPosition(index, point)
        result.append(float(rdMolAlign.CalcRMS(
            probe, ref, maxMatches=max_matches, symmetrizeConjugatedTerminalGroups=True,
        )))
    return result


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source_metadata() -> dict:
    import anchor_dock
    from anchor_dock.core import features, optimization

    package = Path(anchor_dock.__file__).resolve().parent
    fingerprint = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        fingerprint.update(path.relative_to(package).as_posix().encode() + b"\0" + path.read_bytes())
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=package, capture_output=True, check=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"git_head": revision, "source_sha256": fingerprint.hexdigest(),
            "atom_typing_contract": features.ATOM_TYPING_VERSION,
            "optimization_contract": optimization.OPTIMIZATION_CONTRACT}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pose_validity(mol: Chem.Mol, reference: torch.Tensor, poses: torch.Tensor, receptor: torch.Tensor) -> list[dict]:
    """Numerical/geometry diagnostics, not a complete chemical validator."""
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    if bonds:
        left, right = torch.tensor(bonds, device=poses.device).T
        target = torch.linalg.vector_norm(reference[left] - reference[right], dim=-1)
        lengths = torch.linalg.vector_norm(poses[:, left] - poses[:, right], dim=-1)
        drift = (lengths - target).abs().amax(dim=1)
    else:
        drift = torch.zeros(poses.shape[0], dtype=poses.dtype, device=poses.device)
    minimum = torch.cdist(poses, receptor[None]).amin(dim=(1, 2))
    finite = torch.isfinite(poses).all(dim=(1, 2))
    return [{"finite": bool(is_finite), "max_bond_length_drift_angstrom": float(bond_drift),
             "minimum_receptor_heavy_distance_angstrom": float(distance),
             "passes_sanity": bool(is_finite and bond_drift <= 0.002 and distance >= 1.0)}
            for is_finite, bond_drift, distance in zip(finite.cpu(), drift.cpu(), minimum.cpu(), strict=True)]


def run_benchmark(args: argparse.Namespace) -> dict:
    # Select another immutable checkout before importing any AnchorDock module.
    if args.source_root:
        if any(name == "anchor_dock" or name.startswith("anchor_dock.") for name in sys.modules):
            raise RuntimeError("--source-root must precede importing AnchorDock")
        sys.path.insert(0, str(Path(args.source_root).resolve()))
    from anchor_dock.core.engine import DockingEngine
    from anchor_dock.core.io import load_receptor_context
    from anchor_dock.core.optimization import SE3PoseModel, optimize_pose_module
    from anchor_dock.core.scoring import scorer_fingerprint
    from anchor_dock.core.topology import find_rotatable_bonds

    source_at_start = _source_metadata()
    driver_at_start = _sha(Path(__file__))
    if args.poses_per_level < 1 or args.steps < 1 or args.threads < 1:
        raise ValueError("poses-per-level, steps and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ligand_path, receptor_path = Path(args.ligand).resolve(), Path(args.receptor).resolve()
    records = list(Chem.SDMolSupplier(str(ligand_path), removeHs=True))
    if len(records) != 1 or records[0] is None:
        raise ValueError("benchmark requires exactly one valid SDF molecule")
    mol = records[0]
    if len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumConformers() != 1:
        raise ValueError("ligand must have one component and one conformer")
    native = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32, device=device)
    pivot = int(torch.linalg.vector_norm(native - native.mean(dim=0), dim=1).argmin())
    labels, translations, rotations, torsion_sigmas = ["crystal_control"], [torch.zeros(3)], [torch.zeros(3)], [0.0]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    levels = (("mild", 0.5, 10.0, 5.0), ("moderate", 1.5, 30.0, 15.0))
    for label, translation_rms, rotation_degrees, torsion_degrees in levels:
        for _ in range(args.poses_per_level):
            labels.append(label)
            translations.append(torch.randn(3, generator=generator) * (translation_rms / math.sqrt(3)))
            rotations.append(torch.randn(3, generator=generator) * (math.radians(rotation_degrees) / math.sqrt(3)))
            torsion_sigmas.append(math.radians(torsion_degrees))
    count = len(labels)
    perturbation = SE3PoseModel(
        mol, native[None].repeat(count, 1, 1), pivot,
        native[pivot][None] + torch.stack(translations).to(device), torch.stack(rotations).to(device), device,
    )
    with torch.no_grad():
        angles = torch.randn(perturbation.kinematics.thetas.shape, generator=generator)
        angles *= torch.tensor(torsion_sigmas)[:, None]
        perturbation.kinematics.thetas.copy_(angles.to(device))
        initial = perturbation().detach()
    initial_rmsds = fixed_frame_symmetry_rmsd(mol, native.cpu(), initial.cpu())
    pose_hash = hashlib.sha256(initial.cpu().contiguous().numpy().tobytes()).hexdigest()
    _sync(device)
    started = time.perf_counter()
    receptor = load_receptor_context(receptor_path, device)
    receptor_heavy_mask = torch.tensor([atom.GetAtomicNum() > 1 for atom in receptor.mol.GetAtoms()], device=device)
    receptor_heavy = receptor.coords[receptor_heavy_mask]
    if receptor_heavy.shape[0] == 0:
        raise ValueError("benchmark receptor must contain heavy atoms")
    engine = DockingEngine(args.scorer, device=device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem = engine.prepare(mol, receptor, num_rotatable_bonds=len(find_rotatable_bonds(mol)))
    _sync(device)
    preparation_seconds = time.perf_counter() - started
    with torch.no_grad():
        crystal_raw = problem.scorer.raw_components(native[None])
        initial_raw = problem.scorer.raw_components(initial)
        initial_report = problem.scorer.report(initial_raw, crystal_raw.intramolecular[0])
    initial_validity = _pose_validity(mol, native, initial, receptor_heavy)
    runs = []
    for optimizer in args.optimizers:
        lr = {"lbfgs": args.lbfgs_learning_rate, "cartesian": args.cartesian_learning_rate}.get(optimizer, args.learning_rate)
        # Dense Cartesian pullback trials evaluate the scorer once per row;
        # single-pose chunks avoid rescoring unchanged neighboring poses.
        chunks = [(index, index + 1) for index in range(count)] if optimizer in ("lbfgs", "cartesian") else [(0, count)]
        outputs, stats = [], []
        _sync(device)
        started = time.perf_counter()
        for start, stop in chunks:
            model = SE3PoseModel(
                mol, initial[start:stop], pivot, initial[start:stop, pivot],
                torch.zeros(stop - start, 3, device=device), device,
            )
            final, stat = optimize_pose_module(
                model, problem.scorer.search_energy, num_steps=args.steps,
                learning_rate=lr, optimizer=optimizer, early_stopping=False,
            )
            outputs.append(final.detach())
            stats.append(stat.as_dict())
        _sync(device)
        elapsed = time.perf_counter() - started
        final = torch.cat(outputs)
        rmsds = fixed_frame_symmetry_rmsd(mol, native.cpu(), final.cpu())
        validity = _pose_validity(mol, native, final, receptor_heavy)
        with torch.no_grad():
            raw = problem.scorer.raw_components(final)
            reported = problem.scorer.report(raw, crystal_raw.intramolecular[0])
        rows = [{
            "index": index, "level": label,
            "initial_rmsd_angstrom": initial_rmsds[index], "final_rmsd_angstrom": rmsds[index],
            "initial_search_energy": float(initial_raw.search_energy[index]), "final_search_energy": float(raw.search_energy[index]),
            "initial_reported_score": float(initial_report.score[index]), "final_reported_score": float(reported.score[index]),
            "initial_intermolecular": float(initial_raw.intermolecular[index]), "final_intermolecular": float(raw.intermolecular[index]),
            "initial_intramolecular": float(initial_raw.intramolecular[index]), "final_intramolecular": float(raw.intramolecular[index]),
            "initial_validity": initial_validity[index], "final_validity": validity[index],
            "initial_within_2_angstrom": bool(initial_rmsds[index] <= 2.0 and initial_validity[index]["passes_sanity"]),
            "final_within_2_angstrom": bool(rmsds[index] <= 2.0 and validity[index]["passes_sanity"]),
        } for index, label in enumerate(labels)]
        groups = {}
        for level in ("mild", "moderate"):
            selected = [row for row in rows if row["level"] == level]
            groups[level] = {"poses": len(selected),
                "initial_median_rmsd_angstrom": statistics.median(row["initial_rmsd_angstrom"] for row in selected),
                "final_median_rmsd_angstrom": statistics.median(row["final_rmsd_angstrom"] for row in selected),
                "initial_within_2_angstrom": sum(row["initial_within_2_angstrom"] for row in selected),
                "final_within_2_angstrom": sum(row["final_within_2_angstrom"] for row in selected),
                "newly_recovered": sum(not row["initial_within_2_angstrom"] and row["final_within_2_angstrom"] for row in selected),
                "lost": sum(row["initial_within_2_angstrom"] and not row["final_within_2_angstrom"] for row in selected)}
        runs.append({"optimizer": optimizer, "learning_rate": lr, "steps_per_pose": args.steps,
            "optimization_seconds": elapsed, "poses": rows, "by_level": groups,
            "initial_within_2_angstrom": sum(row["initial_within_2_angstrom"] for row in rows[1:]),
            "final_within_2_angstrom": sum(row["final_within_2_angstrom"] for row in rows[1:]),
            "newly_recovered": sum(not row["initial_within_2_angstrom"] and row["final_within_2_angstrom"] for row in rows[1:]),
            "lost": sum(row["initial_within_2_angstrom"] and not row["final_within_2_angstrom"] for row in rows[1:]),
            "total_perturbed": len(rows) - 1,
            "all_search_energies_nonincreasing": all(row["final_search_energy"] <= row["initial_search_energy"] + 1e-5 for row in rows),
            "stats": stats})
    return {
        "benchmark": "known-pocket seeded local pose recovery", "label": args.label,
        "complex_id": args.complex_id or ligand_path.stem,
        "limitations": ["One crystal complex; not a population success-rate estimate.",
            "Crystal-derived starts; not conformer generation, global docking or cross-docking.",
            "Inferred chemistry; no authoritative prepared-receptor parity claim.",
            "One timing sample per optimizer; no throughput/speedup claim.",
            "Sanity checks cover finite coordinates, bond drift and <1A heavy clashes only."],
        "source": source_at_start,
        "source_tree_changed_during_run": source_at_start["source_sha256"] != _source_metadata()["source_sha256"],
        "benchmark_driver_sha256": driver_at_start,
        "benchmark_driver_changed_during_run": driver_at_start != _sha(Path(__file__)),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
            "rdkit": importlib.metadata.version("rdkit"), "device": str(device),
            "cuda_available": torch.cuda.is_available(), "torch_threads": args.threads},
        "inputs": {"ligand_sha256": _sha(ligand_path), "receptor_sha256": _sha(receptor_path),
            "source_urls": args.source_url, "ligand_filename": ligand_path.name, "receptor_filename": receptor_path.name,
            "ligand_heavy_atoms": mol.GetNumAtoms(), "receptor_atoms": receptor.coords.shape[0],
            "receptor_heavy_atoms": receptor_heavy.shape[0],
            "rdkit_strict_rotors": len(find_rotatable_bonds(mol)), "pivot_atom_index": pivot,
            "seed": args.seed, "initial_poses_sha256": pose_hash, "poses_per_level": args.poses_per_level,
            "levels": [{"name": label, "translation_rms_angstrom": t, "rotation_vector_rms_degrees": r,
                        "independent_torsion_std_degrees": d} for label, t, r, d in levels]},
        "metric": {"name": "RDKit CalcRMS, fixed receptor frame", "ligand_alignment": False,
            "symmetrize_conjugated_terminal_groups": True, "max_matches": 100_000,
            "reporting_reference": "same crystal intramolecular energy across starts/optimizers"},
        "preparation_seconds": preparation_seconds,
        "preparation_warnings": sorted({str(item.message) for item in caught}),
        "scorer": engine.scorer_name, "scorer_fingerprint": scorer_fingerprint(engine.scoring_model),
        "score_units": engine.score_units, "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand", default=ROOT / "examples/10gs/10gs_ligand.sdf", type=Path)
    parser.add_argument("--receptor", default=ROOT / "examples/10gs/10gs_pocket.pdb", type=Path)
    parser.add_argument("--source-root", type=Path, help="alternative checkout's src directory; fresh process only")
    parser.add_argument("--label", default="current")
    parser.add_argument("--complex-id")
    parser.add_argument("--source-url", action="append", default=[], help="input provenance URL; repeat as needed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scorer", choices=("vina", "vinardo", "softdock"), default="vina")
    parser.add_argument("--optimizers", nargs="+", default=["adam", "lbfgs"])
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--lbfgs-learning-rate", type=float, default=0.5)
    parser.add_argument("--cartesian-learning-rate", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--poses-per-level", type=int, default=4)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"label": result["label"], "device": result["environment"]["device"], "runs": [{
        "optimizer": run["optimizer"], "seconds": run["optimization_seconds"],
        "final_within_2_angstrom": f'{run["final_within_2_angstrom"]}/{run["total_perturbed"]}', "by_level": run["by_level"],
    } for run in result["runs"]]}, indent=2))


if __name__ == "__main__":
    main()
