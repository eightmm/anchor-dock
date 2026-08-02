from __future__ import annotations

import gzip
import importlib
import json
from pathlib import Path

from rdkit import Chem

from anchor_dock import DockingJob, dock_batch, dock_free
from anchor_dock.batch import iter_batch_items


def test_public_api_has_no_legacy_run_aliases() -> None:
    package = importlib.import_module("anchor_dock")
    expected = {"dock_reference", "dock_covalent", "dock_free", "dock_batch"}
    assert expected <= set(package.__all__)
    assert not any(name.startswith("run_") for name in package.__all__)
    assert not hasattr(package, "dock_reference_batch")
    for retired in ("lig_align", "cov_vina"):
        try:
            importlib.import_module(retired)
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError(f"retired namespace {retired} remains importable")


def test_free_docking_smoke(cys_pdb: Path, tmp_path: Path) -> None:
    result = dock_free(
        cys_pdb,
        "CCO",
        tmp_path / "free",
        num_confs=2,
        num_starts=3,
        opt_steps=2,
        opt_batch_size=3,
        top_k=2,
        device="cpu",
        verbose=False,
    )
    assert result["mode"] == "free"
    assert result["num_poses"] == 2
    assert Path(result["output_file"]).is_file()


def test_batch_parses_smi_csv_jsonl_sdf_and_compression(tmp_path: Path) -> None:
    smi = tmp_path / "ligands.smi"
    smi.write_text("CCO ethanol\nCCN ethylamine\n")
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text("smiles,name,mode\nCCC,propane,free\n")
    jsonl = tmp_path / "jobs.jsonl"
    jsonl.write_text(json.dumps({"smiles": "CCCC", "name": "butane", "mode": "free"}) + "\n")
    gz = tmp_path / "more.smi.gz"
    gz.write_bytes(gzip.compress(b"CO methanol\n"))
    mol = Chem.MolFromSmiles("c1ccccc1")
    sdf = tmp_path / "multi.sdf"
    writer = Chem.SDWriter(str(sdf))
    mol.SetProp("_Name", "benzene")
    writer.write(mol)
    writer.close()

    assert len(list(iter_batch_items(smi))) == 2
    assert len(list(iter_batch_items(csv_path))) == 1
    assert len(list(iter_batch_items(jsonl))) == 1
    assert len(list(iter_batch_items(gz))) == 1
    assert len(list(iter_batch_items(sdf))) == 1
    assert len(list(iter_batch_items(tmp_path))) == 6


def test_mixed_batch_and_resume_with_dispatch_stub(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_dispatch(job, output_dir, options):
        calls.append(job.mode)
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"mode": job.mode, "output_file": str(output_dir / "pose.sdf"), "best_score": -1.0}

    monkeypatch.setattr("anchor_dock.batch._dispatch", fake_dispatch)
    jobs = [
        DockingJob.reference("CCO", protein_pdb="p.pdb", reference_ligand="r.sdf", id="ref"),
        DockingJob.covalent("C=CC(=O)N", protein_pdb="p.pdb", reactive_residue="CYS1:A", id="cov"),
        DockingJob.free("CCN", protein_pdb="p.pdb", id="free"),
    ]
    first = dock_batch(jobs, output_dir=tmp_path / "batch")
    assert [result["mode"] for result in first] == ["reference", "covalent", "free"]
    assert calls == ["reference", "covalent", "free"]
    second = dock_batch(jobs, output_dir=tmp_path / "batch", resume=True)
    assert all(result["resumed"] for result in second)
    assert calls == ["reference", "covalent", "free"]


def test_csv_option_coercion_and_homogeneous_row_overrides(monkeypatch, tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []

    def fake_dispatch(job, output_dir, options):
        del job
        captured.append(options)
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"mode": "free", "output_file": str(output_dir / "pose.sdf"), "best_score": -1.0}

    monkeypatch.setattr("anchor_dock.batch._dispatch", fake_dispatch)
    table = tmp_path / "ligands.csv"
    table.write_text(
        'smiles,name,optimize,opt_steps,box_size,top_k\n'
        'CCO,ethanol,true,7,"[18, 20, 22]",3\n'
    )
    results = dock_batch(
        table,
        mode="free",
        protein_pdb="protein.pdb",
        output_dir=tmp_path / "typed",
    )
    assert results[0]["success"]
    assert captured == [{
        "optimize": True,
        "opt_steps": 7,
        "box_size": [18, 20, 22],
        "top_k": 3,
        "verbose": False,
    }]


def test_resume_invalidates_when_options_change(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_dispatch(job, output_dir, options):
        del job
        calls.append(int(options["opt_steps"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"mode": "free", "output_file": str(output_dir / "pose.sdf"), "best_score": -1.0}

    monkeypatch.setattr("anchor_dock.batch._dispatch", fake_dispatch)
    job = DockingJob.free("CCO", protein_pdb="protein.pdb", id="ethanol")
    root = tmp_path / "resume"
    first = dock_batch([job], output_dir=root, opt_steps=2)
    same = dock_batch([job], output_dir=root, opt_steps=2, resume=True)
    changed = dock_batch([job], output_dir=root, opt_steps=3, resume=True)
    assert calls == [2, 3]
    assert first[0]["job_signature"] == same[0]["job_signature"]
    assert first[0]["job_signature"] != changed[0]["job_signature"]
    assert same[0]["resumed"] is True
    assert "resumed" not in changed[0]


def test_manifest_paths_are_resolved_relative_to_manifest(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "protein.pdb").write_text("END\n")
    (inputs / "reference.sdf").write_text("$$$$\n")
    manifest = inputs / "jobs.jsonl"
    manifest.write_text(
        json.dumps({
            "mode": "reference",
            "smiles": "CCO",
            "protein_pdb": "protein.pdb",
            "reference_ligand": "reference.sdf",
        }) + "\n"
    )
    item = next(iter_batch_items(manifest))
    assert isinstance(item, DockingJob)
    assert Path(item.protein_pdb) == inputs / "protein.pdb"
    assert Path(item.reference_ligand) == inputs / "reference.sdf"
