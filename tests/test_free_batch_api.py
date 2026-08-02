from __future__ import annotations

import gzip
import io
import json
import os
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from rdkit import Chem

from anchor_dock import DockingJob, dock_batch, dock_free
from anchor_dock.batch import iter_batch_items


def _pose_result(output_dir: Path, mode: str = "free") -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "pose.sdf"
    output.write_text("pose\n")
    return {"mode": mode, "output_file": str(output), "best_score": -1.0}


class _SingleValueReiterable:
    def __init__(self, value: object) -> None:
        self.value = value

    def __iter__(self):
        yield self.value


def _expected_job_directory(root: Path, job: DockingJob) -> Path:
    from anchor_dock.batch import _canonical_ligand_text, _safe_id

    raw_id = job.id or "ligand"
    identity = (
        f"{job.mode}|{_canonical_ligand_text(job.ligand)}|{job.protein_pdb}|"
        f"{job.reference_ligand}|{job.reactive_residue}|{raw_id}"
    )
    return root / _safe_id(raw_id, "ligand", identity)


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
    assert result["torsion_penalty_requested"] is True
    assert result["torsion_penalty_applied"] is False
    assert result["score_rotatable_bonds"] == 0
    assert result["best_score"] == result["best_search_energy"]
    assert Path(result["output_file"]).is_file()


def test_free_rotation_starts_are_seeded_and_haar_uniform() -> None:
    from anchor_dock.free import _sample_uniform_rotation_vectors

    first_generator = torch.Generator().manual_seed(17)
    second_generator = torch.Generator().manual_seed(17)
    first = _sample_uniform_rotation_vectors(20_000, first_generator)
    second = _sample_uniform_rotation_vectors(20_000, second_generator)
    assert torch.equal(first, second)
    mean_angle = torch.linalg.vector_norm(first, dim=1).mean()
    assert float(mean_angle) == pytest.approx(torch.pi / 2 + 2 / torch.pi, abs=0.03)


def test_zero_step_optimization_is_recorded_as_requested_but_not_applied(
    cys_pdb: Path,
    tmp_path: Path,
) -> None:
    result = dock_free(
        cys_pdb,
        "CCO",
        tmp_path / "zero-step",
        num_confs=2,
        num_starts=2,
        optimize=True,
        opt_steps=0,
        top_k=1,
        device="cpu",
        verbose=False,
    )
    assert result["optimization_requested"] is True
    assert result["optimization_applied"] is False
    assert result["optimization_improved"] is False
    assert result["optimized"] is False
    pose = next(mol for mol in Chem.SDMolSupplier(result["output_file"]) if mol is not None)
    assert pose.GetProp("AnchorDock_Optimization_Requested") == "True"
    assert pose.GetProp("AnchorDock_Optimization_Applied") == "False"
    assert pose.GetProp("AnchorDock_Search_Method") == "multistart_random_placement"


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

    stream = io.BytesIO(gzip.compress(b"CN methylamine\n"))
    stream.name = "memory.smi.gz"
    assert len(list(iter_batch_items(stream))) == 1


@pytest.mark.parametrize("container_key", ["jobs", "ligands"])
def test_empty_json_batch_container_completes_with_zero_items(
    tmp_path: Path,
    container_key: str,
) -> None:
    source = tmp_path / f"empty-{container_key}.json"
    source.write_text(json.dumps({container_key: []}))
    root = tmp_path / f"output-{container_key}"

    results = dock_batch(source, output_dir=root)

    assert results == []
    assert (root / "results.jsonl").read_text() == ""
    assert json.loads((root / "summary.json").read_text()) == {
        "status": "complete",
        "errors": 0,
        "results": 0,
        "skipped": 0,
        "successful": 0,
        "total_items": 0,
    }


def test_json_batch_rejects_ambiguous_container_keys(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.json"
    source.write_text(json.dumps({"jobs": [], "ligands": []}))

    with pytest.raises(ValueError, match="both 'jobs' and 'ligands'"):
        list(iter_batch_items(source))


@pytest.mark.parametrize("wrap_directory", [False, True])
def test_batch_directory_source_excludes_nested_output_tree(
    monkeypatch,
    tmp_path: Path,
    wrap_directory: bool,
) -> None:
    calls: list[str] = []

    def fake_dispatch(job, output_dir, options):
        del options
        calls.append(str(job.ligand))
        return _pose_result(output_dir)

    monkeypatch.setattr("anchor_dock.batch._dispatch", fake_dispatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "ligands.smi").write_text("CCO ethanol\n")
    root = source / "output"
    root.mkdir()
    (root / "stale-job.json").write_text(json.dumps({"smiles": "CCC", "name": "stale"}))

    input_source: object = [source] if wrap_directory else source
    results = dock_batch(
        input_source,
        mode="free",
        protein_pdb="protein.pdb",
        output_dir=root,
    )

    assert len(results) == 1
    assert results[0]["success"] is True
    assert calls == ["CCO"]
    summary = json.loads((root / "summary.json").read_text())
    assert summary["total_items"] == 1
    assert summary["errors"] == 0


def test_batch_preserves_explicit_file_inside_output_tree(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "anchor_dock.batch._dispatch",
        lambda job, output_dir, options: _pose_result(output_dir, job.mode),
    )
    root = tmp_path / "output"
    root.mkdir()
    explicit = root / "explicit.smi"
    explicit.write_text("CCN ethylamine\n")

    results = dock_batch(
        explicit,
        mode="free",
        protein_pdb="protein.pdb",
        output_dir=root,
    )

    assert len(results) == 1
    assert results[0]["success"] is True


@pytest.mark.parametrize("wrap_directory", [False, True])
def test_batch_rejects_same_source_and_output_before_writing_manifests(
    tmp_path: Path,
    wrap_directory: bool,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "ligands.smi").write_text("CCO ethanol\n")
    input_source: object = [source] if wrap_directory else source

    with pytest.raises(ValueError, match="same directory"):
        dock_batch(
            input_source,
            mode="free",
            protein_pdb="protein.pdb",
            output_dir=source,
        )

    assert not (source / "summary.json").exists()
    assert not (source / "results.jsonl").exists()


@pytest.mark.parametrize("manifest_name", ["results.jsonl", "summary.json"])
@pytest.mark.parametrize("wrap_file", [False, True])
def test_batch_rejects_input_file_that_is_an_output_manifest_before_overwrite(
    tmp_path: Path,
    manifest_name: str,
    wrap_file: bool,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    source = root / manifest_name
    original = b'{"smiles":"CCO","name":"preserve-me"}\n'
    source.write_bytes(original)
    input_source: object = [source] if wrap_file else source

    with pytest.raises(ValueError, match="conflicts with an output manifest"):
        dock_batch(
            input_source,
            mode="free",
            protein_pdb="protein.pdb",
            output_dir=root,
        )

    assert source.read_bytes() == original
    sibling = root / ({"results.jsonl", "summary.json"} - {manifest_name}).pop()
    assert not sibling.exists()


@pytest.mark.parametrize("manifest_name", ["results.jsonl", "summary.json"])
@pytest.mark.parametrize("nested", [False, True])
def test_batch_preflights_generator_manifest_collisions_before_overwrite(
    tmp_path: Path,
    manifest_name: str,
    nested: bool,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    source = root / manifest_name
    original = b'{"smiles":"CCO","name":"preserve-generator-source"}\n'
    source.write_bytes(original)

    def inputs():
        yield source

    input_source: object = [inputs()] if nested else inputs()
    with pytest.raises(ValueError, match="conflicts with an output manifest"):
        dock_batch(
            input_source,
            mode="free",
            protein_pdb="protein.pdb",
            output_dir=root,
        )

    assert source.read_bytes() == original
    sibling = root / ({"results.jsonl", "summary.json"} - {manifest_name}).pop()
    assert not sibling.exists()


def test_batch_preflights_generator_source_output_directory_collision(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (root / "ligands.smi").write_text("CCO ethanol\n")

    def inputs():
        yield root

    with pytest.raises(ValueError, match="same directory"):
        dock_batch(
            inputs(),
            mode="free",
            protein_pdb="protein.pdb",
            output_dir=root,
        )

    assert not (root / "summary.json").exists()
    assert not (root / "results.jsonl").exists()


def test_batch_processes_normal_one_shot_generator(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_dispatch(job, output_dir, options):
        del options
        calls.append(str(job.ligand))
        return _pose_result(output_dir)

    monkeypatch.setattr("anchor_dock.batch._dispatch", fake_dispatch)

    def inputs():
        yield "CCO"
        yield "CCN"

    results = dock_batch(
        inputs(),
        mode="free",
        protein_pdb="protein.pdb",
        output_dir=tmp_path / "output",
    )

    assert len(results) == 2
    assert all(result["success"] is True for result in results)
    assert calls == ["CCO", "CCN"]


@pytest.mark.parametrize("manifest_name", ["results.jsonl", "summary.json"])
def test_batch_rejects_nonexistent_reserved_manifest_source_path(
    tmp_path: Path,
    manifest_name: str,
) -> None:
    root = tmp_path / "output"

    with pytest.raises(ValueError, match="conflicts with an output manifest"):
        dock_batch(
            root / manifest_name,
            mode="free",
            protein_pdb="protein.pdb",
            output_dir=root,
        )

    assert not root.exists()


@pytest.mark.parametrize("manifest_name", ["results.jsonl", "summary.json"])
def test_batch_rejects_file_like_manifest_input_before_read_or_overwrite(
    tmp_path: Path,
    manifest_name: str,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    source = root / manifest_name
    original = b'{"smiles":"CCO","name":"file-like"}\n'
    source.write_bytes(original)

    with source.open("rb") as stream:
        with pytest.raises(ValueError, match="conflicts with an output manifest"):
            dock_batch(
                stream,
                mode="free",
                protein_pdb="protein.pdb",
                output_dir=root,
            )
        assert stream.tell() == 0

    assert source.read_bytes() == original
    sibling = root / ({"results.jsonl", "summary.json"} - {manifest_name}).pop()
    assert not sibling.exists()


@pytest.mark.parametrize("manifest_name", ["results.jsonl", "summary.json"])
@pytest.mark.parametrize("field", ["ligand", "protein_pdb", "reference_ligand"])
def test_batch_rejects_docking_job_fields_that_collide_with_manifests(
    tmp_path: Path,
    manifest_name: str,
    field: str,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    source = root / manifest_name
    original = b"job-field-input\n"
    source.write_bytes(original)
    if field == "ligand":
        job = DockingJob.free(source, protein_pdb="protein.pdb", id="field")
    elif field == "protein_pdb":
        job = DockingJob.free("CCO", protein_pdb=source, id="field")
    else:
        job = DockingJob.reference(
            "CCO",
            protein_pdb="protein.pdb",
            reference_ligand=source,
            id="field",
        )

    with pytest.raises(ValueError, match="conflicts with an output manifest"):
        dock_batch([job], output_dir=root)

    assert source.read_bytes() == original
    sibling = root / ({"results.jsonl", "summary.json"} - {manifest_name}).pop()
    assert not sibling.exists()


@pytest.mark.parametrize("manifest_name", ["results.jsonl", "summary.json"])
@pytest.mark.parametrize("field", ["protein_pdb", "reference_ligand"])
def test_batch_rejects_homogeneous_default_paths_that_collide_with_manifests(
    tmp_path: Path,
    manifest_name: str,
    field: str,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    source = root / manifest_name
    original = b"homogeneous-input\n"
    source.write_bytes(original)
    kwargs: dict[str, object] = {
        "mode": "free",
        "protein_pdb": "protein.pdb",
        "output_dir": root,
    }
    kwargs[field] = source

    with pytest.raises(ValueError, match="conflicts with an output manifest"):
        dock_batch("CCO", **kwargs)

    assert source.read_bytes() == original
    sibling = root / ({"results.jsonl", "summary.json"} - {manifest_name}).pop()
    assert not sibling.exists()


@pytest.mark.parametrize("manifest_name", ["results.jsonl", "summary.json"])
def test_batch_preflights_custom_reiterable_manifest_collision(
    tmp_path: Path,
    manifest_name: str,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    source = root / manifest_name
    original = b'{"smiles":"CCO","name":"reiterable"}\n'
    source.write_bytes(original)

    with pytest.raises(ValueError, match="conflicts with an output manifest"):
        dock_batch(
            _SingleValueReiterable(source),
            mode="free",
            protein_pdb="protein.pdb",
            output_dir=root,
        )

    assert source.read_bytes() == original


def test_batch_preflights_custom_reiterable_output_directory_collision(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (root / "ligands.smi").write_text("CCO ethanol\n")

    with pytest.raises(ValueError, match="same directory"):
        dock_batch(
            _SingleValueReiterable(root),
            mode="free",
            protein_pdb="protein.pdb",
            output_dir=root,
        )

    assert not (root / "summary.json").exists()
    assert not (root / "results.jsonl").exists()


def test_batch_rejects_json_source_at_its_computed_result_path(tmp_path: Path) -> None:
    root = tmp_path / "output"
    job = DockingJob.free("CCO", protein_pdb="protein.pdb", id="collision")
    job_dir = _expected_job_directory(root, job)
    job_dir.mkdir(parents=True)
    source = job_dir / "result.json"
    original = json.dumps(
        {
            "smiles": "CCO",
            "name": "collision",
            "mode": "free",
            "protein_pdb": "protein.pdb",
        }
    )
    source.write_text(original)

    with pytest.raises(ValueError, match="conflicts with a batch job output directory"):
        dock_batch(source, output_dir=root)

    assert source.read_text() == original
    assert not (root / "summary.json").exists()
    assert not (root / "results.jsonl").exists()


def test_batch_rejects_job_input_resolving_inside_its_own_output_directory(tmp_path: Path) -> None:
    root = tmp_path / "output"
    alias = tmp_path / "protein-link.pdb"
    job = DockingJob.free("CCO", protein_pdb=alias, id="self-input")
    job_dir = _expected_job_directory(root, job)
    job_dir.mkdir(parents=True)
    source = job_dir / "protein.pdb"
    original = b"self-owned-input\n"
    source.write_bytes(original)
    alias.symlink_to(source)

    with pytest.raises(ValueError, match="conflicts with a batch job output directory"):
        dock_batch([job], output_dir=root)

    assert source.read_bytes() == original
    assert alias.read_bytes() == original
    assert not (root / "summary.json").exists()
    assert not (root / "results.jsonl").exists()


def test_batch_rejects_future_job_input_inside_an_earlier_job_output_directory(tmp_path: Path) -> None:
    root = tmp_path / "output"
    first = DockingJob.free("CCO", protein_pdb="protein.pdb", id="first")
    first_dir = _expected_job_directory(root, first)
    first_dir.mkdir(parents=True)
    source = first_dir / "future.smi"
    original = b"CCN future\n"
    source.write_bytes(original)
    second = DockingJob.free(source, protein_pdb="protein.pdb", id="second")

    with pytest.raises(ValueError, match="conflicts with a batch job output directory"):
        dock_batch([first, second], output_dir=root)

    assert source.read_bytes() == original
    assert not (root / "summary.json").exists()
    assert not (root / "results.jsonl").exists()


def test_batch_rejects_invalid_result_path_used_as_raw_source(tmp_path: Path) -> None:
    root = tmp_path / "output"
    invalid_dir = root / "invalid-00001"
    invalid_dir.mkdir(parents=True)
    source = invalid_dir / "result.json"
    original = "{}"
    source.write_text(original)

    with pytest.raises(ValueError, match="conflicts with a batch job output directory"):
        dock_batch(source, output_dir=root)

    assert source.read_text() == original
    assert not (root / "summary.json").exists()
    assert not (root / "results.jsonl").exists()


@pytest.mark.parametrize("manifest_name", ["results.jsonl", "summary.json"])
def test_batch_atomic_manifest_write_does_not_follow_predictable_temp_symlink(
    monkeypatch,
    tmp_path: Path,
    manifest_name: str,
) -> None:
    monkeypatch.setattr(
        "anchor_dock.batch._dispatch",
        lambda job, output_dir, options: _pose_result(output_dir, job.mode),
    )
    root = tmp_path / "output"
    root.mkdir()
    external = tmp_path / "external-input"
    original = b"must-not-be-overwritten\n"
    external.write_bytes(original)
    predictable = root / f".{manifest_name}.{os.getpid()}.tmp"
    predictable.symlink_to(external)

    results = dock_batch(
        [DockingJob.free("CCO", protein_pdb=external, id="atomic")],
        output_dir=root,
    )

    assert results[0]["success"] is True
    assert predictable.is_symlink()
    assert external.read_bytes() == original


def test_batch_rejects_dangling_symlink_input_inside_job_output_directory(tmp_path: Path) -> None:
    root = tmp_path / "output"
    first = DockingJob.free("CCO", protein_pdb="protein.pdb", id="first")
    first_dir = _expected_job_directory(root, first)
    first_dir.mkdir(parents=True)
    dangling = first_dir / "result.json"
    dangling.symlink_to(tmp_path / "missing-protein.pdb")
    second = DockingJob.free("CCN", protein_pdb=dangling, id="second")

    with pytest.raises(ValueError, match="conflicts with a batch job output directory"):
        dock_batch([first, second], output_dir=root)

    assert dangling.is_symlink()
    assert not (root / "summary.json").exists()
    assert not (root / "results.jsonl").exists()


def test_mixed_batch_and_resume_with_dispatch_stub(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_dispatch(job, output_dir, options):
        del options
        calls.append(job.mode)
        return _pose_result(output_dir, job.mode)

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
        return _pose_result(output_dir)

    monkeypatch.setattr("anchor_dock.batch._dispatch", fake_dispatch)
    table = tmp_path / "ligands.csv"
    table.write_text('smiles,name,optimize,opt_steps,box_size,top_k\nCCO,ethanol,true,7,"[18, 20, 22]",3\n')
    results = dock_batch(
        table,
        mode="free",
        protein_pdb="protein.pdb",
        output_dir=tmp_path / "typed",
    )
    assert results[0]["success"]
    assert captured == [
        {
            "optimize": True,
            "opt_steps": 7,
            "box_size": [18, 20, 22],
            "top_k": 3,
            "verbose": False,
        }
    ]


def test_resume_invalidates_when_options_change(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_dispatch(job, output_dir, options):
        del job
        calls.append(int(options["opt_steps"]))
        return _pose_result(output_dir)

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


def test_resume_invalidates_when_signature_epoch_or_output_schema_changes(monkeypatch, tmp_path: Path) -> None:
    import anchor_dock.batch as batch_module

    calls = 0

    def fake_dispatch(job, output_dir, options):
        nonlocal calls
        del job, options
        calls += 1
        return _pose_result(output_dir)

    monkeypatch.setattr(batch_module, "_dispatch", fake_dispatch)
    job = DockingJob.free("CCO", protein_pdb="protein.pdb", id="epoch")
    root = tmp_path / "resume-epoch"
    current_epoch = batch_module._BATCH_SCHEMA_VERSION
    current_output_schema = batch_module.OUTPUT_SCHEMA_VERSION
    assert current_epoch == "3"
    assert current_output_schema == "2"

    monkeypatch.setattr(batch_module, "_BATCH_SCHEMA_VERSION", "2")
    previous = dock_batch([job], output_dir=root)[0]
    monkeypatch.setattr(batch_module, "_BATCH_SCHEMA_VERSION", current_epoch)
    current = dock_batch([job], output_dir=root, resume=True)[0]
    monkeypatch.setattr(batch_module, "OUTPUT_SCHEMA_VERSION", "future")
    future = dock_batch([job], output_dir=root, resume=True)[0]

    assert calls == 3
    assert previous["job_signature"] != current["job_signature"]
    assert current["job_signature"] != future["job_signature"]
    assert "resumed" not in current
    assert "resumed" not in future


def test_resume_signature_tracks_resolved_device_and_runtime_versions(monkeypatch, tmp_path: Path) -> None:
    import anchor_dock.batch as batch_module

    calls = 0

    def fake_dispatch(job, output_dir, options):
        nonlocal calls
        del job, options
        calls += 1
        return _pose_result(output_dir)

    monkeypatch.setattr(batch_module, "_dispatch", fake_dispatch)
    monkeypatch.setattr(batch_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(batch_module.torch, "__version__", "torch-a")
    monkeypatch.setattr(batch_module.np, "__version__", "numpy-a")
    monkeypatch.setattr(batch_module.rdBase, "rdkitVersion", "rdkit-a")
    job = DockingJob.free("CCO", protein_pdb="protein.pdb", id="runtime")
    root = tmp_path / "runtime-signature"

    results = [dock_batch([job], output_dir=root)[0]]
    monkeypatch.setattr(batch_module.torch, "__version__", "torch-b")
    results.append(dock_batch([job], output_dir=root, resume=True)[0])
    monkeypatch.setattr(batch_module.np, "__version__", "numpy-b")
    results.append(dock_batch([job], output_dir=root, resume=True)[0])
    monkeypatch.setattr(batch_module.rdBase, "rdkitVersion", "rdkit-b")
    results.append(dock_batch([job], output_dir=root, resume=True)[0])
    monkeypatch.setattr(batch_module.torch.cuda, "is_available", lambda: True)
    results.append(dock_batch([job], output_dir=root, resume=True)[0])

    assert calls == 5
    assert len({result["job_signature"] for result in results}) == 5
    assert results[0]["batch_runtime_identity"] == {
        "device": "cpu",
        "numpy": "numpy-a",
        "rdkit": "rdkit-a",
        "torch": "torch-a",
    }
    assert results[-1]["batch_runtime_identity"] == {
        "device": "cuda",
        "numpy": "numpy-b",
        "rdkit": "rdkit-b",
        "torch": "torch-b",
    }


def test_resume_retries_failure_missing_and_external_artifacts(monkeypatch, tmp_path: Path) -> None:
    calls = 0

    def fake_dispatch(job, output_dir, options):
        nonlocal calls
        del job, options
        calls += 1
        return _pose_result(output_dir)

    monkeypatch.setattr("anchor_dock.batch._dispatch", fake_dispatch)
    job = DockingJob.free("CCO", protein_pdb="protein.pdb", id="ethanol")
    root = tmp_path / "resume-integrity"
    result = dock_batch([job], output_dir=root)[0]
    assert result["output_artifact_fingerprint"].startswith("sha256:")
    result_path = next(root.glob("*/result.json"))

    Path(result["output_file"]).unlink()
    result = dock_batch([job], output_dir=root, resume=True)[0]
    assert calls == 2 and "resumed" not in result

    cached = json.loads(result_path.read_text())
    cached["success"] = False
    result_path.write_text(json.dumps(cached))
    result = dock_batch([job], output_dir=root, resume=True)[0]
    assert calls == 3 and result["success"] is True

    external = tmp_path / "external.sdf"
    external.write_text("external\n")
    cached = json.loads(result_path.read_text())
    cached["output_file"] = str(external)
    result_path.write_text(json.dumps(cached))
    result = dock_batch([job], output_dir=root, resume=True)[0]
    assert calls == 4 and "resumed" not in result

    Path(result["output_file"]).write_text("fake\n")
    result = dock_batch([job], output_dir=root, resume=True)[0]
    assert calls == 5 and "resumed" not in result


def test_preflight_error_is_recorded_and_next_job_runs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "anchor_dock.batch._dispatch",
        lambda job, output_dir, options: _pose_result(output_dir, job.mode),
    )
    jobs = [
        DockingJob("free", "CCO", id="missing-protein"),
        DockingJob.free("CCN", protein_pdb="protein.pdb", id="valid"),
    ]
    results = dock_batch(jobs, output_dir=tmp_path / "preflight", on_error="record")
    assert [result["success"] for result in results] == [False, True]
    summary = json.loads((tmp_path / "preflight" / "summary.json").read_text())
    assert summary == {
        "status": "complete",
        "errors": 1,
        "results": 2,
        "skipped": 0,
        "successful": 1,
        "total_items": 2,
    }


@pytest.mark.parametrize("failure", [RuntimeError("failed"), KeyboardInterrupt("interrupted")])
def test_batch_invalidates_root_completion_state_before_dispatch(
    monkeypatch,
    tmp_path: Path,
    failure: BaseException,
) -> None:
    root = tmp_path / f"root-state-{type(failure).__name__}"
    job = DockingJob.free("CCO", protein_pdb="protein.pdb", id="ethanol")
    monkeypatch.setattr(
        "anchor_dock.batch._dispatch",
        lambda job, output_dir, options: _pose_result(output_dir, job.mode),
    )
    dock_batch([job], output_dir=root)
    assert json.loads((root / "summary.json").read_text())["status"] == "complete"
    assert (root / "results.jsonl").read_text()

    def fail_dispatch(*args, **kwargs):
        del args, kwargs
        raise failure

    monkeypatch.setattr("anchor_dock.batch._dispatch", fail_dispatch)
    with pytest.raises(type(failure)):
        dock_batch([job], output_dir=root, on_error="raise")

    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "running"
    assert (root / "results.jsonl").read_text() == ""
    job_result = json.loads(next(path for path in root.glob("*/result.json")).read_text())
    assert job_result["batch_runtime_identity"]["device"] in {"cpu", "cuda"}
    assert set(job_result["batch_runtime_identity"]) == {"device", "numpy", "rdkit", "torch"}


def test_custom_scorer_weights_change_resume_signature(monkeypatch, tmp_path: Path) -> None:
    calls = 0

    class WeightedScorer(nn.Module):
        def __init__(self, weight: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(weight))

        def forward(self, ligand_coords, receptor_coords, ligand_features, receptor_features):
            del receptor_coords, ligand_features, receptor_features
            return ligand_coords.square().sum(dim=(1, 2)) * self.weight

    def fake_dispatch(job, output_dir, options):
        nonlocal calls
        del job, options
        calls += 1
        return _pose_result(output_dir)

    monkeypatch.setattr("anchor_dock.batch._dispatch", fake_dispatch)
    job = DockingJob.free("CCO", protein_pdb="protein.pdb", id="weighted")
    root = tmp_path / "fingerprint"
    first = dock_batch([job], output_dir=root, scorer=WeightedScorer(1.0))[0]
    second = dock_batch([job], output_dir=root, scorer=WeightedScorer(9.0), resume=True)[0]
    assert calls == 2
    assert first["job_signature"] != second["job_signature"]


def test_custom_scorer_labels_change_resume_signature(monkeypatch, tmp_path: Path) -> None:
    from anchor_dock.core.scoring import NeuralScorerAdapter

    calls = 0

    class Scorer(nn.Module):
        def forward(self, ligand_coords, receptor_coords, ligand_features, receptor_features):
            del receptor_coords, ligand_features, receptor_features
            return ligand_coords.square().sum(dim=(1, 2))

    def fake_dispatch(job, output_dir, options):
        nonlocal calls
        del job, options
        calls += 1
        return _pose_result(output_dir)

    monkeypatch.setattr("anchor_dock.batch._dispatch", fake_dispatch)
    model = Scorer()
    job = DockingJob.free("CCO", protein_pdb="protein.pdb", id="labels")
    root = tmp_path / "label-signature"
    first = dock_batch(
        [job],
        output_dir=root,
        scorer=NeuralScorerAdapter(model, name="calibrated-a", units="unit-a"),
    )[0]
    renamed = dock_batch(
        [job],
        output_dir=root,
        scorer=NeuralScorerAdapter(model, name="calibrated-b", units="unit-a"),
        resume=True,
    )[0]
    reunit = dock_batch(
        [job],
        output_dir=root,
        scorer=NeuralScorerAdapter(model, name="calibrated-b", units="unit-b"),
        resume=True,
    )[0]
    assert calls == 3
    assert len({first["job_signature"], renamed["job_signature"], reunit["job_signature"]}) == 3


def test_metadata_change_invalidates_resume_signature(monkeypatch, tmp_path: Path) -> None:
    calls = 0

    def fake_dispatch(job, output_dir, options):
        nonlocal calls
        del job, options
        calls += 1
        return _pose_result(output_dir)

    monkeypatch.setattr("anchor_dock.batch._dispatch", fake_dispatch)
    root = tmp_path / "metadata-signature"
    first_job = DockingJob.free(
        "CCO",
        protein_pdb="protein.pdb",
        id="ethanol",
        metadata={"campaign": 1},
    )
    second_job = DockingJob.free(
        "CCO",
        protein_pdb="protein.pdb",
        id="ethanol",
        metadata={"campaign": 2},
    )
    first = dock_batch([first_job], output_dir=root)[0]
    second = dock_batch([second_job], output_dir=root, resume=True)[0]
    assert calls == 2
    assert first["job_signature"] != second["job_signature"]


def test_file_content_change_invalidates_resume_with_same_size_and_mtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def fake_dispatch(job, output_dir, options):
        nonlocal calls
        del job, options
        calls += 1
        return _pose_result(output_dir)

    monkeypatch.setattr("anchor_dock.batch._dispatch", fake_dispatch)
    protein = tmp_path / "protein.pdb"
    protein.write_text("AAAA")
    timestamp = protein.stat().st_mtime_ns
    job = DockingJob.free("CCO", protein_pdb=protein, id="content")
    root = tmp_path / "content-signature"
    first = dock_batch([job], output_dir=root)[0]
    protein.write_text("BBBB")
    os.utime(protein, ns=(timestamp, timestamp))
    second = dock_batch([job], output_dir=root, resume=True)[0]
    assert calls == 2
    assert first["job_signature"] != second["job_signature"]


def test_manifest_paths_are_resolved_relative_to_manifest(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "protein.pdb").write_text("END\n")
    (inputs / "reference.sdf").write_text("$$$$\n")
    manifest = inputs / "jobs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "mode": "reference",
                "smiles": "CCO",
                "protein_pdb": "protein.pdb",
                "reference_ligand": "reference.sdf",
            }
        )
        + "\n"
    )
    item = next(iter_batch_items(manifest))
    assert isinstance(item, DockingJob)
    assert Path(item.protein_pdb) == inputs / "protein.pdb"
    assert Path(item.reference_ligand) == inputs / "reference.sdf"


def test_invalid_direct_docking_job_mode_is_preflight_failure(monkeypatch, tmp_path: Path) -> None:
    def unreachable_dispatch(job, output_dir, options):
        del job, output_dir, options
        pytest.fail("dispatch must not run for an invalid direct DockingJob mode")

    monkeypatch.setattr("anchor_dock.batch._dispatch", unreachable_dispatch)
    job = DockingJob("fre", "CCO", protein_pdb="protein.pdb", id="typo-mode")
    results = dock_batch([job], output_dir=tmp_path / "invalid-mode", on_error="record")
    assert results[0]["success"] is False
    assert "mode must be reference, covalent or free" in results[0]["error"]
    summary = json.loads((tmp_path / "invalid-mode" / "summary.json").read_text())
    assert summary["errors"] == 1
    assert summary["successful"] == 0


def test_invalid_direct_docking_job_mode_raises_under_on_error_raise(tmp_path: Path) -> None:
    job = DockingJob("fre", "CCO", protein_pdb="protein.pdb", id="typo-mode")
    with pytest.raises(ValueError, match="mode must be reference, covalent or free"):
        dock_batch([job], output_dir=tmp_path / "invalid-mode-raise", on_error="raise")


def test_dispatch_defensively_rejects_invalid_mode(tmp_path: Path) -> None:
    from anchor_dock.batch import _dispatch

    bad_job = DockingJob("fre", "CCO", protein_pdb="protein.pdb", id="typo-mode")
    with pytest.raises(ValueError, match="unsupported docking mode"):
        _dispatch(bad_job, tmp_path / "dispatch", {})


def test_covalent_dispatch_uses_canonical_pipeline_without_compat_warning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import anchor_dock._compat as compat
    import anchor_dock.covalent as covalent_package
    import anchor_dock.covalent.pipeline as covalent_pipeline
    from anchor_dock.batch import _dispatch

    captured: dict[str, object] = {}

    def fake_canonical(protein_pdb, ligand, reactive_residue, output_dir, **options):
        del protein_pdb, ligand, reactive_residue
        captured["options"] = options
        return _pose_result(Path(output_dir), "covalent")

    def unreachable_compat(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("canonical covalent batch dispatch must not cross the 0.2 compatibility wrapper")

    monkeypatch.setattr(covalent_pipeline, "dock_covalent", fake_canonical)
    monkeypatch.setattr(compat, "dock_covalent", unreachable_compat)
    monkeypatch.setattr(covalent_package, "dock_covalent", unreachable_compat)

    job = DockingJob.covalent("C=CC(=O)N", protein_pdb="p.pdb", reactive_residue="CYS1:A")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _dispatch(job, tmp_path / "covalent-dispatch", {})
    assert "optimize" not in captured["options"]


def test_reference_dispatch_uses_canonical_pipeline_without_compat_warning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import anchor_dock._compat as compat
    import anchor_dock.reference as reference_package
    import anchor_dock.reference.pipeline as reference_pipeline
    from anchor_dock.batch import _dispatch

    captured: dict[str, object] = {}

    def fake_canonical(protein_pdb, reference_ligand, ligand, output_dir, **options):
        del protein_pdb, reference_ligand, ligand
        captured["options"] = options
        return _pose_result(Path(output_dir), "reference")

    def unreachable_compat(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("canonical reference batch dispatch must not cross the 0.2 compatibility wrapper")

    monkeypatch.setattr(reference_pipeline, "dock_reference", fake_canonical)
    monkeypatch.setattr(compat, "dock_reference", unreachable_compat)
    monkeypatch.setattr(reference_package, "dock_reference", unreachable_compat)

    job = DockingJob.reference("CCO", protein_pdb="p.pdb", reference_ligand="r.sdf")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _dispatch(job, tmp_path / "reference-dispatch", {})
    assert "optimize" not in captured["options"]
