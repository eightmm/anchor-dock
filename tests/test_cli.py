import json

import pytest

from anchor_dock.cli import build_parser, main

INTERACTION_ARGS = [
    "interaction",
    "-p",
    "p.pdb",
    "-q",
    "CCO",
    "--receptor-residue",
    "CYS145:A",
    "--receptor-atom",
    "SG",
    "--ligand-smarts",
    "[#8:1]",
    "--target-distance",
    "3.0",
    "--distance-tolerance",
    "0.5",
]
MULTI_INTERACTIONS = [
    {
        "receptor_residue": "CYS145:A",
        "receptor_atom": "SG",
        "ligand_smarts": "[#8:1]",
        "target_distance": 3.0,
        "distance_tolerance": 0.5,
    },
    {
        "receptor_residue": "HIS41:A",
        "receptor_atom": "NE2",
        "ligand_smarts": "[#7:1]",
        "target_distance": 3.2,
        "distance_tolerance": 0.4,
        "restraint_weight": 4.0,
    },
]
MULTI_INTERACTION_ARGS = [
    "interaction",
    "-p",
    "p.pdb",
    "-q",
    "CCO",
    "--interactions-json",
    json.dumps(MULTI_INTERACTIONS),
]


def test_cli_exposes_all_native_modes() -> None:
    parser = build_parser()
    assert parser.parse_args(["reference", "-p", "p.pdb", "-r", "r.sdf", "-q", "CCO"]).command == "reference"
    assert parser.parse_args(["covalent", "-p", "p.pdb", "-q", "C=CC=O"]).command == "covalent"
    assert parser.parse_args(INTERACTION_ARGS).command == "interaction"
    assert parser.parse_args(["batch", "jobs.jsonl"]).command == "batch"


def test_cli_rejects_removed_free_mode() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["free", "-p", "p.pdb", "-q", "CCO"])


def test_batch_cli_returns_nonzero_when_any_job_failed(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "anchor_dock.cli.dock_batch",
        lambda *args, **kwargs: [{"success": True}, {"success": False, "error": "failed"}],
    )
    assert main(["batch", "jobs.jsonl"]) == 1
    assert json.loads(capsys.readouterr().out)[1]["success"] is False


def test_batch_cli_skip_still_returns_nonzero_for_parse_errors(tmp_path, capsys) -> None:
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text("{not-json}\n")
    output = tmp_path / "batch"
    assert main(["batch", str(manifest), "--on-error", "skip", "-o", str(output)]) == 1
    assert json.loads(capsys.readouterr().out) == []
    assert json.loads((output / "summary.json").read_text())["errors"] == 1


def test_removed_vina_lp_fails_at_cli_preflight(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main([*INTERACTION_ARGS, "--scorer", "vina_lp"])
    assert error.value.code == 2
    assert "unvalidated 0.2 'vina_lp'" in capsys.readouterr().err


def test_covalent_cli_omitted_optimize_is_explicitly_false(monkeypatch, capsys) -> None:
    captured = None

    def fake_covalent(*args, **kwargs):
        nonlocal captured
        captured = kwargs
        return {"mode": "covalent"}

    monkeypatch.setattr("anchor_dock.cli.dock_covalent", fake_covalent)
    with pytest.warns(FutureWarning, match="preserves the 0.2 default False"):
        assert main(["covalent", "-p", "p.pdb", "-q", "C=CC(=O)N"]) == 0
    assert captured is not None and captured["optimize"] is False
    assert json.loads(capsys.readouterr().out)["mode"] == "covalent"


def _fake_dock_interaction(monkeypatch):
    captured = {}

    def fake_interaction(*args, **kwargs):
        captured.update(kwargs)
        return {"mode": "interaction"}

    monkeypatch.setattr("anchor_dock.cli.dock_interaction", fake_interaction)
    return captured


def test_interaction_cli_defaults_and_selectors(monkeypatch, capsys) -> None:
    captured = _fake_dock_interaction(monkeypatch)
    assert main(INTERACTION_ARGS) == 0
    assert captured["optimize"] is True
    assert captured["receptor_residue"] == "CYS145:A"
    assert captured["receptor_atom"] == "SG"
    assert captured["ligand_smarts"] == "[#8:1]"
    assert captured["target_distance"] == 3.0
    assert captured["distance_tolerance"] == 0.5
    assert captured["num_confs"] == 32
    assert captured["num_candidates"] == 128
    assert captured["preselect_k"] == 16
    assert captured["max_matches"] == 16
    assert captured["max_joint_matches"] == 64
    assert captured["opt_steps"] == 50
    assert captured["release_steps"] == 25
    assert captured["opt_batch_size"] == 32
    assert captured["scorer"] == "softdock"
    assert json.loads(capsys.readouterr().out)["mode"] == "interaction"


def test_interaction_cli_explicit_optimize_true(monkeypatch, capsys) -> None:
    captured = _fake_dock_interaction(monkeypatch)
    assert main([*INTERACTION_ARGS, "--optimize"]) == 0
    assert captured["optimize"] is True
    capsys.readouterr()


def test_interaction_cli_no_optimize_forwards_false(monkeypatch, capsys) -> None:
    captured = _fake_dock_interaction(monkeypatch)
    assert main([*INTERACTION_ARGS, "--no-optimize"]) == 0
    assert captured["optimize"] is False
    capsys.readouterr()


def test_interaction_cli_optimize_flags_are_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*INTERACTION_ARGS, "--optimize", "--no-optimize"])


def test_interaction_cli_forwards_ordered_multi_interactions(monkeypatch, capsys) -> None:
    captured = _fake_dock_interaction(monkeypatch)
    assert main([*MULTI_INTERACTION_ARGS, "--max-joint-matches", "23"]) == 0
    assert captured["interactions"] == MULTI_INTERACTIONS
    assert captured["max_joint_matches"] == 23
    assert "receptor_residue" not in captured
    assert json.loads(capsys.readouterr().out)["mode"] == "interaction"


def test_interaction_cli_rejects_multi_and_single_selector_mix(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main([*MULTI_INTERACTION_ARGS, "--receptor-residue", "ASP189:A"])
    assert error.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_interaction_cli_rejects_incomplete_single_selector_set(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["interaction", "-p", "p.pdb", "-q", "CCO", "--receptor-residue", "CYS145:A"])
    assert error.value.code == 2
    assert "missing:" in capsys.readouterr().err


@pytest.mark.parametrize("payload", ["[]", "{}", '[{"receptor_residue": "CYS145:A"}, 7]', "@spec.json"])
def test_interaction_cli_rejects_invalid_inline_interactions_json(payload: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["interaction", "-p", "p.pdb", "-q", "CCO", "--interactions-json", payload]
        )


def test_batch_cli_forwards_interaction_fields(monkeypatch, capsys) -> None:
    captured = {}

    def fake_batch(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("anchor_dock.cli.dock_batch", fake_batch)
    assert main([
        "batch", "jobs.jsonl", "--mode", "interaction", "--protein", "p.pdb",
        "--receptor-residue", "CYS145:A", "--receptor-atom", "SG",
        "--ligand-smarts", "[#8:1]", "--target-distance", "3.0",
        "--distance-tolerance", "0.5",
    ]) == 0
    assert {name: captured[name] for name in (
        "mode", "receptor_residue", "receptor_atom", "ligand_smarts",
        "target_distance", "distance_tolerance",
    )} == {
        "mode": "interaction",
        "receptor_residue": "CYS145:A",
        "receptor_atom": "SG",
        "ligand_smarts": "[#8:1]",
        "target_distance": 3.0,
        "distance_tolerance": 0.5,
    }
    capsys.readouterr()


def test_batch_cli_forwards_multi_interactions(monkeypatch, capsys) -> None:
    captured = {}

    def fake_batch(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("anchor_dock.cli.dock_batch", fake_batch)
    assert main(
        [
            "batch",
            "jobs.jsonl",
            "--mode",
            "interaction",
            "--protein",
            "p.pdb",
            "--interactions-json",
            json.dumps(MULTI_INTERACTIONS),
            "--max-joint-matches",
            "17",
        ]
    ) == 0
    assert captured["interactions"] == MULTI_INTERACTIONS
    assert captured["max_joint_matches"] == 17
    assert captured["receptor_residue"] is None
    capsys.readouterr()


def test_batch_cli_multi_interactions_requires_interaction_mode(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "batch",
                "jobs.jsonl",
                "--interactions-json",
                json.dumps(MULTI_INTERACTIONS),
            ]
        )
    assert error.value.code == 2
    assert "requires --mode interaction" in capsys.readouterr().err
