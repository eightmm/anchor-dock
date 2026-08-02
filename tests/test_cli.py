import json

import pytest

from anchor_dock.cli import build_parser, main


def test_cli_exposes_all_native_modes() -> None:
    parser = build_parser()
    assert parser.parse_args(["reference", "-p", "p.pdb", "-r", "r.sdf", "-q", "CCO"]).command == "reference"
    assert parser.parse_args(["covalent", "-p", "p.pdb", "-q", "C=CC=O"]).command == "covalent"
    assert parser.parse_args(["free", "-p", "p.pdb", "-q", "CCO"]).command == "free"
    assert parser.parse_args(["batch", "jobs.jsonl"]).command == "batch"


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
        main(["free", "-p", "p.pdb", "-q", "CCO", "--scorer", "vina_lp"])
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


def _fake_dock_free(monkeypatch):
    captured = {}

    def fake_free(*args, **kwargs):
        captured.update(kwargs)
        return {"mode": "free"}

    monkeypatch.setattr("anchor_dock.cli.dock_free", fake_free)
    return captured


def test_free_cli_defaults_to_optimize_true(monkeypatch, capsys) -> None:
    captured = _fake_dock_free(monkeypatch)
    assert main(["free", "-p", "p.pdb", "-q", "CCO"]) == 0
    assert captured["optimize"] is True
    assert json.loads(capsys.readouterr().out)["mode"] == "free"


def test_free_cli_explicit_optimize_true(monkeypatch, capsys) -> None:
    captured = _fake_dock_free(monkeypatch)
    assert main(["free", "-p", "p.pdb", "-q", "CCO", "--optimize"]) == 0
    assert captured["optimize"] is True
    capsys.readouterr()


def test_free_cli_no_optimize_forwards_false(monkeypatch, capsys) -> None:
    captured = _fake_dock_free(monkeypatch)
    assert main(["free", "-p", "p.pdb", "-q", "CCO", "--no-optimize"]) == 0
    assert captured["optimize"] is False
    capsys.readouterr()


def test_free_cli_optimize_flags_are_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["free", "-p", "p.pdb", "-q", "CCO", "--optimize", "--no-optimize"])
