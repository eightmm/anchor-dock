"""Native execution/identity contracts; CLI stubs are not engine parity tests."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from anchor_dock.backends.native import (
    EMPIRICAL_FUNCTIONS,
    NativeBackendError,
    NativeRefinementBackend,
    _output_pose,
)
from anchor_dock.core.prepared import PreparedLigand

ROOT = Path(__file__).resolve().parents[1]
ORACLE = json.loads((ROOT / "tests/fixtures/native_scoring.json").read_text())


@pytest.fixture
def native_inputs():
    payload = ORACLE["cases"]["secondary_amine"]
    molecule = Chem.MolFromMolBlock(payload["molblock"], removeHs=False)
    ligand = PreparedLigand.from_pdbqt(
        molecule, payload["pdbqt"],
        serial_to_atom_index={int(key): value for key, value in payload["serial_to_atom_index"].items()},
    )
    # The oracle prepares glycine as a ligand; receptor input declares it rigid.
    receptor = "\n".join(line for line in ORACLE["receptor"]["pdbqt"].splitlines()
                         if line.startswith(("ATOM  ", "HETATM"))) + "\n"
    return receptor, payload["pdbqt"], ligand


def executable_stub(tmp_path, *, behavior="normal"):
    """An explicit contract stub, never reported as a scientific executable."""
    script = tmp_path / "native contract ; stub"
    code = f'''#!{sys.executable}
import os, pathlib, sys, time
args=sys.argv[1:]
mode={behavior!r}
if '--version' in args:
    print('contract-test-stub; NOT a real docking engine')
    raise SystemExit(0)
if mode=='failure':
    print('controlled native failure',file=sys.stderr)
    raise SystemExit(7)
if mode=='timeout': time.sleep(2)
if mode=='no_output': raise SystemExit(0)
print('CUDA_VISIBLE_DEVICES='+os.environ.get('CUDA_VISIBLE_DEVICES','unset'))
text=pathlib.Path(args[args.index('-l')+1]).read_text()
initial='--score_only' in args
shift=0 if initial else (100 if mode=='outside' else .1)
lines=[]
first=True
for line in text.splitlines():
    if line.startswith(('ATOM  ','HETATM')):
        x=float(line[30:38])+shift
        if mode=='bond_change' and not initial and first: x+=1
        line=line[:30]+f'{{x:8.3f}}'+line[38:]
        if not initial and first:
            if mode=='serial': line=line[:6]+'99999'+line[11:]
            if mode=='charge': line=line[:70]+'+0.999'+line[76:]
            if mode=='type': line=line[:77]+'OA'
        first=False
    lines.append(line)
scores=[]
if mode!='missing_score': scores.append('REMARK minimizedAffinity '+('nan' if mode=='nonfinite' else ('-1.0' if initial else '-2.0')))
if '--cnn_scoring' in args and args[args.index('--cnn_scoring')+1]!='none':
    scores+=['REMARK CNNscore '+('0.6' if initial else '0.8'),'REMARK CNNaffinity 7.2']
if mode=='ambiguous_score': scores.append('REMARK minimizedAffinity -3')
out='MODEL 0\\n'+'\\n'.join(scores+lines)+'\\nENDMDL\\n'
if mode=='multiple_models': out=out+out
if mode=='malformed': out='not a PDBQT pose'
pathlib.Path(args[args.index('--out')+1]).write_text(out)
'''
    script.write_text(code)
    script.chmod(0o755)
    return script


@pytest.mark.parametrize("sf", EMPIRICAL_FUNCTIONS)
def test_smina_contract_selects_exact_sf_and_preserves_atom_mapping(tmp_path, native_inputs, sf):
    receptor, text, ligand = native_inputs
    binary = executable_stub(tmp_path)
    result = NativeRefinementBackend("smina", sf, executable=binary).refine(receptor, text, ligand)
    torch.testing.assert_close(result.coordinates, ligand.coordinates(dtype=torch.float64) + torch.tensor([.1, 0, 0]))
    assert result.initial_scores["affinity"].value == -1
    assert result.scores["affinity"].value == -2
    assert result.scores["affinity"].direction == "lower"
    assert result.mol.GetNumAtoms() == ligand.mol.GetNumAtoms()
    assert "NOT a real docking engine" in result.metadata["engine_version"]
    argv = result.metadata["argv"]["refinement"]
    assert argv[0] == str(binary)
    assert argv[argv.index("--scoring") + 1] == sf
    assert "--minimize" in argv and "--score_only" not in argv
    assert "--score_only" in result.metadata["argv"]["initial"]
    assert result.metadata["binary_sha256"].startswith("sha256:")
    assert result.metadata["prepared_ligand_fingerprint"] == ligand.fingerprint
    assert json.loads(json.dumps(result.as_dict()))["scoring_function"] == sf
    assert "AnchorDock interaction" in result.metadata["native_constraints"]


@pytest.mark.parametrize("mode", ["none", "rescore", "refinement"])
def test_gnina_objective_is_not_confused_with_reported_cnn_affinity(tmp_path, native_inputs, mode):
    receptor, text, ligand = native_inputs
    backend = NativeRefinementBackend("gnina", "vinardo", executable=executable_stub(tmp_path), cnn_scoring=mode)
    result = backend.refine(receptor, text, ligand)
    argv = result.metadata["argv"]["refinement"]
    assert argv[argv.index("--cnn_scoring") + 1] == mode
    assert "--no_gpu" in argv
    if mode != "none":
        assert result.scores["CNNscore"].value == .8
        assert result.scores["CNNscore"].units == "probability"
        assert result.scores["CNNaffinity"].direction == "higher"
        assert result.scores["affinity"].direction == "lower"
    else:
        assert "CNNscore" not in result.scores
    if mode == "refinement":
        assert "CNN pose loss" in result.optimized_objective
        assert "diagnostics" in result.optimized_objective
    else:
        assert "CNN pose loss" not in result.optimized_objective


def test_external_model_contents_are_snapshotted_and_fingerprinted(tmp_path, native_inputs, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,5,3")
    receptor, text, ligand = native_inputs
    paths = [tmp_path / folder / "same-name.pt" for folder in ("first", "second")]
    for index, path in enumerate(paths):
        path.parent.mkdir()
        path.write_bytes(bytes([index]))
    result = NativeRefinementBackend("gnina", executable=executable_stub(tmp_path), cnn_scoring="refinement",
                                     cnn_model_files=tuple(paths), device="cuda:2").refine(receptor, text, ligand)
    assert len(set(result.metadata["cnn_model_sha256"].values())) == 2
    args = result.metadata["argv"]["refinement"]
    assert args[args.index("--device") + 1] == "0"
    assert "--no_gpu" not in args and "--cnn_model" in args
    assert result.metadata["environment_overrides"] == {"CUDA_VISIBLE_DEVICES": "3"}
    assert "CUDA_VISIBLE_DEVICES=3" in result.metadata["log_tail"]


def test_scorer_fingerprint_is_independent_of_temporary_paths_and_optimization_steps(tmp_path, native_inputs):
    binary = executable_stub(tmp_path)
    backend = NativeRefinementBackend("smina", executable=binary)
    first = backend.refine(*native_inputs, seed=42, max_steps=10)
    second = backend.refine(*native_inputs, seed=43, max_steps=20)
    changed = NativeRefinementBackend("smina", "dkoes_fast", executable=binary).refine(*native_inputs)
    assert first.metadata["scorer_fingerprint"] == second.metadata["scorer_fingerprint"]
    assert first.metadata["scorer_fingerprint"] != changed.metadata["scorer_fingerprint"]
    assert first.metadata["seed"] != second.metadata["seed"]


def test_cli_padding_cap_and_hydrogen_mobility_are_explicit_for_both_calls(tmp_path, native_inputs):
    backend = NativeRefinementBackend("gnina", executable=executable_stub(tmp_path), force_cap=25.)
    result = backend.refine(*native_inputs, padding=6.2)
    for args in (result.metadata["argv"]["initial"], result.metadata["argv"]["refinement"]):
        assert args[args.index("--autobox_add") + 1] == "6.2"
        assert args[args.index("--force_cap") + 1] == "25.0"
        assert "--flex_hydrogens" in args
        assert "--center_x" not in args
    original = native_inputs[2].coordinates(dtype=torch.float64)
    span = original.amax(0) - original.amin(0) + 2 * 6.2
    expected_span = torch.ceil(span / .375) * .375
    bounds = result.metadata["bounds_angstrom"]
    torch.testing.assert_close(torch.tensor(bounds["upper"]) - torch.tensor(bounds["lower"]), expected_span.float())
    changed = NativeRefinementBackend("gnina", executable=backend.executable, force_cap=1000.).refine(*native_inputs)
    assert changed.metadata["scorer_fingerprint"] != result.metadata["scorer_fingerprint"]


def test_cli_zero_steps_is_score_only_and_preserves_native_input_pose(tmp_path, native_inputs):
    backend = NativeRefinementBackend("smina", executable=executable_stub(tmp_path))
    result = backend.refine(*native_inputs, max_steps=0, seed=0)
    torch.testing.assert_close(result.coordinates, native_inputs[2].coordinates(dtype=torch.float64), atol=0., rtol=0.)
    assert result.initial_scores == result.scores
    assert result.metadata["argv"]["refinement"] is None
    assert result.metadata["requested_seed"] == result.metadata["effective_seed"] == 0
    assert not result.metadata["optimization_performed"]


def test_requested_gnina_gpu_must_be_visible(tmp_path, native_inputs, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    backend = NativeRefinementBackend("gnina", executable=executable_stub(tmp_path), device="cuda:0")
    with pytest.raises(ValueError, match="not visible"):
        backend.refine(*native_inputs)


@pytest.mark.parametrize("mode,match", [
    ("failure", "exited 7"), ("timeout", "timed out"), ("no_output", "did not produce"),
    ("missing_score", "missing minimizedAffinity"), ("nonfinite", "nonfinite"),
    ("ambiguous_score", "ambiguous native score"), ("multiple_models", "exactly one"),
    ("malformed", "invalid native output"), ("serial", "serial identity"),
    ("charge", "types or charges"), ("type", "types or charges"),
    ("bond_change", "bond length"), ("outside", "outside the explicit local box"),
])
def test_native_contract_failures_never_return_a_successful_pose(tmp_path, native_inputs, mode, match):
    receptor, text, ligand = native_inputs
    backend = NativeRefinementBackend("smina", executable=executable_stub(tmp_path, behavior=mode))
    with pytest.raises((NativeBackendError, ValueError), match=match):
        backend.refine(receptor, text, ligand, timeout=.1 if mode == "timeout" else 10)


@pytest.mark.parametrize("options", [
    {"engine": "invalid"}, {"engine": "vina", "scoring_function": "dkoes_fast"},
    {"engine": "smina", "scoring_function": "default"}, {"engine": "smina", "cnn_scoring": "refinement"},
    {"engine": "gnina", "cnn_scoring": "all"}, {"engine": "gnina", "cnn_models": "dense"},
    {"engine": "gnina", "cnn_scoring": "refinement", "cnn_models": ("--bad",)},
    {"engine": "vina", "maps_prefix": "maps"}, {"engine": "vina", "scoring_function": "ad4"},
    {"engine": "smina", "device": "cuda:0"}, {"engine": "gnina", "device": "cuda"},
])
def test_invalid_native_configuration_fails_before_running(options):
    with pytest.raises(ValueError):
        NativeRefinementBackend(**options)


@pytest.mark.parametrize("key,value", [("seed", -1), ("seed", True), ("max_steps", -1), ("cpu", 0),
                                      ("timeout", math.nan), ("padding", math.inf)])
def test_invalid_runtime_options_fail_before_dependency_lookup(native_inputs, key, value):
    with pytest.raises(ValueError, match=key):
        NativeRefinementBackend("smina").refine(*native_inputs, **{key: value})


def test_missing_native_dependency_is_explicit(tmp_path, native_inputs):
    backend = NativeRefinementBackend("smina", executable=tmp_path / "does-not-exist")
    assert not backend.availability("smina", backend.executable)["available"]
    with pytest.raises(ImportError, match="install smina"):
        backend.refine(*native_inputs)


def test_prepared_serial_mapping_and_parameter_changes_fail(tmp_path, native_inputs):
    receptor, text, ligand = native_inputs
    backend = NativeRefinementBackend("smina", executable=executable_stub(tmp_path))
    arrays = PreparedLigand.from_arrays(ligand.mol, ligand.ad4_types, ligand.charges,
                                        rotatable_bonds=ligand.rotatable_bonds, torsdof=ligand.torsdof)
    with pytest.raises(ValueError, match="serial identity"):
        backend.refine(receptor, text, arrays)
    first = next(line for line in text.splitlines() if line.startswith("ATOM"))
    changed = text.replace(first, first[:70] + "+0.999" + first[76:])
    with pytest.raises(ValueError, match="parameters disagree"):
        backend.refine(receptor, changed, ligand)


@pytest.mark.skipif(importlib.util.find_spec("vina") is None, reason="optional native Vina binding unavailable")
@pytest.mark.parametrize("sf", ["vina", "vinardo"])
def test_actual_native_vina_cpu_refinement(native_inputs, sf):
    receptor, text, ligand = native_inputs
    result = NativeRefinementBackend("vina", sf).refine(receptor, text, ligand, max_steps=20, padding=6., timeout=60)
    assert torch.isfinite(result.coordinates).all()
    assert result.coordinates.dtype == torch.float64
    assert result.scores["search_energy"].value <= result.initial_scores["search_energy"].value + .01
    assert result.metadata["runtime_sha256"]
    assert result.metadata["score_precision_decimals"] == 3
    assert result.metadata["engine_version"]
    _output_pose(result.pdbqt, ligand)


@pytest.mark.skipif(importlib.util.find_spec("vina") is None, reason="optional native Vina binding unavailable")
def test_actual_native_vina_ad4_uses_explicit_map_snapshots(tmp_path, native_inputs):
    receptor, text, ligand = native_inputs
    prefix = tmp_path / "maps"
    header = "GRID_PARAMETER_FILE synthetic.gpf\nGRID_DATA_FILE synthetic.fld\nMACROMOLECULE synthetic.pdbqt\nSPACING 2\nNELEMENTS 16 16 16\nCENTER 0 0 0\n"
    content = header + "0\n" * 17**3
    for kind in sorted(set(ligand.ad4_types)) + ["e", "d"]:
        Path(f"{prefix}.{kind}.map").write_text(content)
    result = NativeRefinementBackend("vina", "ad4", maps_prefix=prefix).refine(receptor, text, ligand, max_steps=5, timeout=60)
    assert result.metadata["map_sha256"]
    assert result.metadata["bounds_angstrom"] == {"lower": [-16., -16., -16.], "upper": [16., 16., 16.]}
    assert result.scores["intermolecular"].value == 0.
    assert abs(result.scores["affinity"].value - .2983 * ligand.torsdof) <= .001
    assert result.metadata["worker_options"]["sf"] == "ad4"
    # Receptor map contents are the authoritative objective, so they belong
    # in its identity even with exactly the same PDBQT coordinate inputs.
    for kind in sorted(set(ligand.ad4_types)) + ["e", "d"]:
        Path(f"{prefix}.{kind}.map").write_text(header + ".1\n" * 17**3)
    changed = NativeRefinementBackend("vina", "ad4", maps_prefix=prefix).refine(receptor, text, ligand, max_steps=0, timeout=60)
    assert result.metadata["scorer_fingerprint"] != changed.metadata["scorer_fingerprint"]
    assert result.scores["affinity"].value != changed.scores["affinity"].value


@pytest.mark.skipif(importlib.util.find_spec("vina") is None, reason="optional native Vina binding unavailable")
def test_actual_native_vina_zero_steps_and_zero_seed_are_explicit(native_inputs):
    result = NativeRefinementBackend("vina").refine(*native_inputs, max_steps=0, seed=0, timeout=60)
    torch.testing.assert_close(result.coordinates, native_inputs[2].coordinates(dtype=torch.float64), atol=0., rtol=0.)
    assert result.initial_scores == result.scores
    assert result.metadata["requested_seed"] == 0
    assert result.metadata["effective_seed"] == result.metadata["worker_options"]["seed"] == 1
    assert not result.metadata["optimization_performed"]
