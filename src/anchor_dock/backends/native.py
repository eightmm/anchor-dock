"""Native local refinement with explicit engine and scoring-function contracts.

These adapters delegate optimization to an installed engine. They do not expose
its gradients to Torch or reproduce AnchorDock's guided/release constraints.
PDBQT serial identity is retained rather than guessing an SDF atom mapping.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch
from rdkit import Chem

from ..core.prepared import PreparedLigand, parse_pdbqt

NATIVE_REFINEMENT_CONTRACT = "native-local-refinement-pdbqt-serial-v1"
EMPIRICAL_FUNCTIONS = ("vina", "vinardo", "dkoes_fast", "dkoes_scoring", "dkoes_scoring_old", "ad4_scoring")
_SCORE_REMARKS = {"minimizedAffinity", "minimizedRMSD", "CNNscore", "CNNaffinity", "CNNaffinity_variance"}
_GNINA_SOURCE = "https://github.com/gnina/gnina/tree/01cae6aa6e014e0bf11e511f1fe7a61b17975855"


class NativeBackendError(RuntimeError):
    """The selected native dependency failed or violated its output contract."""


@dataclass(frozen=True)
class NativeScore:
    value: float
    units: str
    direction: str
    role: str = "reported"

    def as_dict(self) -> dict[str, object]:
        return {"value": self.value, "units": self.units, "direction": self.direction, "role": self.role}


@dataclass(frozen=True)
class NativeRefinementResult:
    engine: str
    scoring_function: str
    coordinates: torch.Tensor
    scores: dict[str, NativeScore]
    initial_scores: dict[str, NativeScore]
    optimized_objective: str
    metadata: dict[str, object]
    pdbqt: str
    _mol: Chem.Mol

    @property
    def mol(self) -> Chem.Mol:
        """Copy of the supplied chemical graph with the final coordinates."""
        return Chem.Mol(self._mol)

    def as_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine, "scoring_function": self.scoring_function,
            "coordinates": self.coordinates.detach().cpu().tolist(),
            "scores": {name: value.as_dict() for name, value in self.scores.items()},
            "initial_scores": {name: value.as_dict() for name, value in self.initial_scores.items()},
            "optimized_objective": self.optimized_objective, "metadata": self.metadata,
        }


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _integer(name, value, minimum, maximum=2**31 - 1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return int(value)


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _single_model(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("PDBQT must be nonempty text, not a filename")
    starts = ends = 0
    inside = False
    kept = []
    for line in text.splitlines():
        word = line.split(maxsplit=1)[0] if line.strip() else ""
        if word == "MODEL":
            starts += 1
            if inside or starts > 1:
                raise ValueError("native local refinement requires exactly one PDBQT model")
            inside = True
        elif word == "ENDMDL":
            if not inside:
                raise ValueError("unmatched PDBQT ENDMDL")
            ends += 1
            inside = False
        else:
            if starts and not inside and word in {"ATOM", "HETATM", "ROOT", "BRANCH"}:
                raise ValueError("PDBQT atoms/tree outside its model")
            kept.append(line)
    if inside or starts != ends:
        raise ValueError("unclosed PDBQT model")
    return "\n".join(kept) + "\n"


def _clean_input(text: str) -> str:
    lines = []
    for line in _single_model(text).splitlines():
        words = line.split()
        if len(words) >= 2 and words[0] == "REMARK" and (words[1] in _SCORE_REMARKS or words[1] == "VINA"):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def _validate_input(receptor: str, text: str, ligand: PreparedLigand) -> tuple[str, str]:
    if not isinstance(ligand, PreparedLigand):
        raise TypeError("ligand must be PreparedLigand with an explicit PDBQT serial mapping")
    if len(ligand.pdbqt_serials) != ligand.mol.GetNumAtoms() or len(set(ligand.pdbqt_serials)) != len(ligand.pdbqt_serials):
        raise ValueError("native refinement requires PreparedLigand.from_pdbqt serial identity")
    receptor, text = _clean_input(receptor), _clean_input(text)
    parse_pdbqt(receptor, rigid=True)
    # Checks coordinate/order/element/tree correspondence against the supplied
    # graph. Nonpolar H already omitted by preparation are not guessed back.
    supplied = PreparedLigand.from_pdbqt(
        ligand.mol, text, serial_to_atom_index={serial: index for index, serial in enumerate(ligand.pdbqt_serials)},
    )
    if (supplied.ad4_types != ligand.ad4_types or supplied.charges != ligand.charges
            or supplied.torsdof != ligand.torsdof
            or {frozenset(pair) for pair in supplied.rotatable_bonds} != {frozenset(pair) for pair in ligand.rotatable_bonds}):
        raise ValueError("supplied PDBQT parameters disagree with prepared ligand")
    return receptor, text


def _output_pose(text: str, ligand: PreparedLigand) -> tuple[torch.Tensor, Chem.Mol]:
    normalized = _single_model(text)
    serials = [int(line[6:11]) for line in normalized.splitlines() if line[:6].strip() in {"ATOM", "HETATM"}]
    if serials and (len(serials) != len(set(serials)) or set(serials) != set(ligand.pdbqt_serials)):
        raise NativeBackendError("native output changed ligand atom serial identity")
    parsed = parse_pdbqt(normalized)
    by_serial = {atom.serial: atom for atom in parsed.atoms}
    if set(by_serial) != set(ligand.pdbqt_serials):
        raise NativeBackendError("native output changed ligand atom serial identity")
    for index, serial in enumerate(ligand.pdbqt_serials):
        atom = by_serial[serial]
        if atom.atom_type != ligand.ad4_types[index] or abs(atom.charge - ligand.charges[index]) > 1e-6:
            raise NativeBackendError("native output changed prepared atom types or charges")
    indices = {serial: index for index, serial in enumerate(ligand.pdbqt_serials)}
    branches = {frozenset((indices[a], indices[b])) for a, b in parsed.branches}
    if branches != {frozenset(pair) for pair in ligand.rotatable_bonds} or parsed.torsdof != ligand.torsdof:
        raise NativeBackendError("native output changed the prepared torsion tree")
    xyz = torch.tensor([by_serial[serial].coordinates for serial in ligand.pdbqt_serials], dtype=torch.float64)
    original = ligand.coordinates(dtype=torch.float64)
    mol = ligand.mol
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        drift = abs(torch.linalg.vector_norm(xyz[left] - xyz[right]) - torch.linalg.vector_norm(original[left] - original[right]))
        if drift > .01:
            raise NativeBackendError("native local torsion refinement changed a bond length by more than 0.01 angstrom")
    for index, point in enumerate(xyz.tolist()):
        mol.GetConformer().SetAtomPosition(index, point)
    return xyz, mol


def _cli_scores(text: str, *, cnn: bool) -> dict[str, NativeScore]:
    values = {}
    for line in text.splitlines():
        words = line.split()
        if len(words) >= 2 and words[0] == "REMARK" and words[1] in _SCORE_REMARKS:
            if len(words) != 3 or words[1] in values:
                raise NativeBackendError("ambiguous native score remark")
            try:
                number = float(words[2])
            except ValueError as exc:
                raise NativeBackendError("nonnumeric native score") from exc
            if not math.isfinite(number):
                raise NativeBackendError("nonfinite native score")
            values[words[1]] = number
    if "minimizedAffinity" not in values:
        raise NativeBackendError("native output is missing minimizedAffinity")
    scores = {"affinity": NativeScore(values["minimizedAffinity"], "kcal/mol-like", "lower")}
    if cnn:
        if "CNNscore" not in values or not 0 <= values["CNNscore"] <= 1:
            raise NativeBackendError("native CNN output requires CNNscore in [0,1]")
        scores["CNNscore"] = NativeScore(values["CNNscore"], "probability", "higher", "CNN pose diagnostic")
        if "CNNaffinity" in values:
            scores["CNNaffinity"] = NativeScore(values["CNNaffinity"], "pK-like", "higher", "CNN affinity diagnostic")
    return scores


def _run(argv: list[str], *, cwd: Path, deadline: float, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise NativeBackendError("native refinement timeout expired")
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=remaining, check=False,
                                env=None if env is None else {**os.environ, **env})
    except subprocess.TimeoutExpired as exc:
        raise NativeBackendError("native refinement timed out") from exc
    except OSError as exc:
        raise NativeBackendError(f"cannot execute native backend: {exc}") from exc
    if result.returncode:
        raise NativeBackendError(f"native backend exited {result.returncode}: {result.stderr[-4000:]}")
    return result


_VINA_WORKER = """
import json, sys
from pathlib import Path
from vina import Vina
options = json.loads(Path(sys.argv[1]).read_text())
v = Vina(sf_name=options['sf'], cpu=options['cpu'], seed=options['seed'], verbosity=0)
if options['sf'] != 'ad4':
    v.set_receptor(options['receptor'])
v.set_ligand_from_file(options['ligand'])
if options['sf'] == 'ad4':
    v.load_maps(options['maps'])
else:
    v.compute_vina_maps(options['center'], options['size'], force_even_voxels=True)
initial = v.score().tolist()
final = v.optimize(max_steps=options['max_steps']).tolist() if options['max_steps'] else initial
v.write_pose(options['output'], overwrite=True)
Path(options['result']).write_text(json.dumps({'initial':initial, 'final':final}))
"""


def _vina_scores(values) -> dict[str, NativeScore]:
    if (not isinstance(values, list) or len(values) != 8
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)):
        raise NativeBackendError("native Vina returned malformed energy components")
    inter, intra = values[1] + values[2], sum(values[3:6])
    return {
        "affinity": NativeScore(values[0], "kcal/mol-like", "lower"),
        "intermolecular": NativeScore(inter, "kcal/mol-like", "lower", "native component"),
        "intramolecular": NativeScore(intra, "kcal/mol-like", "lower", "native component"),
        "search_energy": NativeScore(inter + intra, "kcal/mol-like", "lower", "native search component sum"),
        "torsion": NativeScore(values[6], "kcal/mol-like", "lower", "reporting term"),
        "unbound_reference": NativeScore(values[7], "kcal/mol-like", "lower", "reporting reference"),
    }


class NativeRefinementBackend:
    """Run local refinement using actual native implementations.

    ``vina`` uses the optional Python binding in a killable subprocess. Smina
    and GNINA use an installed executable. Native ``vinardo`` versions differ
    from AnchorDock's modern Vina-lineage convention; engine identity matters.
    GNINA ``rescore`` does not optimize its CNN. ``refinement`` optimizes the
    native CNN pose loss plus native intra/boundary terms; CNNaffinity remains
    a diagnostic. The final empirical affinity is not that CNN objective.

    Inputs are prepared PDBQT text and an explicitly mapped PreparedLigand.
    No protonation/charge conversion, global search, AnchorDock interaction
    guidance or covalent/flexible receptor options are silently added.

    Zero steps means score-only with the starting pose preserved. Seed zero
    is explicitly translated to one only for Vina, whose zero otherwise asks
    for a random seed; requested/effective values are reported. CLI force cap
    is supplied identically to score-only and minimization to avoid native
    mode-dependent defaults. Native bond perception can differ from the graph.
    """

    def __init__(
        self, engine: str, scoring_function: str = "vina", *, executable: str | os.PathLike[str] | None = None,
        cnn_scoring: str = "none", cnn_models: tuple[str, ...] = (),
        cnn_model_files: tuple[str | os.PathLike[str], ...] = (), device: str = "cpu",
        maps_prefix: str | os.PathLike[str] | None = None, force_cap: float = 1000.,
    ) -> None:
        if engine not in {"vina", "smina", "gnina"}:
            raise ValueError("native engine must be vina, smina or gnina")
        allowed = ("vina", "vinardo", "ad4") if engine == "vina" else EMPIRICAL_FUNCTIONS
        if scoring_function not in allowed:
            raise ValueError(f"{engine} scoring_function must be one of {allowed}")
        if cnn_scoring not in {"none", "rescore", "refinement"}:
            raise ValueError("cnn_scoring must be none, rescore or refinement")
        if engine != "gnina" and (cnn_scoring != "none" or cnn_models or cnn_model_files):
            raise ValueError("CNN settings require the gnina engine")
        if (isinstance(cnn_models, (str, bytes)) or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                                                      for name in cnn_models)):
            raise ValueError("cnn_models must contain explicit built-in model names")
        if isinstance(cnn_model_files, (str, bytes)) or (cnn_models and cnn_model_files):
            raise ValueError("choose either built-in CNN models or external model files")
        if cnn_scoring == "none" and (cnn_models or cnn_model_files):
            raise ValueError("CNN models require CNN scoring")
        if device != "cpu" and (engine != "gnina" or not isinstance(device, str) or not re.fullmatch(r"cuda:\d+", device)):
            raise ValueError("native device must be cpu, or cuda:N for GNINA")
        if engine == "vina" and executable is not None:
            raise ValueError("native vina uses its optional Python binding, not a CLI executable")
        if maps_prefix is not None and not (engine == "vina" and scoring_function == "ad4"):
            raise ValueError("maps_prefix is supported only by native Vina's ad4 mode")
        if engine == "vina" and scoring_function == "ad4" and maps_prefix is None:
            raise ValueError("native Vina ad4 requires explicit precomputed maps_prefix")
        self.force_cap = _positive("force_cap", force_cap)
        if engine == "vina" and self.force_cap != 1000.:
            raise ValueError("native Vina binding final-score cap is fixed at 1000")
        self.engine, self.scoring_function = engine, scoring_function
        self.executable = os.fspath(executable) if executable is not None else None
        self.cnn_scoring, self.cnn_models = cnn_scoring, tuple(cnn_models)
        self.cnn_model_files = tuple(Path(path).expanduser().resolve() for path in cnn_model_files)
        self.device, self.maps_prefix = device, Path(maps_prefix).expanduser().resolve() if maps_prefix is not None else None

    @staticmethod
    def availability(engine: str, executable: str | os.PathLike[str] | None = None) -> dict[str, object]:
        if engine not in {"vina", "smina", "gnina"}:
            raise ValueError("unknown native engine")
        if engine == "vina":
            found = importlib.util.find_spec("vina") is not None
            return {"available": found, "reason": "optional vina Python binding found" if found else "install optional vina Python binding"}
        resolved = shutil.which(os.fspath(executable) if executable is not None else engine)
        return {"available": resolved is not None, "reason": resolved or f"install {engine} or provide its executable path"}

    def refine(
        self, receptor_pdbqt: str, ligand_pdbqt: str, ligand: PreparedLigand, *,
        seed: int = 42, max_steps: int = 100, cpu: int = 1, timeout: float = 120., padding: float = 4.,
    ) -> NativeRefinementResult:
        requested_seed, max_steps = _integer("seed", seed, 0), _integer("max_steps", max_steps, 0)
        # Vina reserves zero for nondeterministic auto-seeding. Keep the public
        # seed-zero workflow deterministic and expose the explicit translation.
        seed = 1 if self.engine == "vina" and requested_seed == 0 else requested_seed
        cpu, timeout, padding = _integer("cpu", cpu, 1), _positive("timeout", timeout), _positive("padding", padding)
        receptor, text = _validate_input(receptor_pdbqt, ligand_pdbqt, ligand)
        state = self.availability(self.engine, self.executable)
        if not state["available"]:
            raise ImportError(state["reason"])
        deadline = time.monotonic() + timeout
        original = ligand.coordinates(dtype=torch.float64)
        lower, upper = original.amin(0) - padding, original.amax(0) + padding
        if not (self.engine == "vina" and self.scoring_function == "ad4"):
            # Match native compute_vina_maps(force_even_voxels=True): its box
            # is expanded to an even number of 0.375-A intervals.
            center = (upper + lower) / 2
            intervals = torch.ceil((upper - lower) / .375).to(torch.long)
            if self.engine == "vina":
                intervals += intervals % 2
            half_size = intervals.to(torch.float64) * .375 / 2
            lower, upper = center - half_size, center + half_size
        metadata = {
            "contract": NATIVE_REFINEMENT_CONTRACT, "engine": self.engine, "scoring_function": self.scoring_function,
            "prepared_ligand_fingerprint": ligand.fingerprint,
            "input_receptor_sha256": _hash_text(receptor_pdbqt), "input_ligand_sha256": _hash_text(ligand_pdbqt),
            "used_receptor_sha256": _hash_text(receptor), "used_ligand_sha256": _hash_text(text),
            "seed": requested_seed, "requested_seed": requested_seed, "effective_seed": seed,
            "seed_policy": "native Vina zero is explicitly mapped to one; CLI seeds are passed through",
            "max_steps": max_steps, "cpu": cpu, "device": self.device, "force_cap": self.force_cap,
            "timeout_seconds": timeout, "padding_angstrom": padding,
            "bounds_angstrom": {"lower": lower.tolist(), "upper": upper.tolist()},
            "cnn_scoring": self.cnn_scoring, "cnn_models": list(self.cnn_models),
            "optimization_performed": max_steps > 0,
            "zero_steps_policy": "score only, preserve starting pose; native heuristic zero is not requested",
            "native_constraints": "local refinement only; no AnchorDock interaction/covalent/flexible-receptor constraints",
            "native_bond_perception": "native PDBQT distance/type perception; supplied graph correspondence does not prove identical native bonds",
            "native_hydrogen_mobility": "prepared hydrogen-only branches retained; CLI --flex_hydrogens explicitly enabled",
            "reference_source": _GNINA_SOURCE if self.engine == "gnina" else "native installed engine; version and content hashes below",
        }
        with tempfile.TemporaryDirectory(prefix="anchor-native-") as directory:
            work = Path(directory)
            (work / "receptor.pdbqt").write_text(receptor)
            (work / "ligand.pdbqt").write_text(text)
            if self.engine == "vina":
                initial, scores, output, logs = self._vina(work, ligand, seed, max_steps, cpu, deadline, lower, upper, metadata)
            else:
                initial, scores, output, logs = self._cli(work, ligand, seed, max_steps, cpu, padding, deadline, metadata)
            try:
                coordinates, mol = _output_pose(output, ligand)
            except (ValueError, KeyError) as exc:
                raise NativeBackendError(f"invalid native output pose: {exc}") from exc
            lower = coordinates.new_tensor(metadata["bounds_angstrom"]["lower"])
            upper = coordinates.new_tensor(metadata["bounds_angstrom"]["upper"])
            # Native united-atom box checks apply to heavy ligand atoms. Polar
            # H can extend beyond that boundary without adding an inter term.
            selected = torch.tensor([atom.GetAtomicNum() != 1 for atom in mol.GetAtoms()])
            if ((coordinates[selected] < lower - .001) | (coordinates[selected] > upper + .001)).any():
                raise NativeBackendError("native refinement returned a pose outside the explicit local box")
            metadata.update(output_pdbqt_sha256=_hash_text(output),
                            output_coordinates_sha256="sha256:" + hashlib.sha256(coordinates.numpy().tobytes()).hexdigest(),
                            log_sha256=_hash_text(logs), log_tail=logs[-16000:])
            scorer_state = {
                "contract": NATIVE_REFINEMENT_CONTRACT, "engine": self.engine, "scoring_function": self.scoring_function,
                "engine_version": metadata["engine_version"], "binary_sha256": metadata.get("binary_sha256"),
                "runtime_sha256": metadata.get("runtime_sha256"), "cnn_scoring": self.cnn_scoring,
                "worker_sha256": metadata.get("worker_sha256"),
                "map_sha256": metadata.get("map_sha256"),
                "map_conventions": metadata.get("map_conventions"),
                "environment_overrides": metadata.get("environment_overrides"),
                "cnn_models": metadata["cnn_models"], "cnn_model_hashes": list(metadata.get("cnn_model_sha256", {}).values()),
                "device": self.device,
                "force_cap": self.force_cap,
            }
            metadata["scorer_fingerprint"] = _hash_text(json.dumps(scorer_state, sort_keys=True, separators=(",", ":")))
        if self.engine == "gnina" and self.cnn_scoring == "refinement":
            objective = "native GNINA CNN pose loss plus native intramolecular/boundary terms; CNNaffinity and reported empirical affinity are diagnostics"
        else:
            objective = f"native {self.engine} {self.scoring_function} local search energy with native intra/boundary conventions"
        return NativeRefinementResult(self.engine, self.scoring_function, coordinates, scores, initial, objective, metadata, output, mol)

    def _vina(self, work, ligand, seed, max_steps, cpu, deadline, lower, upper, metadata):
        spec = importlib.util.find_spec("vina")
        package = Path(spec.origin).parent
        runtime_hashes = {path.relative_to(package).as_posix(): _hash_file(path)
                          for path in sorted(package.rglob("*")) if path.is_file() and path.suffix in {".py", ".so", ".pyd"}}
        if not runtime_hashes:
            raise NativeBackendError("cannot fingerprint native Vina runtime")
        options = {
            "sf": self.scoring_function, "cpu": cpu, "seed": seed, "max_steps": max_steps,
            "receptor": "receptor.pdbqt", "ligand": "ligand.pdbqt", "output": "refined.pdbqt", "result": "result.json",
            "center": ((upper + lower) / 2).tolist(), "size": (upper - lower).tolist(), "maps": None,
        }
        if self.maps_prefix is not None:
            hashes = {}
            for kind in sorted(set(ligand.ad4_types)) + ["e", "d"]:
                source = Path(f"{self.maps_prefix}.{kind}.map")
                if not source.is_file():
                    raise ValueError(f"required AD4 map is missing: {source.name}")
                destination = work / f"grid.{kind}.map"
                shutil.copyfile(source, destination)
                hashes[kind] = _hash_file(destination)
            metadata["map_sha256"] = hashes
            metadata["map_receptor_association"] = "caller supplied; maps are authoritative"
            metadata["map_conventions"] = "native Vina AD4 map/pair evaluation and standalone same-as-bound reporting"
            from ..core.autogrid import load_autogrid_maps
            fields = load_autogrid_maps(work / "grid", sorted(set(ligand.ad4_types)))
            origin = fields.electrostatic.origin
            end = [a + n * fields.electrostatic.spacing for a, n in zip(origin, fields.electrostatic.nelements, strict=True)]
            metadata["bounds_angstrom"] = {"lower": list(origin), "upper": end}
            metadata["box_policy"] = "AD4 map bounds; padding does not alter supplied maps"
            options["maps"] = "grid"
        (work / "options.json").write_text(json.dumps(options))
        argv = [sys.executable, "-c", _VINA_WORKER, "options.json"]
        result = _run(argv, cwd=work, deadline=deadline)
        try:
            values = json.loads((work / "result.json").read_text())
            initial, scores = _vina_scores(values["initial"]), _vina_scores(values["final"])
            output = (work / "refined.pdbqt").read_text()
        except (OSError, ValueError, KeyError) as exc:
            raise NativeBackendError("native Vina did not produce valid energies and PDBQT") from exc
        metadata.update(engine_version=importlib.metadata.version("vina"), runtime_sha256=runtime_hashes,
                        argv=[sys.executable, "-c", "<fingerprinted Vina worker>", "options.json"],
                        worker_sha256=_hash_text(_VINA_WORKER), worker_options=options, score_precision_decimals=3,
                        coordinate_precision_angstrom=.001)
        return initial, scores, output, result.stdout + result.stderr

    def _cli(self, work, ligand, seed, max_steps, cpu, padding, deadline, metadata):
        executable = Path(shutil.which(self.executable or self.engine)).resolve()
        signature = executable.stat()
        binary_hash = _hash_file(executable)
        env = {}
        if self.device != "cpu":
            index = int(self.device.split(":", 1)[1])
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible is not None:
                choices = [entry.strip() for entry in visible.split(",") if entry.strip()]
                if index >= len(choices) or choices[index] == "-1":
                    raise ValueError("requested GNINA CUDA index is not visible in CUDA_VISIBLE_DEVICES")
                selected = choices[index]
            else:
                selected = str(index)
            env["CUDA_VISIBLE_DEVICES"] = selected
            metadata["native_cuda_index"] = 0
        metadata["environment_overrides"] = env
        version = _run([str(executable), "--version"], cwd=work, deadline=deadline, env=env)
        version_text = (version.stdout + version.stderr).strip()
        if not version_text:
            raise NativeBackendError("native executable returned an empty version")
        common = [str(executable), "-r", "receptor.pdbqt", "-l", "ligand.pdbqt", "--scoring", self.scoring_function,
                  "--cpu", str(cpu), "--seed", str(seed), "--addH", "false", "--flex_hydrogens",
                  "--autobox_ligand", "ligand.pdbqt", "--autobox_add", str(padding),
                  "--force_cap", str(self.force_cap)]
        model_hashes = {}
        if self.engine == "gnina":
            common += ["--cnn_scoring", self.cnn_scoring]
            # GNINA's Torch path ignores --device >0. Restrict visibility in
            # this child and use its sole logical device zero instead.
            common += ["--no_gpu"] if self.device == "cpu" else ["--device", "0"]
            if self.cnn_models:
                common += ["--cnn", *self.cnn_models]
            elif self.cnn_model_files:
                model_names = []
                for index, source in enumerate(self.cnn_model_files):
                    if not source.is_file():
                        raise ValueError("CNN model file is missing")
                    destination = work / f"model-{index}{source.suffix}"
                    shutil.copyfile(source, destination)
                    model_hashes[f"{index}:{source.name}"] = _hash_file(destination)
                    model_names.append(destination.name)
                common += ["--cnn_model", *model_names]
            elif self.cnn_scoring != "none":
                metadata["cnn_models"] = ["native default ensemble; identified by executable version/hash"]
        initial_argv = common + ["--score_only", "--out", "initial.pdbqt"]
        refine_argv = common + ["--minimize", "--minimize_iters", str(max_steps), "--out", "refined.pdbqt"]
        before = _run(initial_argv, cwd=work, deadline=deadline, env=env)
        final = _run(refine_argv, cwd=work, deadline=deadline, env=env) if max_steps else before
        try:
            initial_text = (work / "initial.pdbqt").read_text()
            output = (work / "refined.pdbqt").read_text() if max_steps else initial_text
            try:
                initial_coords, _ = _output_pose(initial_text, ligand)
            except (ValueError, KeyError) as exc:
                raise NativeBackendError(f"invalid native output pose: {exc}") from exc
            if not torch.allclose(initial_coords, ligand.coordinates(dtype=torch.float64), atol=.0011, rtol=0.):
                raise NativeBackendError("native score_only unexpectedly moved the starting pose")
            initial = _cli_scores(initial_text, cnn=self.engine == "gnina" and self.cnn_scoring != "none")
            scores = _cli_scores(output, cnn=self.engine == "gnina" and self.cnn_scoring != "none")
        except (OSError, UnicodeError) as exc:
            raise NativeBackendError("native executable did not produce readable PDBQT outputs") from exc
        after_signature = executable.stat()
        if (signature.st_size, signature.st_mtime_ns, signature.st_ino) != (after_signature.st_size, after_signature.st_mtime_ns, after_signature.st_ino):
            raise NativeBackendError("native executable changed during refinement")
        metadata.update(engine_version=version_text[:4000], binary_sha256=binary_hash,
                        argv={"version": [str(executable), "--version"], "initial": initial_argv,
                              "refinement": refine_argv if max_steps else None},
                        cnn_model_sha256=model_hashes, coordinate_precision_angstrom=.001)
        logs = before.stdout + before.stderr
        if max_steps:
            logs += final.stdout + final.stderr
        return initial, scores, output, logs
