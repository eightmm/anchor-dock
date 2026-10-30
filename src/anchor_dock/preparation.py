"""Optional preparation of ordinary ligand and whole-receptor chemical states.

RDKit coordinate ingestion is available without Meeko. Parameterization uses
Meeko only when requested; no pKa calculation or tautomer enumeration occurs.
PDBQT serialization rounds coordinates/charges to three decimals. Original
ligand coordinates remain available in ``source_mol``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from .core.io import ReceptorContext, _read_single_molecule
from .core.prepared import PreparedLigand, parse_pdbqt, receptor_context_from_prepared_pdbqt

PREPARATION_CONTRACT = "meeko-explicit-state-coordinate-preserving-v1"


class PreparationDependencyError(ImportError):
    """An optional preparation package is unavailable."""


def _meeko():
    try:
        return importlib.import_module("meeko")
    except ImportError as error:
        raise PreparationDependencyError("Meeko preparation requires the optional anchor-dock[prepare] dependencies") from error


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    return int(value)


def load_ligand_for_refinement(
    ligand: str | Path | Chem.Mol, *, seed: int = 42,
    generate_coordinates_if_missing: bool = True, add_hydrogens: bool = False,
) -> Chem.Mol:
    """Preserve graph order/stereo and existing 3D coordinates on a copy.

    One molecule/conformer is accepted. Missing/2D coordinates are embedded
    only when explicitly allowed. Hydrogens added to an existing pose do not
    move its existing atoms. This does not assign a new protonation state.
    """
    seed = _seed(seed)
    if not isinstance(generate_coordinates_if_missing, bool) or not isinstance(add_hydrogens, bool):
        raise ValueError("coordinate generation and hydrogen options must be bool")
    if isinstance(ligand, Chem.Mol):
        mol = Chem.Mol(ligand)
    else:
        text = str(ligand)
        path = Path(text)
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if is_file:
            mol = _read_single_molecule(path, remove_hydrogens=False)
        elif isinstance(ligand, Path) or path.suffix.lower() in {".sdf", ".mol", ".mol2", ".pdb", ".ent"}:
            raise FileNotFoundError(text)
        elif text.startswith("InChI="):
            mol = Chem.MolFromInchi(text)
        else:
            mol = Chem.MolFromSmiles(text)
    if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("ligand must be one valid connected molecule")
    Chem.SanitizeMol(mol)
    if mol.GetNumConformers() > 1:
        raise ValueError("ligand refinement requires exactly one selected conformer")
    has_pose = mol.GetNumConformers() == 1 and mol.GetConformer().Is3D()
    if has_pose and not torch.isfinite(torch.as_tensor(mol.GetConformer().GetPositions())).all():
        raise ValueError("ligand coordinates must be finite")
    if not has_pose and not generate_coordinates_if_missing:
        raise ValueError("ligand has no 3D pose; enable coordinate generation explicitly")
    original_graph = Chem.Mol(mol)
    if add_hydrogens or not has_pose:
        mol = Chem.AddHs(mol, addCoords=has_pose)
    if not has_pose:
        mol.RemoveAllConformers()
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = seed
        if AllChem.EmbedMolecule(mol, parameters) != 0:
            raise ValueError("seeded ETKDG conformer generation failed")
        if not add_hydrogens:
            # AddHs appends atoms. Copy back only the original atom positions,
            # retaining caller-supplied explicit H, atom order and graph state.
            conformer = Chem.Conformer(original_graph.GetNumAtoms())
            conformer.Set3D(True)
            for index in range(original_graph.GetNumAtoms()):
                conformer.SetAtomPosition(index, mol.GetConformer().GetAtomPosition(index))
            original_graph.RemoveAllConformers()
            original_graph.AddConformer(conformer)
            mol = original_graph
    mol.SetBoolProp("_AnchorDockCoordinatesGenerated", not has_pose)
    mol.SetIntProp("_AnchorDockInputAtomCount", original_graph.GetNumAtoms())
    return mol


def _index_mapping(pdbqt: str) -> dict[int, int]:
    result = {}
    for line in pdbqt.splitlines():
        if line.startswith("REMARK INDEX MAP "):
            values = [int(v) for v in line.split()[3:]]
            if len(values) % 2:
                raise ValueError("Meeko emitted an invalid INDEX MAP")
            for original, serial in zip(values[::2], values[1::2], strict=True):
                if original < 1 or serial < 1 or serial in result:
                    raise ValueError("Meeko emitted a duplicate/invalid INDEX MAP")
                result[serial] = original - 1
    if not result:
        raise ValueError("Meeko did not emit the required explicit INDEX MAP")
    return result


def _identity(payload: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class LigandPreparation:
    prepared: PreparedLigand
    pdbqt: str
    _source_mol: Chem.Mol
    _metadata: dict[str, object]

    @property
    def source_mol(self) -> Chem.Mol:
        return Chem.Mol(self._source_mol)

    @property
    def metadata(self) -> dict[str, object]:
        return deepcopy(self._metadata)

    @property
    def mol(self) -> Chem.Mol:
        return self.prepared.mol

    @property
    def fingerprint(self) -> str:
        return _identity(dict(prepared=self.prepared.fingerprint, pdbqt=self.pdbqt, metadata=self.metadata))


def prepare_ligand(ligand: str | Path | Chem.Mol, *, seed: int = 42,
                   generate_coordinates_if_missing: bool = True, charge_model: str = "gasteiger") -> LigandPreparation:
    """Prepare an ordinary ligand without changing a supplied 3D heavy pose."""
    if charge_model != "gasteiger":
        raise ValueError("automatic ligand preparation currently supports the explicit gasteiger charge model only")
    meeko = _meeko()
    source = load_ligand_for_refinement(ligand, seed=seed,
                                       generate_coordinates_if_missing=generate_coordinates_if_missing,
                                       add_hydrogens=True)
    setups = meeko.MoleculePreparation(charge_model=charge_model).prepare(Chem.Mol(source))
    if len(setups) != 1:
        raise ValueError("automatic preparation supports one ordinary ligand; macrocycle/reactive variants need explicit preparation")
    text, success, error = meeko.PDBQTWriterLegacy.write_string(setups[0], add_index_map=True)
    if not success:
        raise ValueError(f"Meeko ligand parameterization failed: {error}")
    prepared = PreparedLigand.from_pdbqt(source, text, serial_to_atom_index=_index_mapping(text))
    metadata = dict(contract=PREPARATION_CONTRACT, meeko_version=meeko.__version__, rdkit_version=rdBase.rdkitVersion,
                    charge_model=charge_model, protonation="supplied graph; no pKa or tautomer inference",
                    seed=int(seed), pdbqt_coordinate_decimals=3, pdbqt_charge_decimals=3,
                    coordinates_generated=source.GetBoolProp("_AnchorDockCoordinatesGenerated"),
                    input_atom_count=source.GetIntProp("_AnchorDockInputAtomCount"),
                    output_atom_to_input_atom=[index if index < source.GetIntProp("_AnchorDockInputAtomCount") else None
                                               for index in prepared.original_atom_indices],
                    source_graph_sha256=hashlib.sha256(Chem.MolToMolBlock(source).encode()).hexdigest())
    return LigandPreparation(prepared, text, Chem.Mol(source), metadata)


@dataclass(frozen=True)
class ReceptorPreparation:
    pdbqt: str
    _mol: Chem.Mol
    _serial_to_atom_index: dict[int, int]
    _metadata: dict[str, object]

    @property
    def mol(self) -> Chem.Mol:
        return Chem.Mol(self._mol)

    @property
    def serial_to_atom_index(self) -> dict[int, int]:
        return dict(self._serial_to_atom_index)

    @property
    def metadata(self) -> dict[str, object]:
        return deepcopy(self._metadata)

    def context(self, device: torch.device | str = "cpu") -> ReceptorContext:
        return receptor_context_from_prepared_pdbqt(self.mol, self.pdbqt,
                                                   serial_to_atom_index=self.serial_to_atom_index, device=device)

    @property
    def fingerprint(self) -> str:
        return _identity(dict(pdbqt=self.pdbqt, graph=Chem.MolToMolBlock(self.mol),
                              mapping=self.serial_to_atom_index, metadata=self.metadata))

    def autogrid_gpf(self, *, center, size, ligand_types: Sequence[str],
                     receptor_filename: str = "receptor.pdbqt", map_prefix: str = "receptor",
                     spacing: float = .375) -> str:
        """Create an AutoGrid input, rounding intervals UP to enclose the box.

        Center serialization is rounded to three decimals; dimensions expand
        to enclose the original requested physical box after that rounding.
        This does not execute AutoGrid or claim that maps have been generated.
        Write ``self.pdbqt`` at receptor_filename in the AutoGrid working dir.
        """
        if not isinstance(spacing, (int, float)) or isinstance(spacing, bool) or not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("AutoGrid spacing must be finite and positive")
        if abs(spacing - float(f"{spacing:.3f}")) > 1e-12:
            raise ValueError("AutoGrid spacing supports at most three decimal places in this serializer")
        vectors = []
        for name, value in (("center", center), ("size", size)):
            try:
                vector = tuple(float(v) for v in value)
            except (ValueError, TypeError) as error:
                raise ValueError(f"AutoGrid {name} must be a finite three-vector") from error
            if len(vector) != 3 or not all(math.isfinite(v) for v in vector) or (name == "size" and min(vector) <= 0):
                raise ValueError(f"AutoGrid {name} must be a finite {'positive ' if name == 'size' else ''}three-vector")
            vectors.append(vector)
        if isinstance(ligand_types, str):
            raise ValueError("ligand_types must be an explicit atom type sequence")
        types = tuple(ligand_types)
        if not types or len(set(types)) != len(types) or any(not isinstance(v, str) or v in {"e", "d"} or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", v) for v in types):
            raise ValueError("AutoGrid atom types must be unique safe names")
        for value in (receptor_filename, map_prefix):
            if not isinstance(value, str) or not value or any(v.isspace() or ord(v) < 32 or ord(v) == 127 for v in value) or "#" in value:
                raise ValueError("AutoGrid filenames must be nonempty and contain no whitespace or comment markers")
        _meeko()
        gridbox = importlib.import_module("meeko.gridbox")
        center_serialized = tuple(float(f"{v:.3f}") for v in vectors[0])
        sizes = [size + 2 * abs(a-b) for size, a, b in zip(vectors[1], vectors[0], center_serialized, strict=True)]
        intervals = tuple(2 * math.ceil(v / (2 * spacing)) for v in sizes)
        receptor_types = sorted({a.atom_type for a in parse_pdbqt(self.pdbqt, rigid=True).atoms})
        text, _ = gridbox.get_gpf_string(center_serialized, [v * spacing for v in intervals], receptor_filename,
                                               receptor_types, types, map_prefix=map_prefix, spacing=spacing)
        lines = text.splitlines()
        # Avoid floor-roundoff in the helper's interval calculation.
        lines[0] = "npts " + " ".join(str(v) for v in intervals)
        return "\n".join(lines) + "\n"


def _pdb_policy(text: str, default_altloc: str | None, wanted_altloc: Mapping[str, str],
                deleted: set[str]) -> tuple[str, dict[str, tuple[str, set[str]]]]:
    for choice in (default_altloc, *wanted_altloc.values()):
        if choice is not None and (not isinstance(choice, str) or len(choice) != 1 or not choice.isalnum()):
            raise ValueError("alternate-location choices must be one alphanumeric character")
    residues, seen, lines = {}, set(), []
    if sum(line.startswith("MODEL ") for line in text.splitlines()) > 1:
        raise ValueError("select one receptor model before preparation")
    for line in text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            key = f"{line[21:22].strip()}:{line[22:26].strip()}{line[26:27].strip()}"
            if key in deleted:
                continue
            altloc = line[16:17].strip()
            chosen = wanted_altloc.get(key, default_altloc)
            if altloc and chosen is None:
                raise ValueError(f"receptor alternate location at {key}; select wanted_altloc/default_altloc explicitly")
            if altloc and altloc != chosen:
                continue
            name, resname = line[12:16].strip(), line[17:20].strip()
            atom_key = (key, name)
            if atom_key in seen:
                raise ValueError(f"duplicate receptor atom identity {atom_key}")
            seen.add(atom_key)
            if key not in residues:
                residues[key] = (resname, set())
            if residues[key][0] != resname:
                raise ValueError(f"multiple residue names at {key}")
            residues[key][1].add(name)
            line = line[:16] + " " + line[17:]
        if not line.startswith(("MODEL ", "ENDMDL")):
            lines.append(line)
    if not residues:
        raise ValueError("receptor has no selected atoms")
    unknown_choices = set(wanted_altloc) - set(residues)
    if unknown_choices:
        raise ValueError(f"alternate-location selectors did not match selected receptor residues: {sorted(unknown_choices)}")
    return "\n".join(lines) + "\n", residues


def _polymer_graph(polymer, *, preserve_pqr_charges: bool) -> tuple[Chem.Mol, dict[int, int]]:
    """Build graph order and serial mapping from explicit Meeko indices."""
    graph, offsets, mapping, serial = Chem.RWMol(), {}, {}, 0
    monomers = polymer.get_valid_monomers()
    for residue_id, monomer in monomers.items():
        offsets[residue_id] = graph.GetNumAtoms()
        local = monomer.rdkit_mol
        inverse = {value: key for key, value in monomer.molsetup_mapidx.items()}
        if len(inverse) != local.GetNumAtoms():
            raise ValueError("Meeko receptor atom mapping is incomplete")
        charges = {}
        if preserve_pqr_charges:
            for atom in local.GetAtoms():
                raw_index = monomer.mapidx_to_raw.get(atom.GetIdx())
                if raw_index is None:
                    raise ValueError("PQR is missing atoms required by the selected chemical template")
                raw = monomer.raw_rdkit_mol.GetAtomWithIdx(raw_index)
                charges[atom.GetIdx()] = float(raw.GetProp("PQRCharge"))
        for atom in local.GetAtoms():
            graph.AddAtom(Chem.Atom(atom))
        for bond in local.GetBonds():
            graph.AddBond(offsets[residue_id] + bond.GetBeginAtomIdx(),
                          offsets[residue_id] + bond.GetEndAtomIdx(), bond.GetBondType())
        if preserve_pqr_charges:
            # Meeko rectifies residue charges after reading them. Restore the
            # caller's values, conserving charge when nonpolar H are merged.
            for atom in local.GetAtoms():
                setup_atom = monomer.molsetup.atoms[inverse[atom.GetIdx()]]
                if setup_atom.is_ignore:
                    if atom.GetAtomicNum() != 1 or atom.GetDegree() != 1:
                        raise ValueError("PQR charge preservation cannot merge a non-hydrogen atom")
                    parent = atom.GetNeighbors()[0].GetIdx()
                    charges[parent] += charges[atom.GetIdx()]
            for atom in local.GetAtoms():
                monomer.molsetup.atoms[inverse[atom.GetIdx()]].charge = charges[atom.GetIdx()]
        for atom in monomer.molsetup.atoms:
            if not atom.is_ignore:
                if atom.index not in monomer.molsetup_mapidx:
                    raise ValueError("Meeko emitted an unsupported off-site receptor atom")
                serial += 1
                mapping[serial] = offsets[residue_id] + monomer.molsetup_mapidx[atom.index]
    for (left_res, right_res), pairs in polymer.bonds.items():
        if left_res not in monomers or right_res not in monomers:
            raise ValueError("prepared receptor has a bond to an ignored residue")
        left, right = monomers[left_res], monomers[right_res]
        for a, b in pairs:
            graph.AddBond(offsets[left_res] + left.mapidx_from_raw[a],
                          offsets[right_res] + right.mapidx_from_raw[b], Chem.BondType.SINGLE)
    graph = graph.GetMol()
    conformer = Chem.Conformer(graph.GetNumAtoms())
    for key, monomer in monomers.items():
        for setup_index, local_index in monomer.molsetup_mapidx.items():
            atom = monomer.molsetup.atoms[setup_index]
            conformer.SetAtomPosition(offsets[key] + local_index, atom.coord)
            chain, residue = key.split(":")
            info = Chem.AtomPDBResidueInfo()
            raw_index = monomer.mapidx_to_raw.get(local_index)
            raw_info = (None if raw_index is None else monomer.raw_rdkit_mol.GetAtomWithIdx(raw_index).GetPDBResidueInfo())
            info.SetName(raw_info.GetName() if raw_info is not None else f"{monomer.atom_names[local_index]:>4}")
            if raw_info is not None:
                info.SetSerialNumber(raw_info.GetSerialNumber())
            info.SetResidueName(monomer.input_resname)
            info.SetChainId(chain)
            info.SetResidueNumber(int(residue[:-1] if residue[-1].isalpha() else residue))
            info.SetInsertionCode(residue[-1] if residue[-1].isalpha() else " ")
            graph.GetAtomWithIdx(offsets[key] + local_index).SetMonomerInfo(info)
    graph.AddConformer(conformer)
    Chem.SanitizeMol(graph)
    unresolved = []
    for atom in graph.GetAtoms():
        if atom.GetTotalNumHs(includeNeighbors=False):
            info = atom.GetPDBResidueInfo()
            unresolved.append(f"{info.GetChainId()}:{info.GetResidueNumber()}{info.GetInsertionCode().strip()}:{info.GetName().strip()}")
    if unresolved:
        raise ValueError("unresolved template links or non-explicit hydrogens at " + ", ".join(unresolved[:8])
                         + "; use the whole receptor and declare correct terminal templates in set_template")
    return graph, mapping


def prepare_receptor(receptor: str | Path, *, charge_model: str | None = None,
                     set_template: Mapping[str, str] | None = None,
                     residues_to_delete: Sequence[str] = (), default_altloc: str | None = None,
                     wanted_altloc: Mapping[str, str] | None = None) -> ReceptorPreparation:
    """Prepare the entire supplied PDB/PQR before any pocket selection.

    PQR defaults to reading its existing charges; requesting re-assignment is
    rejected. Names/templates and supplied H declare the chemical state. No pKa
    or tautomer selection occurs. Unknown residues/missing heavy atoms fail.
    """
    path = Path(receptor)
    if path.suffix.lower() not in {".pdb", ".ent", ".pqr"}:
        raise ValueError("automatic receptor preparation accepts PDB or PQR")
    text = path.read_text()
    pqr = path.suffix.lower() == ".pqr"
    charge_model = ("read" if pqr else "gasteiger") if charge_model is None else charge_model
    if charge_model != ("read" if pqr else "gasteiger"):
        raise ValueError("PDB preparation uses gasteiger; PQR preserves supplied charges with charge_model='read'")
    if isinstance(residues_to_delete, str):
        raise ValueError("residues_to_delete must be an explicit residue-ID sequence")
    templates, deleted, altlocs = dict(set_template or {}), set(residues_to_delete), dict(wanted_altloc or {})
    meeko = _meeko()
    library = meeko.ResidueChemTemplates.create_from_defaults()
    supported = set(library.residue_templates) | set(library.ambiguous)
    if any(value not in supported for value in templates.values()):
        raise ValueError("custom template names require an explicitly supported Meeko residue template")
    if pqr:
        raw = meeko.Polymer._pqr_to_residue_mols(text)
        residues = {key: (entry[1], {a.GetPDBResidueInfo().GetName().strip() for a in entry[0].GetAtoms()})
                    for key, entry in raw.items() if key not in deleted and entry[0] is not None}
        if default_altloc is not None or altlocs:
            raise ValueError("PQR does not carry alternate-location selectors")
        selected_text = text
    else:
        selected_text, residues = _pdb_policy(text, default_altloc, altlocs, deleted)
    if not residues or any(name not in supported and key not in templates for key, (name, _) in residues.items()):
        raise ValueError("receptor includes unknown residues; declare supported templates or explicit deletion before preparation")
    for key, (name, atoms) in residues.items():
        if name == "HIS" and not atoms.intersection({"HD1", "HE2"}) and key not in templates:
            raise ValueError(f"HIS protonation at {key} is unspecified; supply polar H or set_template HID/HIE/HIP")
    preparator = meeko.MoleculePreparation(charge_model=charge_model, charge_atom_prop="PQRCharge" if pqr else None)
    try:
        factory = meeko.Polymer.from_pqr_string if pqr else meeko.Polymer.from_pdb_string
        polymer = factory(selected_text, chem_templates=library, mk_prep=preparator,
                          set_template=templates, residues_to_delete=list(deleted), allow_bad_res=False)
        graph, mapping = _polymer_graph(polymer, preserve_pqr_charges=pqr)
        pdbqt, flexible = meeko.PDBQTWriterLegacy.write_from_polymer(polymer)
        if flexible:
            raise ValueError("automatic preparation requires a fixed receptor")
        # Validate the final graph/serialized coordinates before returning it.
        receptor_context_from_prepared_pdbqt(graph, pdbqt, serial_to_atom_index=mapping)
    except (ValueError, RuntimeError) as error:
        raise ValueError("receptor preparation failed; verify complete residues, declared termini/HIS templates and supplied H: " + str(error)) from error
    serialized_charges = [a.charge for a in parse_pdbqt(pdbqt, rigid=True).atoms]
    expected_charges = [a.charge for mon in polymer.get_valid_monomers().values() for a in mon.molsetup.atoms if not a.is_ignore]
    metadata = dict(contract=PREPARATION_CONTRACT, meeko_version=meeko.__version__, rdkit_version=rdBase.rdkitVersion,
                    source_sha256=hashlib.sha256(text.encode()).hexdigest(), input_format="pqr" if pqr else "pdb",
                    charge_model=charge_model, charge_preservation="PQR input charges; nonpolar-H merge; 3-decimal serialization" if pqr else "Meeko residue template gasteiger",
                    protonation="named/template chemical states; no pKa inference",
                    scope="entire supplied receptor before pocket selection", pdbqt_charge_decimals=3,
                    set_template=templates, selected_templates={key: mon.residue_template_key for key, mon in polymer.get_valid_monomers().items()},
                    residues_deleted=sorted(deleted), default_altloc=default_altloc, wanted_altloc=altlocs,
                    source_radius_role="PQR radii are not substituted for AD4/XS/LJ parameters" if pqr else None)
    metadata.update(charge_net_before_serialization=sum(expected_charges), charge_net_serialized=sum(serialized_charges),
                    maximum_charge_rounding_error=max(abs(a-b) for a, b in zip(serialized_charges, expected_charges, strict=True)),
                    charge_policy=charge_model, protonation_policy="as named and explicit residue templates; no pKa inference")
    return ReceptorPreparation(pdbqt, graph, mapping, metadata)
