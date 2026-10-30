"""Explicit prepared graph/PDBQT mapping; no chemistry is inferred from PDBQT."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import torch
from rdkit import Chem

from .features import AUTHORITATIVE_TYPING_VERSION, compute_atom_features_from_ad4
from .io import ReceptorContext, receptor_structure_fingerprint
from .topology import PREPARED_ROTORS_PROPERTY

PREPARED_INPUT_CONTRACT = "prepared-ad4-graph-tree-v1"


def _graph_identity(mol):
    mol.UpdatePropertyCache(strict=False)
    atoms = tuple((a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope(), a.GetIsAromatic(), str(a.GetChiralTag()),
                   a.GetNumExplicitHs(), a.GetNumImplicitHs()) for a in mol.GetAtoms())
    bonds = tuple(sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                          max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType()), str(b.GetStereo()),
                          tuple(b.GetStereoAtoms()))
                         for b in mol.GetBonds()))
    return atoms, bonds


def _index(value, size):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < size:
        raise ValueError("mapping/branch endpoint must be a valid integer graph atom index")
    return int(value)


def _cut_side(mol, parent, child):
    if parent == child or mol.GetBondBetweenAtoms(parent, child) is None:
        raise ValueError("prepared branch must correspond to an existing graph bond")
    side, queue = {child}, [child]
    while queue:
        current = queue.pop()
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            nxt = neighbor.GetIdx()
            if {current, nxt} == {parent, child}:
                continue
            if nxt not in side:
                side.add(nxt)
                queue.append(nxt)
    if parent in side:
        raise ValueError("prepared branches through rings/macrocycles are unsupported")
    return side


@dataclass(frozen=True)
class PDBQTAtom:
    serial: int
    coordinates: tuple[float, float, float]
    charge: float
    atom_type: str
    group: int


@dataclass(frozen=True)
class ParsedPDBQT:
    atoms: tuple[PDBQTAtom, ...]
    branches: tuple[tuple[int, int], ...]
    torsdof: int


def parse_pdbqt(text: str, *, rigid: bool = False) -> ParsedPDBQT:
    """Read one strict ordinary ligand or rigid receptor, preserving serial IDs.

    Coordinates/charges use fixed columns. Ligands require a closed torsion
    tree. Flexible receptors and multiple models are unsupported. Atom types
    are validated against the supplied chemical graph during preparation.
    """
    if not isinstance(text, str):
        raise TypeError("PDBQT content must be text; read the file explicitly")
    atoms, branches, stack = [], [], []
    serials, groups = set(), {0: set()}
    group, next_group = 0, 1
    root_seen = root_closed = False
    torsdof = None
    for lineno, line in enumerate(text.splitlines(), 1):
        words = line.split()
        if not words or words[0] in {"REMARK", "WARNING"}:
            continue
        tag, word = line[:6].strip(), words[0]
        if tag in {"ATOM", "HETATM"}:
            if not rigid and (not root_seen or (root_closed and not stack) or torsdof is not None):
                raise ValueError(f"atom outside ROOT/BRANCH at line {lineno}")
            try:
                serial = int(line[6:11])
                xyz = tuple(float(line[start:start+8]) for start in (30, 38, 46))
                charge, kind = float(line[70:76]), line[77:].strip()
            except ValueError as error:
                raise ValueError(f"invalid fixed-column atom at line {lineno}") from error
            if serial <= 0 or serial in serials:
                raise ValueError("PDBQT atom serials must be unique positive integers")
            if not kind or len(kind.split()) != 1 or not all(math.isfinite(v) for v in (*xyz, charge)):
                raise ValueError("PDBQT coordinates/charges must be finite and type unambiguous")
            serials.add(serial)
            groups[group].add(serial)
            atoms.append(PDBQTAtom(serial, xyz, charge, kind, group))
        elif rigid and word in {"TER", "END"}:
            continue
        elif not rigid and word == "ROOT":
            if root_seen or len(words) != 1:
                raise ValueError("PDBQT requires one ROOT")
            root_seen = True
        elif not rigid and word == "ENDROOT":
            if not root_seen or root_closed or stack or not groups[0] or len(words) != 1:
                raise ValueError("invalid ENDROOT")
            root_closed = True
        elif not rigid and word == "BRANCH":
            if not root_closed or torsdof is not None or len(words) != 3:
                raise ValueError("BRANCH must follow ENDROOT and precede TORSDOF")
            try:
                left, right = map(int, words[1:])
            except ValueError as error:
                raise ValueError("BRANCH requires integer serials") from error
            if left not in groups[group] or right in serials or left == right or right <= 0:
                raise ValueError("BRANCH parent must be in immediate group; child must be new")
            branches.append((left, right))
            stack.append((left, right, group))
            group, next_group = next_group, next_group + 1
            groups[group] = set()
        elif not rigid and word == "ENDBRANCH":
            try:
                pair = tuple(map(int, words[1:]))
            except ValueError as error:
                raise ValueError("ENDBRANCH requires integer serials") from error
            if not stack or len(pair) != 2 or pair != stack[-1][:2] or pair[1] not in groups[group]:
                raise ValueError("ENDBRANCH disagrees with open branch/child")
            _, _, group = stack.pop()
        elif not rigid and word == "TORSDOF":
            if not root_closed or stack or torsdof is not None or len(words) != 2:
                raise ValueError("TORSDOF requires a closed tree and may occur once")
            try:
                torsdof = int(words[1])
            except ValueError as error:
                raise ValueError("TORSDOF must be a nonnegative integer") from error
            if torsdof < 0:
                raise ValueError("TORSDOF must be a nonnegative integer")
        else:
            raise ValueError(f"unsupported PDBQT record {word!r} at line {lineno}")
    if not atoms or stack or (not rigid and (not root_closed or torsdof is None)):
        raise ValueError("incomplete PDBQT atom set or torsion tree")
    return ParsedPDBQT(tuple(atoms), tuple(branches), 0 if rigid else torsdof)


@dataclass(frozen=True)
class PreparedLigand:
    """Snapshot of graph/types/charges/tree, with explicit polar H retained.

    ``mol`` returns a copy in original graph order; use with ``coordinates()``.
    Vina's half-integral endpoint rotor count is distinct from AD4 TORSDOF.
    """
    _mol: Chem.Mol
    ad4_types: tuple[str, ...]
    charges: tuple[float, ...]
    _coordinates: tuple[tuple[float, float, float], ...]
    rotatable_bonds: tuple[tuple[int, int], ...]
    torsdof: int
    original_atom_indices: tuple[int, ...]
    pdbqt_serials: tuple[int, ...]

    @classmethod
    def from_arrays(cls, mol: Chem.Mol, ad4_types: Sequence[str], charges: Sequence[float], *,
                    rotatable_bonds: Sequence[tuple[int, int]] = (), torsdof: int = 0,
                    coordinates=None, _allow_disconnected: bool = False) -> PreparedLigand:
        """Snapshot aligned arrays without assigning missing chemistry."""
        graph = Chem.Mol(mol)
        if not graph.GetNumAtoms() or (not _allow_disconnected and len(Chem.GetMolFrags(graph)) != 1):
            raise ValueError("prepared ligand requires one nonempty connected chemical graph")
        if isinstance(ad4_types, (str, bytes)):
            raise ValueError("prepared types must be atom-order aligned")
        types = tuple(ad4_types)
        compute_atom_features_from_ad4(graph, types, "cpu")
        values = tuple(float(q) for q in charges)
        if len(values) != graph.GetNumAtoms() or not all(math.isfinite(q) for q in values):
            raise ValueError("one finite prepared charge is required per atom")
        if isinstance(torsdof, bool) or not isinstance(torsdof, Integral) or torsdof < 0:
            raise ValueError("torsdof must be a nonnegative integer")
        if coordinates is None:
            if graph.GetNumConformers() != 1:
                raise ValueError("prepared input requires exactly one conformer or explicit coordinates")
            coordinates = graph.GetConformer().GetPositions()
        xyz = torch.as_tensor(coordinates, dtype=torch.float64)
        if xyz.shape != (graph.GetNumAtoms(), 3) or not torch.isfinite(xyz).all():
            raise ValueError("prepared coordinates must be finite [N,3]")
        bonds, seen = [], set()
        for value in rotatable_bonds:
            if not isinstance(value, Sequence) or len(value) != 2:
                raise ValueError("prepared rotatable bonds must be index pairs")
            left, right = (_index(i, graph.GetNumAtoms()) for i in value)
            pair = tuple(sorted((left, right)))
            if pair in seen:
                raise ValueError("duplicate prepared branch")
            _cut_side(graph, left, right)
            seen.add(pair)
            bonds.append((left, right))
        bonds = tuple(bonds)
        graph.SetProp(PREPARED_ROTORS_PROPERTY, json.dumps(bonds))
        graph.RemoveAllConformers()
        conformer = Chem.Conformer(graph.GetNumAtoms())
        for i, position in enumerate(xyz.tolist()):
            conformer.SetAtomPosition(i, position)
        graph.AddConformer(conformer)
        return cls(graph, types, values, tuple(tuple(v) for v in xyz.tolist()), bonds, int(torsdof),
                   tuple(range(graph.GetNumAtoms())), ())

    @classmethod
    def from_pdbqt(cls, mol: Chem.Mol, text: str, *, serial_to_atom_index: Mapping[int, int],
                   coordinate_tolerance: float = .002) -> PreparedLigand:
        """Read coordinate-checked mapping; only nonpolar C-H may be omitted."""
        return _prepared_from_pdbqt(cls, mol, text, serial_to_atom_index, coordinate_tolerance, rigid=False)

    @property
    def mol(self) -> Chem.Mol:
        return Chem.Mol(self._mol)

    @property
    def fingerprint(self) -> str:
        data = dict(contract=PREPARED_INPUT_CONTRACT, graph=_graph_identity(self._mol),
                    types=self.ad4_types, charges=self.charges, coordinates=self._coordinates,
                    branches=self.rotatable_bonds, torsdof=self.torsdof,
                    original_indices=self.original_atom_indices, serials=self.pdbqt_serials)
        return "sha256:" + hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def validate_mol(self, mol: Chem.Mol) -> None:
        if _graph_identity(mol) != _graph_identity(self._mol):
            raise ValueError("prepared ligand graph/order disagrees with supplied mol; use prepared.mol")

    def coordinates(self, device: torch.device | str = "cpu", *, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(self._coordinates, device=device, dtype=dtype)

    def features(self, device: torch.device | str = "cpu", *, include_mobility: bool = True) -> dict[str, object]:
        result = compute_atom_features_from_ad4(self._mol, self.ad4_types, device)
        result.update(ad4_types=self.ad4_types,
                      ad4_charges=torch.tensor(self.charges, dtype=torch.float32, device=device),
                      ad4_torsdof=self.torsdof, vina_num_tors=self.vina_num_tors,
                      prepared_input_contract=PREPARED_INPUT_CONTRACT, prepared_fingerprint=self.fingerprint)
        if include_mobility:
            result["prepared_intramolecular_pairs"] = self.intramolecular_pairs
        return result

    @property
    def vina_num_tors(self) -> float:
        """Official endpoint count when native reconstructed bonds match graph."""
        degree = [sum(n.GetAtomicNum() != 1 for n in a.GetNeighbors()) for a in self._mol.GetAtoms()]
        total = 0.
        for left, right in self.rotatable_bonds:
            if self._mol.GetAtomWithIdx(left).GetAtomicNum() != 1 and self._mol.GetAtomWithIdx(right).GetAtomicNum() != 1:
                total += .5 * (int(degree[right] > 1) + int(degree[left] > 1))
        return total

    def intramolecular_mask(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Variable-distance pairs beyond graph 1-4, including polar H atoms."""
        size = self._mol.GetNumAtoms()
        variable = torch.zeros(size, size, dtype=torch.bool)
        for left, right in self.rotatable_bonds:
            side = _cut_side(self._mol, left, right)
            inside = torch.tensor([i in side and i != right for i in range(size)])
            outside = torch.tensor([i not in side and i != left for i in range(size)])
            variable |= inside[:, None] & outside[None, :]
            variable |= outside[:, None] & inside[None, :]
        return (variable & (torch.from_numpy(Chem.GetDistanceMatrix(self._mol)) > 3)).to(device=device)

    @property
    def intramolecular_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(map(tuple, torch.triu(self.intramolecular_mask(), diagonal=1).nonzero().tolist()))


def _prepared_from_pdbqt(cls, mol, text, mapping, tolerance, *, rigid):
    if not isinstance(mapping, Mapping):
        raise ValueError("an explicit serial_to_atom_index mapping is required")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("coordinate_tolerance must be finite and nonnegative")
    if mol.GetNumConformers() != 1:
        raise ValueError("coordinate-checked PDBQT mapping requires exactly one graph conformer")
    parsed = parse_pdbqt(text, rigid=rigid)
    serials = {a.serial for a in parsed.atoms}
    if set(mapping) != serials or any(isinstance(i, bool) or not isinstance(i, Integral) for i in mapping):
        raise ValueError("mapping keys must equal PDBQT atom serials")
    mapped = [_index(mapping[a.serial], mol.GetNumAtoms()) for a in parsed.atoms]
    if len(set(mapped)) != len(mapped):
        raise ValueError("PDBQT mapping must be one-to-one")
    present = set(mapped)
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in present:
            if atom.GetAtomicNum() != 1 or any(n.GetAtomicNum() != 6 for n in atom.GetNeighbors()):
                raise ValueError("PDBQT mapping may omit only nonpolar carbon-bonded graph hydrogens")
    conf = mol.GetConformer()
    for atom, original in zip(parsed.atoms, mapped, strict=True):
        expected = conf.GetAtomPosition(original)
        if max(abs(expected[i] - atom.coordinates[i]) for i in range(3)) > tolerance:
            raise ValueError(f"PDBQT coordinate mapping disagrees at serial {atom.serial}")
    original_indices = tuple(sorted(present))
    old_to_new = {old: new for new, old in enumerate(original_indices)}
    by_index = {old_to_new[original]: atom for atom, original in zip(parsed.atoms, mapped, strict=True)}
    graph = Chem.RWMol(mol)
    for i in reversed(range(mol.GetNumAtoms())):
        if i not in present:
            graph.RemoveAtom(i)
    graph = graph.GetMol()
    graph.UpdatePropertyCache(strict=False)
    serial_to_new = {a.serial: old_to_new[mapping[a.serial]] for a in parsed.atoms}
    branches = tuple((serial_to_new[a], serial_to_new[b]) for a, b in parsed.branches)
    for left, right in branches:
        _cut_side(graph, left, right)
    branch_set = {frozenset(pair) for pair in branches}
    for bond in graph.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (by_index[left].group != by_index[right].group) != (frozenset((left, right)) in branch_set):
            raise ValueError("PDBQT rigid groups/branches disagree with supplied chemical graph")
    result = cls.from_arrays(graph, [by_index[i].atom_type for i in range(len(by_index))],
                             [by_index[i].charge for i in range(len(by_index))],
                             rotatable_bonds=branches, torsdof=parsed.torsdof,
                             coordinates=[by_index[i].coordinates for i in range(len(by_index))],
                             _allow_disconnected=rigid)
    return cls(result._mol, result.ad4_types, result.charges, result._coordinates,
               result.rotatable_bonds, result.torsdof, original_indices,
               tuple(by_index[i].serial for i in range(len(by_index))))


def receptor_context_from_prepared_pdbqt(mol: Chem.Mol, text: str, *, serial_to_atom_index: Mapping[int, int],
                                       device: torch.device | str = "cpu", coordinate_tolerance: float = .002) -> ReceptorContext:
    """Rigid context from prepared graph and mapped PDBQT, bypassing templates.

    Residue metadata is preserved; polar H, connectivity, and protonation must
    be supplied. Disconnected receptor graphs are permitted, unlike ligands.
    """
    data = _prepared_from_pdbqt(PreparedLigand, mol, text, serial_to_atom_index, coordinate_tolerance, rigid=True)
    features, coords = data.features(device, include_mobility=False), data.coordinates(device)
    return ReceptorContext(mol=data.mol, coords=coords, features=features, source_path=None,
                           device=torch.device(device), structure_fingerprint=receptor_structure_fingerprint(coords, features),
                           source_fingerprint="sha256:" + hashlib.sha256(text.encode()).hexdigest(),
                           atom_typing_version=AUTHORITATIVE_TYPING_VERSION)
