import io

from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


def get_2d_image(mol, highlight_atoms, align_ref=None, match_pairs=None):
    """
    Renders a 2D depiction of the molecule using RDKit, highlighting specific atoms.
    Optionally aligns the coordinates to a reference molecule to preserve spatial orientation.
    """
    mol_copy = Chem.Mol(mol)
    try:
        mol_copy.RemoveAllConformers()
    except Exception:
        pass

    if align_ref and match_pairs:
        ref_copy = Chem.Mol(align_ref)
        try:
            ref_copy.RemoveAllConformers()
        except Exception:
            pass
        rdDepictor.Compute2DCoords(ref_copy)
        
        coord_map = {}
        conf = ref_copy.GetConformer()
        for r_idx, q_idx in match_pairs:
            coord_map[q_idx] = conf.GetAtomPosition(r_idx)
            
        try:
            rdDepictor.Compute2DCoords(mol_copy, coordMap=coord_map)
        except Exception:
            rdDepictor.Compute2DCoords(mol_copy)
    else:
        rdDepictor.Compute2DCoords(mol_copy)
        
    d2d = rdMolDraw2D.MolDraw2DCairo(400, 400)
    opts = d2d.drawOptions()
    opts.clearBackground = False
    
    color = (0.9, 0.49, 0.13, 0.6) # Gold transparent
    atom_colors = {i: color for i in highlight_atoms}
    
    highlight_bonds = []
    bond_colors = {}
    for bond in mol_copy.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i in highlight_atoms and j in highlight_atoms:
            idx = bond.GetIdx()
            highlight_bonds.append(idx)
            bond_colors[idx] = color
            
    rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol_copy, highlightAtoms=highlight_atoms, highlightAtomColors=atom_colors, highlightBonds=highlight_bonds, highlightBondColors=bond_colors)
    d2d.FinishDrawing()
    img = Image.open(io.BytesIO(d2d.GetDrawingText()))
    return img


def draw_molecule_3d(ax, mol, coords, color, alpha, label, highlight_indices=None, highlight_color='#e67e22'):
    """
    Plots a 3D scatter graph of a molecule on a Matplotlib 3D Axis.
    Highlights specific atoms and bonds if indices are provided.
    """
    xs = coords[:, 0]
    ys = coords[:, 1]
    zs = coords[:, 2]
    
    if highlight_indices is not None:
        c_list = [highlight_color if i in highlight_indices else color for i in range(mol.GetNumAtoms())]
        ax.scatter(xs, ys, zs, c=c_list, s=50, alpha=alpha, label=label)
    else:
        ax.scatter(xs, ys, zs, c=color, s=50, alpha=alpha, label=label)
    
    # Add element labels
    for i in range(mol.GetNumAtoms()):
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        c = highlight_color if (highlight_indices is not None and i in highlight_indices) else color
        ax.text(xs[i]+0.1, ys[i]+0.1, zs[i]+0.1, symbol, size=8, color=c, alpha=alpha)
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        c = highlight_color if (highlight_indices is not None and i in highlight_indices and j in highlight_indices) else color
        ax.plot([coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                [coords[i, 2], coords[j, 2]],
                color=c, alpha=alpha, linewidth=2)
