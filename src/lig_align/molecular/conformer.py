
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.ML.Cluster import Butina


def generate_conformers_and_cluster(mol: Chem.Mol,
                                    device: torch.device,
                                    num_confs: int = 1000,
                                    rmsd_threshold: float = 2.0,
                                    coordMap: dict[int, Point3D] | None = None) -> tuple[Chem.Mol, list[int]]:
    """
    Generate multiple conformers and cluster them to get a diverse representative set.

    When coordMap is provided:
    1. Embed conformers with approximate constraints
    2. Teleport MCS atoms to exact positions on ALL conformers (cheap)
    3. Cluster on post-teleport coords (reflects true pose diversity)
    4. MMFF optimize only cluster representatives

    Args:
        mol: RDKit molecule
        device: torch device (cuda/cpu)
        num_confs: Number of conformers to generate (default: 1000)
        rmsd_threshold: RMSD threshold in Angstroms for clustering (default: 2.0)
        coordMap: Optional coordinate constraints for specific atoms

    Returns:
        mol: Molecule with conformers (H atoms removed)
        representative_cids: List of conformer IDs (cluster centroids)
    """

    # 1. Generate Conformers
    print(f"Generating {num_confs} conformers...")
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = -1.0 # Disable pruning during generation to ensure requested number
    params.randomSeed = 42
    params.numThreads = 0 # Use all available cores

    if coordMap is not None:
        print(f"Applying rigid constraints for {len(coordMap)} atoms during generation...")
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs,
                                          pruneRmsThresh=-1.0,
                                          randomSeed=42,
                                          numThreads=0,
                                          coordMap=coordMap,
                                          ETversion=2)
        if len(cids) == 0:
            print("Warning: Constrained generation failed (likely triangle bounds conflict). Retrying without constraints...")
            cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    else:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

    if len(cids) == 0:
        raise RuntimeError("Conformer generation failed completely.")

    # 2. Teleport MCS atoms to exact positions BEFORE clustering
    #    This ensures clustering reflects true post-surgery diversity
    if coordMap is not None:
        for cid in cids:
            conf = mol.GetConformer(cid)
            for atom_idx, pos in coordMap.items():
                conf.SetAtomPosition(atom_idx, Point3D(pos.x, pos.y, pos.z))

    # 3. RMSD calculation for clustering on heavy atoms (PyTorch Batched Kabsch)
    mol_heavy = Chem.RemoveHs(mol)
    print(f"Calculating PyTorch batched RMSD matrix for {len(cids)} conformers...")
    n_confs = len(cids)
    n_atoms_heavy = mol_heavy.GetNumAtoms()

    dists = []
    if n_confs > 1:
        coords = torch.zeros((n_confs, n_atoms_heavy, 3), device=device)
        for i, cid in enumerate(cids):
            coords[i] = torch.tensor(mol_heavy.GetConformer(cid).GetPositions(), dtype=torch.float32, device=device)

        coords_centered = coords - coords.mean(dim=1, keepdim=True)
        H = torch.einsum('iac,jad->ijcd', coords_centered, coords_centered)
        H_flat = H.reshape(n_confs * n_confs, 3, 3)

        U, S, Vh = torch.linalg.svd(H_flat)
        V = Vh.transpose(1, 2)
        Ut = U.transpose(1, 2)
        R = torch.bmm(V, Ut)
        det = torch.linalg.det(R)

        S_sum = S.sum(dim=1)
        reflection_mask = det < 0
        S_sum[reflection_mask] -= 2 * S[reflection_mask, 2]

        norms = (coords_centered ** 2).sum(dim=(1, 2))
        norms_i = norms.unsqueeze(1).expand(n_confs, n_confs).flatten()
        norms_j = norms.unsqueeze(0).expand(n_confs, n_confs).flatten()

        dist_sq = norms_i + norms_j - 2 * S_sum
        dist_sq = torch.clamp(dist_sq, min=0.0)
        rmsd_matrix = torch.sqrt(dist_sq / n_atoms_heavy).view(n_confs, n_confs)

        # OPTIMIZED: Vectorized extraction of lower triangle
        triu_indices = torch.triu_indices(n_confs, n_confs, offset=1, device=device)
        dists = rmsd_matrix[triu_indices[1], triu_indices[0]].cpu().tolist()

    # 4. Butina Clustering
    print(f"Clustering conformers with RMSD threshold {rmsd_threshold}Å...")
    clusters = Butina.ClusterData(dists, n_confs, rmsd_threshold, isDistData=True)

    # Select centroid from each cluster
    representative_cids = [cluster[0] for cluster in clusters]

    # 5. MMFF optimize only representatives on full mol (with Hs) before H removal
    if coordMap is not None and len(representative_cids) > 0:
        print(f"MMFF optimizing {len(representative_cids)} representatives (skipping {n_confs - len(representative_cids)} non-representative conformers)...")
        for cid in representative_cids:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=200,
                                         nonBondedThresh=100.0, ignoreInterfragInteractions=False)

    # 6. Remove Hs for downstream
    mol = Chem.RemoveHs(mol)

    print(f"Selected {len(representative_cids)} representative conformers from {len(clusters)} clusters.")

    return mol, representative_cids
