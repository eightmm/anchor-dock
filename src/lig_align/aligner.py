import torch
from rdkit import Chem
from typing import List, Tuple, Optional, Dict

from .molecular import generate_conformers_and_cluster, compute_vina_features
from .molecular.mcs import find_mcs_with_positions
from .scoring import vina_scoring
from .selection import final_selection
from .optimization import optimize_torsions_vina

class LigandAligner:
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def step1_generate_conformers(self,
                                  mol: Chem.Mol,
                                  num_confs: int = 1000,
                                  rmsd_threshold: float = 1.0,
                                  coordMap: Optional[Dict[int, 'rdkit.Geometry.Point3D']] = None) -> Tuple[Chem.Mol, List[int]]:
        """
        Generate conformers and cluster by RMSD threshold.

        Args:
            mol: Query molecule
            num_confs: Number of conformers to generate
            rmsd_threshold: RMSD threshold (Å) for clustering
            coordMap: Optional coordinate constraints

        Returns:
            mol: Molecule with conformers
            representative_cids: Cluster centroid IDs
        """
        return generate_conformers_and_cluster(mol, self.device, num_confs, rmsd_threshold, coordMap)

    def step2_find_mcs(self, ref_mol: Chem.Mol, query_mol: Chem.Mol,
                       return_all_positions: bool = False,
                       cross_match: bool = False,
                       min_fragment_size: Optional[int] = None,
                       max_fragments: int = 3) -> List[List[Tuple[int, int]]]:
        """
        Find MCS between reference and query with three modes:

        Mode 1 (Single): return_all_positions=False, cross_match=False
            Returns single best alignment (1:1) - fastest, default

        Mode 2 (Multi): return_all_positions=True, cross_match=False
            Returns all possible positions for symmetric reference (1:N)

        Mode 3 (Cross): cross_match=True
            Returns all cross-matching combinations for complex molecules (N:M)

        Args:
            ref_mol: Reference molecule
            query_mol: Query molecule
            return_all_positions: If True, return ALL possible MCS positions (Mode 2)
            cross_match: If True, use cross-matching for multiple fragments (Mode 3)
            min_fragment_size: Minimum atoms per fragment (only for cross_match)
            max_fragments: Maximum fragments to find (only for cross_match)

        Returns:
            If return_all_positions=False and cross_match=False:
                Single mapping [(ref_idx, query_idx), ...] (backward compatible)
            Otherwise:
                List of mappings [mapping1, mapping2, ...]
        """
        mappings = find_mcs_with_positions(ref_mol, query_mol,
                                          return_all=return_all_positions,
                                          cross_match=cross_match,
                                          min_atoms=3,
                                          min_fragment_size=min_fragment_size,
                                          max_fragments=max_fragments)

        if not mappings:
            raise ValueError("No MCS found between reference and query")

        if not return_all_positions and not cross_match:
            # Mode 1: Backward compatible - return first mapping directly
            return mappings[0]
        else:
            # Mode 2/3: Return all mappings
            return mappings

    def compute_vina_features(self, mol: Chem.Mol) -> dict:
        return compute_vina_features(mol, self.device)

    def step4_vina_scoring(self,
                           aligned_query_coords: torch.Tensor,
                           pocket_coords: torch.Tensor,
                           query_features: dict,
                           pocket_features: dict,
                           num_rotatable_bonds: int = None,
                           weight_preset: str = 'vina',
                           intramolecular_mask: torch.Tensor = None,
                           precomputed_matrices: dict = None) -> torch.Tensor:
        return vina_scoring(aligned_query_coords, pocket_coords, query_features, pocket_features, num_rotatable_bonds, weight_preset, intramolecular_mask=intramolecular_mask, precomputed_matrices=precomputed_matrices)

    def step5_final_selection(self,
                              mol: Chem.Mol,
                              representative_cids: List[int],
                              aligned_coords: torch.Tensor,
                              scores: torch.Tensor,
                              initial_scores: torch.Tensor = None,
                              position_labels: List[int] = None,
                              output_path: str = "output.sdf"):
        """Save all poses sorted by Vina score (best energy first)."""
        return final_selection(mol, representative_cids, aligned_coords, scores, initial_scores, position_labels, output_path)

    def step6_refine_pose(self,
                          mol: Chem.Mol,
                          ref_indices: List[int],
                          init_coords: torch.Tensor,
                          pocket_coords: torch.Tensor,
                          query_features: dict,
                          pocket_features: dict,
                          num_steps: int = 100,
                          lr: float = 0.05,
                          freeze_mcs: bool = True,
                          num_rotatable_bonds: int = None,
                          weight_preset: str = 'vina',
                          batch_size: int = 8,
                          optimizer: str = 'adam',
                          early_stopping: bool = True,
                          patience: int = 30,
                          min_delta: float = 1e-5) -> torch.Tensor:
        """
        Runs gradient-based torsion optimization to minimize Vina Score.

        Automatically handles both single pose and batched optimization.

        Args:
            init_coords: [N_atoms, 3] or [N_poses, N_atoms, 3]
            batch_size: Batch size for multi-pose optimization
            optimizer: Optimizer type ('adam', 'adamw', 'lbfgs')
            early_stopping: Enable early stopping (default: True)
            patience: Steps without improvement before stopping (default: 30)
            min_delta: Minimum improvement to reset patience (default: 1e-5)

        Returns:
            [N_atoms, 3] or [N_poses, N_atoms, 3] depending on input
        """
        return optimize_torsions_vina(mol, ref_indices, init_coords, pocket_coords,
                                      query_features, pocket_features, self.device,
                                      num_steps, lr, freeze_mcs, num_rotatable_bonds,
                                      weight_preset, batch_size, optimizer,
                                      early_stopping, patience, min_delta)
