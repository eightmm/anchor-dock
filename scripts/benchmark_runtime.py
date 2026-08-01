import argparse
import statistics as stats
import time

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.ML.Cluster import Butina

from lig_align.aligner import LigandAligner
from lig_align.alignment import LigandKinematics
from lig_align.io import load_pocket_bundle, process_query_ligand
from lig_align.molecular.relax import relax_pose_with_fixed_core
from lig_align.optimization import optimize_torsions_vina
from lig_align.scoring import compute_intramolecular_mask


def generate_seeded_representatives(mol, device, num_confs, rmsd_threshold, coord_map, seed):
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = -1.0
    params.randomSeed = seed
    params.numThreads = 0

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if len(cids) == 0:
        raise RuntimeError("Conformer generation failed")

    if coord_map is not None:
        for cid in cids:
            conf = mol.GetConformer(cid)
            for atom_idx, pos in coord_map.items():
                conf.SetAtomPosition(atom_idx, pos)
            try:
                AllChem.MMFFOptimizeMolecule(
                    mol,
                    confId=cid,
                    maxIters=200,
                    nonBondedThresh=100.0,
                    ignoreInterfragInteractions=False,
                )
            except Exception:
                pass

    mol = Chem.RemoveHs(mol)
    n_confs = len(cids)
    n_atoms = mol.GetNumAtoms()
    dists = []

    if n_confs > 1:
        coords = torch.zeros((n_confs, n_atoms, 3), device=device)
        for i, cid in enumerate(cids):
            coords[i] = torch.tensor(
                mol.GetConformer(cid).GetPositions(),
                dtype=torch.float32,
                device=device,
            )

        coords_centered = coords - coords.mean(dim=1, keepdim=True)
        h = torch.einsum("iac,jad->ijcd", coords_centered, coords_centered)
        h_flat = h.reshape(n_confs * n_confs, 3, 3)
        u, s, vh = torch.linalg.svd(h_flat)
        v = vh.transpose(1, 2)
        ut = u.transpose(1, 2)
        r = torch.bmm(v, ut)
        det = torch.linalg.det(r)

        s_sum = s.sum(dim=1)
        reflection_mask = det < 0
        s_sum[reflection_mask] -= 2 * s[reflection_mask, 2]

        norms = (coords_centered ** 2).sum(dim=(1, 2))
        norms_i = norms.unsqueeze(1).expand(n_confs, n_confs).flatten()
        norms_j = norms.unsqueeze(0).expand(n_confs, n_confs).flatten()

        dist_sq = torch.clamp(norms_i + norms_j - 2 * s_sum, min=0.0)
        rmsd_matrix = torch.sqrt(dist_sq / n_atoms).view(n_confs, n_confs)
        triu = torch.triu_indices(n_confs, n_confs, offset=1, device=device)
        dists = rmsd_matrix[triu[1], triu[0]].cpu().tolist()

    clusters = Butina.ClusterData(dists, n_confs, rmsd_threshold, isDistData=True)
    return mol, [cluster[0] for cluster in clusters]


def prepare_case(args, seed):
    torch.manual_seed(seed)
    device = torch.device(args.device)
    aligner = LigandAligner(device=device)
    ref_mol = Chem.SDMolSupplier(args.ref_ligand)[0]
    pocket_bundle = load_pocket_bundle(args.protein, device, aligner.compute_vina_features)
    query_mol, _ = process_query_ligand(args.query)
    mapping = aligner.step2_find_mcs(ref_mol, query_mol)

    ref_conf = ref_mol.GetConformer()
    coord_map = {q: ref_conf.GetAtomPosition(r) for r, q in mapping}
    query_indices = [q for _, q in mapping]
    query_mol, rep_cids = generate_seeded_representatives(
        query_mol,
        device,
        args.num_confs,
        args.rmsd_threshold,
        coord_map,
        seed,
    )

    coords = []
    for cid in rep_cids:
        conf = query_mol.GetConformer(cid)
        for r, q in mapping:
            conf.SetAtomPosition(q, ref_conf.GetAtomPosition(r))
        relax_pose_with_fixed_core(query_mol, cid, set(query_indices), max_iters=500)
        coords.append(torch.tensor(conf.GetPositions(), dtype=torch.float32, device=device))

    init_coords = torch.stack(coords)
    intra_mask = compute_intramolecular_mask(query_mol, device)
    torsions = LigandKinematics(
        query_mol,
        query_indices,
        init_coords[0],
        device,
        freeze_mcs=True,
    ).num_torsions

    return {
        "query_mol": query_mol,
        "query_indices": query_indices,
        "init_coords": init_coords,
        "pocket_coords": pocket_bundle.coords,
        "query_features": aligner.compute_vina_features(query_mol),
        "pocket_features": pocket_bundle.features,
        "intra_mask": intra_mask,
        "torsions": torsions,
        "n_poses": len(rep_cids),
    }


def run_batch_probe(prepared, steps, batch_size, early_stopping, seed):
    torch.manual_seed(seed)
    device = prepared["init_coords"].device
    track_cuda_mem = device.type == "cuda" and torch.cuda.is_available()
    if track_cuda_mem:
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    _, stats = optimize_torsions_vina(
        mol=prepared["query_mol"],
        ref_indices=prepared["query_indices"],
        init_coords=prepared["init_coords"],
        pocket_coords=prepared["pocket_coords"],
        query_features=prepared["query_features"],
        pocket_features=prepared["pocket_features"],
        device=device,
        num_steps=steps,
        lr=0.05,
        freeze_mcs=True,
        num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(prepared["query_mol"]),
        weight_preset="vina",
        batch_size=batch_size,
        optimizer="adam",
        early_stopping=early_stopping,
        patience=30,
        min_delta=1e-5,
        return_stats=True,
    )
    total = time.perf_counter() - t0
    peak_allocated_mb = None
    peak_reserved_mb = None
    if track_cuda_mem:
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {
        "total": total,
        "avg_steps": stats["avg_steps"],
        "min_steps": stats["min_steps"],
        "max_steps": stats["max_steps"],
        "n_poses": stats["n_poses"],
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LigAlign runtime and batch behavior.")
    parser.add_argument("--protein", default="examples/10gs/10gs_pocket.pdb")
    parser.add_argument("--ref_ligand", default="examples/10gs/10gs_ligand.sdf")
    parser.add_argument(
        "--query",
        default="COc1ccc2c(c1)c(C(=O)NCc1ccccc1)c(CC(=O)O)n2C",
    )
    parser.add_argument("--num_confs", type=int, default=50)
    parser.add_argument("--rmsd_threshold", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.warning")

    configs = [(batch_size, early) for batch_size in args.batch_sizes for early in (False, True)]
    aggregate = {
        cfg: {
            "total": [],
            "avg_steps": [],
            "n_poses": [],
            "peak_allocated_mb": [],
            "peak_reserved_mb": [],
        }
        for cfg in configs
    }
    seed_meta = []

    for seed in args.seeds:
        prepared = prepare_case(args, seed)
        seed_meta.append(
            {
                "seed": seed,
                "poses": prepared["n_poses"],
                "torsions": prepared["torsions"],
                "heavy_atoms": prepared["query_mol"].GetNumHeavyAtoms(),
            }
        )
        for cfg in configs:
            result = run_batch_probe(prepared, args.steps, cfg[0], cfg[1], seed)
            aggregate[cfg]["total"].append(result["total"])
            aggregate[cfg]["avg_steps"].append(result["avg_steps"])
            aggregate[cfg]["n_poses"].append(result["n_poses"])
            if result["peak_allocated_mb"] is not None:
                aggregate[cfg]["peak_allocated_mb"].append(result["peak_allocated_mb"])
            if result["peak_reserved_mb"] is not None:
                aggregate[cfg]["peak_reserved_mb"].append(result["peak_reserved_mb"])

    print("Benchmark case")
    for item in seed_meta:
        print(
            f"- seed {item['seed']}: poses={item['poses']}, torsions={item['torsions']}, "
            f"heavy_atoms={item['heavy_atoms']}"
        )

    print("\n| Batch size | Early stopping | Total time mean | Total time std | Avg steps mean | Avg steps std | Poses mean | Time per pose | Peak alloc | Peak reserved |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cfg in configs:
        total = aggregate[cfg]["total"]
        avg_steps = aggregate[cfg]["avg_steps"]
        poses = aggregate[cfg]["n_poses"]
        if aggregate[cfg]["peak_allocated_mb"]:
            peak_alloc = f"{stats.mean(aggregate[cfg]['peak_allocated_mb']):.1f} MB"
            peak_res = f"{stats.mean(aggregate[cfg]['peak_reserved_mb']):.1f} MB"
        else:
            peak_alloc = "n/a"
            peak_res = "n/a"
        print(
            f"| {cfg[0]} | {'on' if cfg[1] else 'off'} | "
            f"{stats.mean(total):.3f} s | {stats.pstdev(total):.3f} s | "
            f"{stats.mean(avg_steps):.1f} | {stats.pstdev(avg_steps):.1f} | "
            f"{stats.mean(poses):.1f} | {1000 * stats.mean(total) / stats.mean(poses):.2f} ms | "
            f"{peak_alloc} | {peak_res} |"
        )


if __name__ == "__main__":
    main()
