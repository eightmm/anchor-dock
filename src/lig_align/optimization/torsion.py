"""Backward-compatible LigAlign optimizer backed by AnchorDock core."""

from anchor_dock.core.optimization import optimize_torsions_vina as _optimize


def optimize_torsions_vina(
    mol, ref_indices, init_coords, pocket_coords, query_features, pocket_features, device,
    num_steps=100, lr=0.1, freeze_mcs=True, num_rotatable_bonds=None,
    weight_preset="vina", batch_size=8, optimizer="adam", early_stopping=True,
    patience=30, min_delta=1e-5, return_stats=False,
):
    return _optimize(
        mol, ref_indices, init_coords, pocket_coords, query_features, pocket_features, device,
        num_steps, lr, freeze_mcs, num_rotatable_bonds, weight_preset, batch_size,
        optimizer, early_stopping, patience, min_delta, return_stats=return_stats,
    )


__all__ = ["optimize_torsions_vina"]
