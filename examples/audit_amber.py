"""Compare the optional AMBER force bridge with an independent OpenMM Context.

Requires openmm. Supply a combined complex prmtop/inpcrd and explicit movable
particle indices, for example --indices '[6,7,8,9,10,11,12,13,14,15]'.
This checks a potential/force contract, not docking or binding-affinity accuracy.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch


def audit(prmtop: Path, inpcrd: Path, indices: list[int], model: str | None):
    import openmm
    from openmm import app, unit

    from anchor_dock.core.amber import AMBER_SCORING_CONTRACT, OpenMMScorer

    with OpenMMScorer.from_amber(prmtop, inpcrd, indices, implicit_solvent=model) as scorer:
        initial = scorer.ligand_coordinates(dtype=torch.float64)
        offset = torch.linspace(-.02, .02, initial.numel()).reshape_as(initial)
        xyz = torch.stack((initial, initial + offset)).requires_grad_()
        values = scorer.energy(xyz)
        derivatives, = torch.autograd.grad(values.sum(), xyz)
        reference = scorer.reference_energy
        relative = scorer.relative_energy(xyz)
        torch.testing.assert_close(relative, values - reference, rtol=0, atol=0)
        coordinates = app.AmberInpcrdFile(str(inpcrd))
        topology = app.AmberPrmtopFile(str(prmtop), periodicBoxVectors=coordinates.boxVectors)
        system = topology.createSystem(nonbondedMethod=app.NoCutoff, constraints=None,
                                       rigidWater=False, removeCMMotion=False,
                                       implicitSolvent=getattr(app, model) if model else None,
                                       soluteDielectric=1., solventDielectric=78.5)
        integrator = openmm.VerletIntegrator(.001)
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
        energies, gradients = [], []
        positions = np.asarray(coordinates.positions.value_in_unit(unit.angstrom))
        for pose in xyz.detach().numpy():
            candidate = positions.copy()
            candidate[indices] = pose
            context.setPositions(candidate * unit.angstrom)
            state = context.getState(getEnergy=True, getForces=True)
            energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / 4.184)
            gradients.append(-state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)[indices] / 41.84)
        del context, integrator
        gradcheck = torch.autograd.gradcheck(scorer.relative_energy, (xyz,), eps=1e-5, atol=2e-6, rtol=2e-5)
        return {
            "contract": AMBER_SCORING_CONTRACT, "python": platform.python_version(),
            "torch": torch.__version__, "openmm": openmm.__version__,
            "system_particles": system.getNumParticles(), "movable_particle_indices": indices,
            "implicit_solvent": model, "poses": len(values), "gradcheck": gradcheck,
            "energy_kcal_mol": values.detach().tolist(), "reference_energy_kcal_mol": reference,
            "relative_search_energy_kcal_mol": relative.detach().tolist(),
            "max_energy_error_kcal_mol": float(np.max(np.abs(values.detach().numpy() - energies))),
            "max_gradient_error_kcal_mol_angstrom": float(np.max(np.abs(derivatives.numpy() - gradients))),
            "metadata": scorer.metadata,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", required=True, type=Path)
    parser.add_argument("--inpcrd", required=True, type=Path)
    parser.add_argument("--indices", required=True, help="JSON array of explicit all-atom ligand particle indices")
    parser.add_argument("--implicit-solvent", choices=("none", "HCT", "OBC1", "OBC2", "GBn", "GBn2"), default="none")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    indices = json.loads(options.indices)
    model = None if options.implicit_solvent == "none" else options.implicit_solvent
    result = audit(options.prmtop, options.inpcrd, indices, model)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if options.output:
        options.output.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
