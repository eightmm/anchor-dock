"""The optimizer must follow its stated objective and preserve coordinate origin symmetry."""
from __future__ import annotations

import pytest
import torch

from anchor_dock.core.optimization import optimize_pose_module


class CartesianPose(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.xyz = torch.nn.Parameter(torch.tensor([[[x, 0., 0.]]], dtype=torch.float64))

    def forward(self):
        return self.xyz


def test_lbfgs_closure_gradient_matches_its_unclipped_objective(monkeypatch):
    original = torch.optim.LBFGS
    checks = []

    class CheckedLBFGS(original):
        def step(self, closure):
            def checked_closure():
                loss = closure()
                parameter = self.param_groups[0]['params'][0]
                expected = torch.zeros_like(parameter)
                expected[:, 0, 0] = 200. * (parameter[:, 0, 0] - 5.)
                torch.testing.assert_close(parameter.grad, expected, rtol=1e-12, atol=1e-12)
                checks.append(float(parameter.grad.norm()))
                return loss
            return super().step(checked_closure)

    monkeypatch.setattr(torch.optim, 'LBFGS', CheckedLBFGS)
    model = CartesianPose(0.)
    output, _ = optimize_pose_module(model, lambda x: 100. * (x[:, 0, 0] - 5.).square(),
                                     num_steps=3, optimizer='lbfgs', gradient_clip=1., early_stopping=False)
    assert max(checks) > 1.
    assert abs(float(output.detach()[0, 0, 0]) - 5.) < 5.


@pytest.mark.parametrize('optimizer', ['adam', 'adamw'])
def test_pose_optimization_has_no_hidden_origin_dependent_decay(optimizer):
    results = []
    for origin in (0., 100.):
        model = CartesianPose(origin + 4.)
        output, _ = optimize_pose_module(model, lambda x: (x[:, 0, 0] - origin).square(),
                                         num_steps=30, learning_rate=.1, optimizer=optimizer,
                                         gradient_clip=None, early_stopping=False)
        results.append(output.detach()[0, 0, 0] - origin)
    torch.testing.assert_close(results[0], results[1], rtol=1e-12, atol=1e-12)


def test_changed_typing_contract_invalidates_batch_resume(monkeypatch):
    from anchor_dock import batch
    job = batch.DockingJob.interaction('CCO', protein_pdb='protein.pdb',
        receptor_residue='CYS145:A', receptor_atom='SG', ligand_smarts='[#6:1]',
        target_distance=3., distance_tolerance=.5)
    runtime = batch._runtime_identity({'device':'cpu'})
    before = batch._job_signature(job, {}, runtime)
    monkeypatch.setattr(batch, 'ATOM_TYPING_VERSION', 'different-typing-contract')
    after = batch._job_signature(job, {}, runtime)
    assert before != after


def test_changed_optimizer_contract_invalidates_batch_resume(monkeypatch):
    from anchor_dock import batch
    job = batch.DockingJob.interaction('CCO', protein_pdb='protein.pdb',
        receptor_residue='CYS145:A', receptor_atom='SG', ligand_smarts='[#6:1]',
        target_distance=3., distance_tolerance=.5)
    runtime = batch._runtime_identity({'device':'cpu'})
    before = batch._job_signature(job, {}, runtime)
    monkeypatch.setattr(batch, 'OPTIMIZATION_CONTRACT', 'different-jacobian-contract')
    assert before != batch._job_signature(job, {}, runtime)
