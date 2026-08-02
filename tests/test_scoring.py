from __future__ import annotations

import functools
import operator
import subprocess
import sys

import pytest
import torch
from rdkit import Chem

from anchor_dock.core.features import compute_atom_features
from anchor_dock.core.scoring import (
    VINA_CONFIG,
    VINARDO_CONFIG,
    PairwiseScorer,
    RawScoreComponents,
    pair_terms,
)


def test_official_default_weights_and_cutoffs() -> None:
    assert VINA_CONFIG.cutoff == 8.0
    assert VINA_CONFIG.weights == {
        "gauss1": -0.035579,
        "gauss2": -0.005156,
        "repulsion": 0.840245,
        "hydrophobic": -0.035069,
        "hbond": -0.587439,
    }
    assert VINARDO_CONFIG.cutoff == 8.0
    assert VINARDO_CONFIG.weights == {
        "gauss1": -0.045,
        "gauss2": 0.0,
        "repulsion": 0.8,
        "hydrophobic": -0.035,
        "hbond": -0.600,
    }


def test_vina_piecewise_terms_at_boundaries() -> None:
    surface = torch.tensor([-0.7, -0.35, 0.0, 0.5, 1.0, 1.5])
    terms = pair_terms(
        surface, torch.zeros_like(surface), torch.ones_like(surface), torch.ones_like(surface), VINA_CONFIG
    )
    assert torch.allclose(terms["hbond"], torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(terms["hydrophobic"], torch.tensor([1.0, 1.0, 1.0, 1.0, 0.5, 0.0]))


def test_actual_distance_cutoff_is_hard_zero() -> None:
    features = {
        "radius_vina": torch.tensor([1.9]),
        "radius_vinardo": torch.tensor([2.0]),
        "hydrophobic": torch.tensor([1.0]),
        "donor": torch.tensor([0.0]),
        "acceptor": torch.tensor([0.0]),
        "active": torch.tensor([True]),
    }
    scorer = PairwiseScorer(VINA_CONFIG).prepare(features, torch.tensor([[0.0, 0.0, 0.0]]), features)
    near = scorer.search_energy(torch.tensor([[7.999, 0.0, 0.0]]))
    far = scorer.search_energy(torch.tensor([[8.0, 0.0, 0.0]]))
    assert near.abs().item() > 0
    assert far.item() == 0.0


def test_inferred_xs_radii_distinguish_vina_and_vinardo() -> None:
    mol = Chem.MolFromSmiles("CO")
    features = compute_atom_features(mol, "cpu")
    assert features["xs_types"] == ("C_P", "O_DA") or features["xs_types"] == ("C_P", "O_A")
    assert torch.allclose(features["radius_vina"], torch.tensor([1.9, 1.7]))
    assert torch.allclose(features["radius_vinardo"], torch.tensor([2.0, 1.6]))


def test_report_uses_common_intramolecular_baseline_and_torsion_denominator() -> None:
    features = {
        "radius_vina": torch.tensor([1.9]),
        "radius_vinardo": torch.tensor([2.0]),
        "hydrophobic": torch.tensor([0.0]),
        "donor": torch.tensor([0.0]),
        "acceptor": torch.tensor([0.0]),
        "active": torch.tensor([True]),
    }
    scorer = PairwiseScorer(VINA_CONFIG).prepare(
        features,
        torch.tensor([[0.0, 0.0, 0.0]]),
        features,
        num_rotatable_bonds=5,
    )
    raw = RawScoreComponents(
        intermolecular=torch.tensor([-4.0, -3.0]),
        intramolecular=torch.tensor([1.0, 2.0]),
        search_energy=torch.tensor([-3.0, -1.0]),
    )
    reported = scorer.report(raw, intramolecular_reference=torch.tensor(1.0))
    expected = (raw.search_energy - 1.0) / (1.0 + 0.05846 * 5)
    assert torch.allclose(reported.score, expected)
    assert scorer.torsion_penalty_applied is True
    assert scorer.effective_rotatable_bonds == 5


def test_pairwise_score_is_differentiable() -> None:
    mol = Chem.MolFromSmiles("CCO")
    features = compute_atom_features(mol, "cpu")
    scorer = PairwiseScorer(VINA_CONFIG).prepare(
        features,
        torch.tensor([[3.0, 0.0, 0.0]]),
        compute_atom_features(Chem.MolFromSmiles("C"), "cpu"),
    )
    coords = torch.randn(2, mol.GetNumAtoms(), 3, requires_grad=True)
    scorer.search_energy(coords).sum().backward()
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()


def test_custom_neural_scorer_adapter() -> None:
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, ligand_coords, receptor_coords, ligand_features, receptor_features):
            del receptor_coords, ligand_features, receptor_features
            return ligand_coords.square().sum(dim=(1, 2))

    from anchor_dock.core.scoring import resolve_scorer

    ligand = Chem.MolFromSmiles("CC")
    features = compute_atom_features(ligand, "cpu")
    model = Model()
    assert model.training
    prepared = resolve_scorer(model).prepare(features, torch.zeros(1, 3), features)
    coords = torch.randn(3, 2, 3, requires_grad=True)
    values = prepared.search_energy(coords)
    assert values.shape == (3,)
    values.sum().backward()
    assert coords.grad is not None
    assert model.training
    assert prepared.fingerprint.startswith("sha256:")


def test_neural_scorer_fingerprint_includes_weights() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    first = nn.Linear(2, 1, bias=False)
    second = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        first.weight.fill_(1.0)
        second.weight.fill_(9.0)
    assert scorer_fingerprint(first) != scorer_fingerprint(second)


def test_neural_scorer_fingerprint_includes_plain_hyperparameters() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class Scaler(nn.Module):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.scale = scale

        def forward(self, values):
            return values * self.scale

    assert scorer_fingerprint(Scaler(1.0)) != scorer_fingerprint(Scaler(2.0))
    assert scorer_fingerprint(nn.Dropout(0.1)) != scorer_fingerprint(nn.Dropout(0.9))


def test_neural_scorer_fingerprint_includes_nested_unregistered_tensors() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class Scaler(nn.Module):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.config = {"values": [torch.tensor([scale]), {"offset": torch.tensor(0.5)}]}

        def forward(self, values):
            return values * self.config["values"][0] + self.config["values"][1]["offset"]

    first = Scaler(1.0)
    same = Scaler(1.0)
    changed = Scaler(9.0)
    assert scorer_fingerprint(first) == scorer_fingerprint(same)
    assert scorer_fingerprint(first) != scorer_fingerprint(changed)


def test_neural_scorer_fingerprint_includes_forward_implementation() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class Scaler(nn.Module):
        def forward(self, values):
            return values * 2.0

    model = Scaler()
    before = scorer_fingerprint(model)

    def replacement(self, values):
        return values + 2.0

    Scaler.forward = replacement
    assert scorer_fingerprint(model) != before


def test_neural_scorer_fingerprint_includes_nonpersistent_buffers() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class Scaler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("scale", torch.tensor(1.0), persistent=False)

        def forward(self, values):
            return values * self.scale

    model = Scaler()
    before = scorer_fingerprint(model)
    model.scale.fill_(9.0)
    assert scorer_fingerprint(model) != before


def test_neural_scorer_fingerprint_includes_helper_method_code() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class Scaler(nn.Module):
        def helper(self, values):
            return values * 2.0

        def forward(self, values):
            return self.helper(values)

    model = Scaler()
    before = scorer_fingerprint(model)

    def replacement(self, values):
        return values * 7.0

    Scaler.helper = replacement
    assert scorer_fingerprint(model) != before


def test_neural_scorer_fingerprint_includes_module_class_state() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class Scaler(nn.Module):
        SCALE = 2.0

        def forward(self, values):
            return values * self.SCALE

    model = Scaler()
    before = scorer_fingerprint(model)
    Scaler.SCALE = 7.0
    assert scorer_fingerprint(model) != before


def test_neural_scorer_fingerprint_includes_callable_object_state() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class CallableScale:
        def __init__(self, factor: float) -> None:
            self.factor = factor

        def __call__(self, values):
            return values * self.factor

    class Scorer(nn.Module):
        def __init__(self, factor: float) -> None:
            super().__init__()
            self.transform = CallableScale(factor)

        def forward(self, values):
            return self.transform(values)

    assert scorer_fingerprint(Scorer(2.0)) != scorer_fingerprint(Scorer(7.0))


def test_neural_scorer_fingerprint_includes_callable_object_helper_code() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class CallableTransform:
        def helper(self, values):
            return values * 2.0

        def __call__(self, values):
            return self.helper(values)

    class Scorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transform = CallableTransform()

        def forward(self, values):
            return self.transform(values)

    model = Scorer()
    before = scorer_fingerprint(model)

    def replacement(self, values):
        return values * 7.0

    CallableTransform.helper = replacement
    assert scorer_fingerprint(model) != before


def test_neural_scorer_fingerprint_includes_callable_object_class_state() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class CallableTransform:
        SCALE = 2.0

        def __call__(self, values):
            return values * self.SCALE

    class Scorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transform = CallableTransform()

        def forward(self, values):
            return self.transform(values)

    model = Scorer()
    before = scorer_fingerprint(model)
    CallableTransform.SCALE = 7.0
    assert scorer_fingerprint(model) != before


def test_neural_scorer_fingerprint_includes_partial_arguments() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class Scorer(nn.Module):
        def __init__(self, factor: float) -> None:
            super().__init__()
            self.transform = functools.partial(torch.mul, other=factor)

        def forward(self, values):
            return self.transform(values)

    assert scorer_fingerprint(Scorer(2.0)) != scorer_fingerprint(Scorer(7.0))


def test_neural_scorer_fingerprint_includes_bound_method_instance_state() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class Scale:
        def __init__(self, factor: float) -> None:
            self.factor = factor

        def apply(self, values):
            return values * self.factor

    class Scorer(nn.Module):
        def __init__(self, factor: float) -> None:
            super().__init__()
            self.transform = Scale(factor).apply

        def forward(self, values):
            return self.transform(values)

    assert scorer_fingerprint(Scorer(2.0)) != scorer_fingerprint(Scorer(7.0))


def test_unregistered_module_bound_method_requires_explicit_fingerprint() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import NeuralScorerAdapter, scorer_fingerprint

    class Scale(nn.Module):
        def __init__(self, factor: float) -> None:
            super().__init__()
            self.factor = factor

        def apply(self, values):
            return values * self.factor

    class Scorer(nn.Module):
        def __init__(self, factor: float) -> None:
            super().__init__()
            self.transform = Scale(factor).apply

        def forward(self, values):
            return self.transform(values)

    model = Scorer(2.0)
    with pytest.raises(ValueError, match="unregistered nn.Module"):
        scorer_fingerprint(model)
    assert NeuralScorerAdapter(model, fingerprint="external-module-v1").fingerprint == "external-module-v1"


def test_neural_scorer_callable_fingerprint_is_cross_process_stable() -> None:
    script = """
import functools
import torch
import torch.nn as nn
from anchor_dock.core.scoring import scorer_fingerprint

class CallableScale:
    __slots__ = ("factor",)
    def __init__(self, factor): self.factor = factor
    def __call__(self, values): return values * self.factor

class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = CallableScale(3.0)
        self.second = functools.partial(torch.mul, other=5.0)
        self.third = CallableScale(7.0).__call__
    def forward(self, values): return self.third(self.second(self.first(values)))

print(scorer_fingerprint(Scorer()))
"""
    first = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
    second = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
    assert first == second


def test_neural_scorer_opaque_callable_state_requires_explicit_fingerprint() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import NeuralScorerAdapter, scorer_fingerprint

    class OpaqueCallable:
        def __init__(self) -> None:
            self.resource = object()

        def __call__(self, values):
            return values

    class Scorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transform = OpaqueCallable()

        def forward(self, values):
            return self.transform(values)

    model = Scorer()
    with pytest.raises(ValueError, match="NeuralScorerAdapter"):
        scorer_fingerprint(model)
    assert NeuralScorerAdapter(model, fingerprint="opaque-v1").fingerprint == "opaque-v1"


def test_neural_scorer_cyclic_callable_state_requires_explicit_fingerprint() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class CyclicCallable:
        def __init__(self) -> None:
            self.state: list[object] = []
            self.state.append(self.state)

        def __call__(self, values):
            return values

    class Scorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transform = CyclicCallable()

        def forward(self, values):
            return self.transform(values)

    with pytest.raises(ValueError, match="NeuralScorerAdapter"):
        scorer_fingerprint(Scorer())


def test_neural_scorer_uninspectable_callable_requires_explicit_fingerprint() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class Scorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transform = operator.itemgetter(0)

        def forward(self, values):
            return self.transform((values,))

    with pytest.raises(ValueError, match="NeuralScorerAdapter"):
        scorer_fingerprint(Scorer())


def test_hooked_neural_scorer_requires_explicit_fingerprint() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import NeuralScorerAdapter, scorer_fingerprint

    class Scorer(nn.Module):
        def forward(self, values):
            return values

    model = Scorer()
    model.register_forward_hook(lambda module, args, output: output * 7.0)
    with pytest.raises(ValueError, match="NeuralScorerAdapter"):
        scorer_fingerprint(model)
    assert NeuralScorerAdapter(model, fingerprint="hooked-scorer-v1").fingerprint == "hooked-scorer-v1"


def test_neural_scorer_fingerprint_tracks_referenced_global_code_and_state() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    namespace = {"__name__": "anchor_dock_fingerprint_test", "nn": nn}
    exec(
        "SCALE = 2.0\n"
        "def helper(values):\n"
        "    return values * SCALE\n"
        "class Scorer(nn.Module):\n"
        "    def forward(self, values):\n"
        "        return helper(values)\n",
        namespace,
    )
    model = namespace["Scorer"]()
    initial = scorer_fingerprint(model)
    namespace["SCALE"] = 7.0
    state_changed = scorer_fingerprint(model)
    assert state_changed != initial
    exec("def helper(values):\n    return values + SCALE\n", namespace)
    assert scorer_fingerprint(model) != state_changed


def test_neural_scorer_fingerprint_tracks_referenced_global_type_state() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    namespace = {"__name__": "anchor_dock_global_type_test", "nn": nn}
    exec(
        "class Transform:\n"
        "    SCALE = 2.0\n"
        "    def __call__(self, values):\n"
        "        return values * self.SCALE\n"
        "class Scorer(nn.Module):\n"
        "    def forward(self, values):\n"
        "        return Transform()(values)\n",
        namespace,
    )
    model = namespace["Scorer"]()
    initial = scorer_fingerprint(model)
    namespace["Transform"].SCALE = 7.0
    assert scorer_fingerprint(model) != initial


def test_neural_adapter_fingerprint_tracks_weight_mutation() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import NeuralScorerAdapter

    model = nn.Linear(2, 1, bias=False)
    adapter = NeuralScorerAdapter(model)
    before = adapter.fingerprint
    with torch.no_grad():
        model.weight.add_(1.0)
    assert adapter.fingerprint != before


def test_neural_scorer_fingerprint_supports_scalar_state() -> None:
    import torch.nn as nn

    from anchor_dock.core.scoring import scorer_fingerprint

    class ScalarState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))

    assert scorer_fingerprint(ScalarState()).startswith("sha256:")


def test_vina_rejects_untyped_elements() -> None:
    ligand = Chem.MolFromSmiles("CB(O)O")
    features = compute_atom_features(ligand, "cpu")
    with pytest.raises(ValueError, match="validated XS-like"):
        PairwiseScorer(VINA_CONFIG).prepare(
            features,
            torch.zeros(1, 3),
            compute_atom_features(Chem.MolFromSmiles("C"), "cpu"),
        )


def test_pdb_water_oxygen_is_donor_and_acceptor() -> None:
    block = "HETATM    1  O   HOH A 101       0.000   0.000   0.000  1.00 20.00           O  \nEND\n"
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=True)
    features = compute_atom_features(mol, "cpu")
    assert features["donor"].tolist() == [1.0]
    assert features["acceptor"].tolist() == [1.0]
    assert features["xs_types"] == ("O_DA",)
