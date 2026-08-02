from anchor_dock.cli import build_parser


def test_cli_exposes_all_native_modes() -> None:
    parser = build_parser()
    assert parser.parse_args(["reference", "-p", "p.pdb", "-r", "r.sdf", "-q", "CCO"]).command == "reference"
    assert parser.parse_args(["covalent", "-p", "p.pdb", "-q", "C=CC=O"]).command == "covalent"
    assert parser.parse_args(["free", "-p", "p.pdb", "-q", "CCO"]).command == "free"
    assert parser.parse_args(["batch", "jobs.jsonl"]).command == "batch"
