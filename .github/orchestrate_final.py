from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path.cwd()
REPOSITORY = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["GH_TOKEN"]
BRANCH = "agent/final-engine-cleanup"
PR_NUMBER = 4


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, cwd=ROOT, check=check, text=True)


def output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def api(method: str, path: str, payload: dict[str, object] | None = None) -> tuple[int, object | None]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "anchor-dock-finalizer",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        body = json.loads(raw) if raw else None
        return exc.code, body


def remove_paths(paths: list[str]) -> None:
    for relative in paths:
        path = ROOT / relative
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def commit_if_needed(message: str, refspec: str) -> None:
    run("git", "add", "-A")
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=ROOT,
        check=False,
    ).returncode != 0
    if changed:
        run("git", "commit", "-m", message)
        run("git", "push", "origin", refspec)


def verify_wheel() -> None:
    import zipfile

    wheel = next((ROOT / "dist").glob("anchor_dock-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert any(name.startswith("anchor_dock/") for name in names)
    assert not any(name.startswith("lig_align/") for name in names)
    assert not any(name.startswith("cov_vina/") for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def freeze_standard_ci() -> None:
    path = ROOT / ".github/workflows/anchor-dock-ci.yml"
    text = path.read_text()
    old = (
        "      - name: Resolve environment\n"
        "        run: |\n"
        "          uv lock\n"
        "          uv sync --frozen --group dev\n"
    )
    new = (
        "      - name: Check lock file\n"
        "        run: uv lock --check\n"
        "      - name: Sync frozen environment\n"
        "        run: uv sync --frozen --group dev\n"
    )
    if old in text:
        path.write_text(text.replace(old, new))


def validate_and_merge() -> None:
    status, pr = api("GET", f"/pulls/{PR_NUMBER}")
    if status != 200 or not isinstance(pr, dict):
        raise RuntimeError(f"failed to read PR: status={status} body={pr}")
    if bool(pr.get("merged")):
        print("PR is already merged; continuing with cleanup.")
        return

    run("git", "fetch", "origin", BRANCH)
    run("git", "checkout", "-B", BRANCH, f"origin/{BRANCH}")
    remove_paths(
        [
            ".validation-success.txt",
            ".validation-error.txt",
            ".github/workflows/validate-final.yml",
            ".github/workflows/diagnose-final.yml",
            ".github/workflows/finalize-integration.yml",
        ]
    )

    run("uv", "lock")
    run("uv", "sync", "--frozen", "--group", "dev")
    run("uv", "run", "ruff", "check", "--fix", "src", "tests", "examples")
    run("uv", "run", "ruff", "format", "src", "tests", "examples")
    run("uv", "run", "ruff", "check", "src", "tests", "examples")
    run("uv", "run", "pytest", "-q")
    run("uv", "build", "--wheel")
    run("uv", "run", "anchor-dock", "--help")
    verify_wheel()
    freeze_standard_ci()

    remove_paths(["dist", "build", ".venv"])
    for path in ROOT.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)
    for path in ROOT.rglob("*.pyc"):
        path.unlink(missing_ok=True)

    commit_if_needed(
        "chore: freeze validated AnchorDock 0.3 environment",
        f"HEAD:{BRANCH}",
    )
    head_sha = output("git", "rev-parse", "HEAD")
    merge_status, merge_body = api(
        "PUT",
        f"/pulls/{PR_NUMBER}/merge",
        {
            "merge_method": "squash",
            "sha": head_sha,
            "commit_title": "feat: finalize AnchorDock 0.3 engine and batch API (#4)",
        },
    )
    if not 200 <= merge_status < 300:
        raise RuntimeError(f"merge failed: status={merge_status} body={merge_body}")
    time.sleep(5)


def clean_main_and_branches() -> None:
    run("git", "fetch", "origin", "main")
    run("git", "checkout", "-B", "main", "origin/main")
    remove_paths(
        [
            ".github/orchestrate_final.py",
            ".github/workflows/orchestrate-final.yml",
            ".github/workflows/orchestrate-final-v2.yml",
            ".github/workflows/orchestrate-final-v3.yml",
            ".github/workflows/finalize-main.yml",
            ".github/workflows/validate-final.yml",
            ".github/workflows/diagnose-final.yml",
            ".github/workflows/finalize-integration.yml",
            ".validation-success.txt",
            ".validation-error.txt",
        ]
    )
    commit_if_needed(
        "chore: remove completed integration workflows",
        "HEAD:main",
    )

    encoded = urllib.parse.quote(BRANCH, safe="")
    delete_status, delete_body = api("DELETE", f"/git/refs/heads/{encoded}")
    if delete_status not in {204, 404, 422}:
        raise RuntimeError(f"branch deletion failed: status={delete_status} body={delete_body}")

    status, branches = api("GET", "/branches?per_page=100")
    if status != 200 or not isinstance(branches, list):
        raise RuntimeError(f"failed to list branches: status={status} body={branches}")
    remaining = [branch["name"] for branch in branches]
    if remaining != ["main"]:
        raise RuntimeError(f"unexpected remaining branches: {remaining}")
    print("Final state verified: only main remains.")


def main() -> None:
    run("git", "config", "user.name", "github-actions[bot]")
    run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    validate_and_merge()
    clean_main_and_branches()


if __name__ == "__main__":
    main()
