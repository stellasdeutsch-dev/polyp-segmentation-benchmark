"""Publish the site to GitHub Pages as a snapshot on the gh-pages branch.

    make deploy

Copies the publishable files into a temporary folder, makes one snapshot commit there, and
force-pushes it to the gh-pages branch of the GitHub repository. The project's own history
(main) is not touched: this folder needs no git history at all, and no global git setting is
changed. Authentication goes through the GitHub CLI for this one push.

Data, checkpoints, runs and third-party code are never published (see .gitignore).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INCLUDE = [
    "index.html", "README.md", "Makefile", "pyproject.toml", "requirements-lock.txt", ".gitignore",
    "site", "research", "reports", "src", "scripts", "tests", "configs",
    "data/splits", "data/checksums.json",
]  # fmt: skip
SKIP_NAMES = {"__pycache__", ".DS_Store", ".pytest_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pyc", ".pth", ".pt", ".ckpt"}


def run(*cmd: str, cwd: Path | None = None) -> str:
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{out.stderr.strip()}")
    return out.stdout.strip()


def copy_tree(src: Path, dst: Path) -> int:
    n = 0
    for path in src.rglob("*") if src.is_dir() else [src]:
        rel = path.relative_to(REPO)
        if any(part in SKIP_NAMES for part in rel.parts) or path.suffix in SKIP_SUFFIXES or path.is_dir():
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="stellasdeutsch-dev/polyp-segmentation-benchmark")
    args = parser.parse_args()

    user = json.loads(run("gh", "api", "user"))
    name, email = user["login"], f"{user['id']}+{user['login']}@users.noreply.github.com"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        files = sum(copy_tree(REPO / item, out) for item in INCLUDE if (REPO / item).exists())
        (out / ".nojekyll").write_text("")  # serve files as they are; no Jekyll processing
        run("git", "init", "-q", "-b", "gh-pages", cwd=out)
        run("git", "add", "-A", cwd=out)
        run("git", "-c", f"user.name={name}", "-c", f"user.email={email}",
            "commit", "-q", "-m", "Publish site snapshot", cwd=out)  # fmt: skip
        run("git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential",
            "push", "-q", "--force", f"https://github.com/{args.repo}.git", "gh-pages", cwd=out)  # fmt: skip
    owner, repo = args.repo.split("/")
    print(f"published {files} files to gh-pages -> https://{owner}.github.io/{repo}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
