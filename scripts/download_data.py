"""Download and unpack the PraNet polyp benchmark into data/.

    make data

The archives are the ones linked from the PraNet README (decision D1):
  TrainDataset: 900 Kvasir-SEG + 550 CVC-ClinicDB images
  TestDataset:  Kvasir 100, CVC-ClinicDB 62, CVC-ColonDB 380, ETIS-LaribPolypDB 196, CVC-300 60

Archives are treated as untrusted input: they are extracted with a path check
(nothing may land outside data/), and nothing inside them is executed.
After the first download the SHA-256 of each archive is written to
data/checksums.json; later runs verify against it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
RAW = DATA / "raw"
CHECKSUMS = DATA / "checksums.json"

# Google Drive file ids from the PraNet README (commit 95f12bd), "Downloading necessary data".
ARCHIVES = {
    "TrainDataset": {"id": "1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb", "readme_size": "399.5MB"},
    "TestDataset": {"id": "1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao", "readme_size": "327.2MB"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive: Path, target: Path) -> int:
    """Extract a zip, refusing any member that would land outside `target`."""
    target = target.resolve()
    n = 0
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename
            base = Path(name).name
            if name.startswith("__MACOSX/") or base.startswith("._") or base == ".DS_Store":
                continue  # macOS metadata the authors' archives contain, not data
            dest = (target / name).resolve()
            if not dest.is_relative_to(target):
                raise RuntimeError(f"refusing to extract {name!r}: it points outside {target}")
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, dest.open("wb") as out:
                out.write(src.read())
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download again even if the archive exists")
    args = parser.parse_args()

    import gdown

    RAW.mkdir(parents=True, exist_ok=True)
    known = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}

    for name, info in ARCHIVES.items():
        archive = RAW / f"{name}.zip"
        if args.force or not archive.exists():
            print(f"downloading {name} (README says {info['readme_size']})")
            out = gdown.download(id=info["id"], output=str(archive), quiet=False)
            if out is None or not archive.exists():
                print(f"download of {name} failed", file=sys.stderr)
                return 1
        digest = sha256(archive)
        if name in known and known[name]["sha256"] != digest:
            print(
                f"checksum mismatch for {name}: expected {known[name]['sha256']}, got {digest}",
                file=sys.stderr,
            )
            return 1
        known[name] = {"sha256": digest, "bytes": archive.stat().st_size, "drive_id": info["id"]}
        n = safe_extract(archive, DATA)
        print(f"{name}: {archive.stat().st_size / 1e6:.1f} MB, sha256 {digest[:12]}…, {n} files extracted")

    CHECKSUMS.write_text(json.dumps(known, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
