"""Download helpers shared by the data and reference-model scripts.

Downloads are treated as untrusted input: archives are extracted with a path check (nothing
may land outside the target folder), nothing inside them is executed, and every file's SHA-256
is recorded in data/checksums.json so later runs can verify they have the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHECKSUMS = REPO / "data" / "checksums.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive: Path, target: Path) -> int:
    """Extract a zip, refusing any member that would land outside `target`. Skips macOS metadata."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename
            base = Path(name).name
            if name.startswith("__MACOSX/") or base.startswith("._") or base == ".DS_Store":
                continue
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


def gdrive(file_id: str, out: Path) -> Path:
    """Download a public Google Drive file (handles the large-file confirmation page)."""
    import gdown

    out.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=file_id, output=str(out), quiet=False)
    if result is None or not out.exists():
        raise RuntimeError(f"download of Google Drive file {file_id} failed")
    return out


def http(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r, out.open("wb") as f:  # noqa: S310 (fixed https URLs)
        f.write(r.read())
    return out


def record(name: str, path: Path, **extra) -> dict:
    """Verify `path` against the stored checksum for `name`, or store it the first time."""
    known = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    digest = sha256(path)
    if name in known and known[name]["sha256"] != digest:
        raise RuntimeError(f"checksum mismatch for {name}: expected {known[name]['sha256']}, got {digest}")
    known[name] = {"sha256": digest, "bytes": path.stat().st_size, **extra}
    CHECKSUMS.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUMS.write_text(json.dumps(known, indent=2) + "\n")
    return known[name]
