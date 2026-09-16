"""Generate auditable full-vendor hashes and exact patches against the pinned git archive."""
import difflib
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, required=True,
                    help="git archive of the pinned upstream commit, without a prefix")
archive = parser.parse_args().archive
vendor = CODE / "vendor/show_harness"
patch_dir = CODE / "vendor/patches"
patch_dir.mkdir(exist_ok=True)
archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
assert archive_hash == "9c43e59b66ead6f1eeeb5a4cdaa7b53264e57dc266988b5b30d62f347506aa23"
files, changed = [], []
with tarfile.open(archive) as source:
    for member in source.getmembers():
        if not member.isfile():
            continue
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid archive entry")
        before = source.extractfile(member).read()
        after = (vendor / path).read_bytes()
        original_hash = hashlib.sha256(before).hexdigest()
        current_hash = hashlib.sha256(after).hexdigest()
        files.append(dict(path=str(path), upstream_sha256=original_hash, vendored_sha256=current_hash))
        if before != after:
            changed.append(str(path))
            patch = "".join(difflib.unified_diff(
                before.decode().splitlines(keepends=True), after.decode().splitlines(keepends=True),
                fromfile="a/" + str(path), tofile="b/" + str(path)))
            (patch_dir / (str(path).replace("/", "_") + ".patch")).write_text(patch)
assert sorted(changed) == ["core/runners/real.py", "core/vlm/roles.py"], changed
manifest = dict(origin="https://github.com/showlab/Show-Harness.git",
                commit="137d5718c3b7af0150764d8f9beeb252c9f2794a", license="Apache-2.0",
                archive_sha256=archive_hash, modified_paths=sorted(changed),
                source_files=len(files), files=sorted(files, key=lambda x: x["path"]))
(CODE / "vendor/show-harness-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k: v for k, v in manifest.items() if k != "files"}))
