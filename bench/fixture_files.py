"""Bounded-directory fixture hashing and distinct-file audits (no ML imports)."""

import hashlib
import json
import os
import stat
from pathlib import Path


def sorted_files(directory, suffix):
    """Match sorted Path.rglob leaf order without retaining the whole tree.

    Prepared fixtures use bounded shard directories. As with Path.rglob, do not
    recurse through directory symlinks. Matching file symlinks are still read by
    fingerprint(); the stricter synthetic-fixture audit rejects them separately.
    """
    with os.scandir(directory) as entries:
        ordered = sorted(entries, key=lambda entry: entry.name)
    for entry in ordered:
        path = Path(entry.path)
        if entry.is_dir(follow_symlinks=False):
            if entry.name.endswith(suffix):
                raise IsADirectoryError(path)
            yield from sorted_files(path, suffix)
        elif entry.name.endswith(suffix):
            yield path


def fingerprint(root):
    digest, count, size = hashlib.sha256(), 0, 0
    for folder, suffix in (("images", ".jpg"), ("labels", ".txt")):
        for path in sorted_files(root / folder, suffix):
            data = path.read_bytes()
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + data)
            count += 1
            size += len(data)
    return {"files": count, "bytes": size, "sha256": digest.hexdigest()}


def audit_synthetic(root, expected_pairs):
    """Verify all generated paths, content hashes and lack of file aliases.

    JPEG data is synthetic and may repeat. Distinct files means separate regular
    files with one link, not unique image content or proof of distinct SSD blocks.
    This preflight is outside benchmark timers and intentionally warms OS caches.
    """
    if type(expected_pairs) is not int or expected_pairs <= 0:
        raise ValueError("expected_pairs must be positive")
    manifest = json.loads((root / "startup.json").read_text())
    if manifest.get("kind") != "ultrafast-yolo-startup-synthetic-v1" or manifest.get("count") != expected_pairs:
        raise ValueError("wrong synthetic fixture kind or pair count")
    labels = json.loads((root / "labels/manifest.json").read_text())
    if labels != manifest["labels"] or labels["count"] != expected_pairs or labels["task"] != manifest["task"]:
        raise ValueError("label manifest mismatch")
    expected_shards = {f"{index:04d}" for index in range((expected_pairs + 999) // 1000)}
    corpus_digest, label_digest = hashlib.sha256(), hashlib.sha256()
    total_bytes, label_bytes = 0, 0
    counts = {}
    for folder, suffix in (("images", ".jpg"), ("labels", ".txt")):
        directory = root / folder
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"invalid fixture directory: {directory}")
        actual_shards = {path.name for path in directory.iterdir() if path.is_dir()}
        if actual_shards != expected_shards:
            raise ValueError(f"unexpected shard directories: {folder}")
        if any(directory.glob("*" + suffix)):
            raise ValueError(f"unexpected unsharded inputs: {folder}")
        count = 0
        for shard_name in sorted(expected_shards):
            shard = directory / shard_name
            if shard.is_symlink():
                raise ValueError(f"directory alias: {shard}")
            start = int(shard_name) * 1000
            expected_names = {f"{index:08d}{suffix}" for index in range(start, min(start + 1000, expected_pairs))}
            if {p.name for p in shard.iterdir()} != expected_names:
                raise ValueError(f"unexpected/missing files: {shard}")
            for name in sorted(expected_names):
                path = shard / name
                metadata = path.lstat()
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ValueError(f"expected an unaliased regular file: {path}")
                data = path.read_bytes()
                corpus_digest.update(path.relative_to(root).as_posix().encode() + b"\0" + data)
                total_bytes += len(data)
                count += 1
                if folder == "labels":
                    label_digest.update(path.relative_to(directory).as_posix().encode() + b"\0" + data)
                    label_bytes += len(data)
        counts[folder] = count
    actual = {"files": sum(counts.values()), "bytes": total_bytes, "sha256": corpus_digest.hexdigest()}
    if actual != manifest["fingerprint"]:
        raise ValueError("whole-fixture fingerprint mismatch")
    if label_digest.hexdigest() != labels["sha256_names_and_contents"] or label_bytes != labels["bytes"]:
        raise ValueError("label fingerprint mismatch")
    return {
        "kind": "synthetic distinct-file preflight",
        "pairs": expected_pairs,
        "counts": counts,
        "fingerprint": actual,
        "all_regular_single_link_files": True,
        "scope": "names, bytes and filesystem entries verified; repeated synthetic content and OS-cache warming are explicit",
    }


def source_hashes(project):
    """Source inventory only; separately identify the actual loaded extension."""
    paths = [*sorted((project / "src").glob("*.rs")), *sorted((project / "python").rglob("*.py"))]
    paths.extend(sorted((project / "bench").glob("*.py")))
    paths.extend(project / name for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "tests/reference.py"))
    return {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("retain previous preflight reports")
    result = audit_synthetic(args.corpus.resolve(), args.pairs)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
