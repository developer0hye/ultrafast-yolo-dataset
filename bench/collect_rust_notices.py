"""Collect checksum-verified Cargo notices for explicit normal/build target trees.

Conservatively includes source comment notices from selected crates, including
source variants the linker may not use. This is not a legal-completeness proof.
"""

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

import tomllib

COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*(?:\n[ \t]*//[^\n]*)*", re.DOTALL)
NOTICE = re.compile(r"copyright|licen[cs]e|permission|public domain|derived from|based on", re.IGNORECASE)
NOTICE_FILE = re.compile(r"^(?:LICENSE|LICENCE|COPYING|COPYRIGHT|NOTICE)(?:[._-].*)?$", re.IGNORECASE)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--trees", type=Path, nargs="+", required=True)
    parser.add_argument("--stdlib", nargs="+", default=[], help="version=directory containing COPYRIGHT-library.html")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "retain prior collections"
    args.out.mkdir(parents=True)
    metadata = json.loads(args.metadata.read_text())
    lock = tomllib.loads(args.lock.read_text())
    locked = {(p["name"], p["version"]): p for p in lock["package"]}
    selected, targets = set(), []
    for tree in args.trees:
        packages = set(re.findall(r"^([A-Za-z0-9_-]+) v([^\s]+)", tree.read_text(), re.MULTILINE))
        assert packages, tree
        selected.update(packages)
        targets.append({"tree": tree.name, "sha256": sha(tree.read_bytes()), "packages": sorted(packages)})
    known = {(p["name"], p["version"]) for p in metadata["packages"]}
    assert selected <= known, "tree and metadata disagree"
    copied, collected, excluded, blocks = {}, [], [], {}

    def copy(name, data):
        path = args.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        copied[name] = sha(data)

    for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        if not package["source"]:
            continue
        key = (package["name"], package["version"])
        if key not in selected:
            excluded.append({"name": key[0], "version": key[1], "reason": "absent from all specified target trees"})
            continue
        source = Path(package["manifest_path"]).parent
        archive = source.parents[2] / "cache" / source.parent.name / (source.name + ".crate")
        data = archive.read_bytes()
        archive_sha = sha(data)
        assert archive_sha == locked[key]["checksum"], archive
        notices, source_notices = [], {}
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                path = PurePosixPath(member.name)
                assert not path.is_absolute() and ".." not in path.parts
                assert path.parts[0] == source.name
                if not member.isfile():
                    continue
                relative = PurePosixPath(*path.parts[1:])
                content = None
                if NOTICE_FILE.fullmatch(path.name):
                    content = tar.extractfile(member).read()
                    name = f"crates/{source.name}/{relative}"
                    copy(name, content)
                    notices.append(name)
                if relative.suffix in (".rs", ".c", ".h", ".S", ".s", ".cpp", ".hpp") and not any(
                    part in ("tests", "benches", "examples") for part in relative.parts
                ):
                    if content is None:
                        content = tar.extractfile(member).read()
                    for match in COMMENT.finditer(content.decode("utf-8", errors="replace")):
                        text = match.group()
                        if NOTICE.search(text):
                            digest = sha(text.encode())
                            block = blocks.setdefault(digest, {"text": text, "sources": set()})
                            block["sources"].add(f"{source.name}/{relative}")
                            source_notices[str(relative)] = sha(content)
        assert notices, f"selected crate lacks notice files: {source.name}"
        collected.append(
            {
                "name": key[0],
                "version": key[1],
                "license_expression": package["license"],
                "crate_sha256": archive_sha,
                "notice_files": sorted(notices),
                "source_files_with_collected_comments": source_notices,
            }
        )

    text = ["Conservative comment notice collection from selected checksum-verified Cargo archives.\n"]
    for digest, block in sorted(blocks.items()):
        text.extend(
            [
                "\n" + "=" * 72,
                "Notice SHA-256: " + digest,
                "Source files:\n" + "\n".join(sorted(block["sources"])),
                "",
                block["text"],
            ]
        )
    copy("SOURCE-COMMENT-NOTICES.txt", ("\n".join(text) + "\n").encode())
    toolchains = []
    for value in args.stdlib:
        version, directory = value.split("=", 1)
        assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version), "use an exact Rust release"
        root = Path(directory)
        copyright_file = root / "COPYRIGHT-library.html"
        assert copyright_file.is_file()
        files = [copyright_file, *sorted((root / "licenses").glob("*.txt"))]
        files.extend(sorted((root / "html/static.files").glob("LICENSE-*.txt")))
        names = []
        for path in files:
            name = f"rust-stdlib/{version}/{path.relative_to(root)}"
            copy(name, path.read_bytes())
            names.append(name)
        toolchains.append({"rust_release": version, "files": names})
    manifest = {
        "scope": "selected normal/build Cargo target trees, crate texts/comment notices and supplied Rust standard-library notices; conservative source collection, not actual linked-code or complete redistribution proof",
        "collector_sha256": sha(Path(__file__).read_bytes()),
        "cargo_lock_sha256": sha(args.lock.read_bytes()),
        "metadata_sha256": sha(args.metadata.read_bytes()),
        "target_trees": targets,
        "packages": collected,
        "excluded_packages": excluded,
        "rust_stdlib": toolchains,
        "unique_comment_blocks": len(blocks),
        "files_sha256": dict(sorted(copied.items())),
        "remaining": [
            "Review embedded notices and system dependencies on each supported platform",
            "Refresh Rust standard-library notices for any different build toolchain",
            "Verify wheel and sdist packaging and installed bytes",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"packages": len(collected), "excluded": excluded, "files": len(copied), "comment_blocks": len(blocks)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
