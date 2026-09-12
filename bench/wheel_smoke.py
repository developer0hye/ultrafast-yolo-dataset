"""Exercise an installed wheel with runtime dependencies and the frozen label oracle.

Run from a clean venv after installing the wheel, NumPy, and Pillow. Ultralytics
and Torch are deliberately absent from this standalone-package check.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import PIL
import ultrafast_yolo_dataset as uyd
from PIL import Image
from ultrafast_yolo_dataset._cache import CacheMiss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert Path(uyd.__file__).is_relative_to(sys.prefix), "must import the installed wheel"
    assert importlib.util.find_spec("ultralytics") is None, "use a standalone clean environment"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from reference import verify_image_label

    texts = ["0 .5 .5 .2 .2", "", None, "invalid", "0 0 0 1 0 1 1", "0\u00a0.5 .5 .2 .2", "0 -0.0 0 1 0 1 1"]
    cases = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        images, labels = [], []
        for index, text in enumerate(texts):
            image, label = root / f"image-{index}.png", root / f"label-{index}.txt"
            Image.new("RGB", (32, 32)).save(image)
            if text is not None:
                label.write_text(text)
            images.append(str(image))
            labels.append(str(label))
        expected = [verify_image_label((im, lb, "val: ", False, 2, 0, 0, False)) for im, lb in zip(images, labels)]
        for task in ("detect", "segment"):
            for policy in ("content", "metadata"):
                path = root / f"{task}-{policy}.uydcache"
                options = {"num_classes": 2, "task": task, "fingerprint": policy, "prefix": "val: "}
                cold = uyd.scan_cached(path, images, labels, **options)
                warm = uyd.scan_cached(path, images, labels, **options)
                assert not cold.cache_hit and warm.cache_hit and cold.cache_write_error is None
                assert cold.summary == warm.summary and cold.diagnostics == warm.diagnostics
                assert [d.message for d in warm.diagnostics] == [r[-1] for r in expected if r[-1]]
                rows, valid = warm.to_ultralytics_labels(), [r for r in expected if r[0] is not None]
                assert len(rows) == len(valid)
                assert dict(warm.summary) == {
                    "total": len(images),
                    **{
                        name: sum(r[index] for r in expected)
                        for name, index in (("missing", 5), ("found", 6), ("empty", 7), ("corrupt", 8))
                    },
                }
                for row, reference in zip(rows, valid):
                    assert row["im_file"] == reference[0] and row["shape"] == reference[2]
                    for actual, wanted in ((row["cls"], reference[1][:, :1]), (row["bboxes"], reference[1][:, 1:])):
                        assert actual.dtype == wanted.dtype and actual.shape == wanted.shape
                        assert actual.tobytes() == wanted.tobytes() and actual.flags.writeable
                    assert len(row["segments"]) == len(reference[3])
                    for actual, wanted in zip(row["segments"], reference[3]):
                        assert actual.dtype == wanted.dtype and actual.shape == wanted.shape
                        assert actual.tobytes() == wanted.tobytes() and actual.flags.writeable
                cases.append({"task": task, "policy": policy, "cache_hit": warm.cache_hit, "parity": True})
        edited = Path(labels[0])
        before = edited.stat()
        edited.write_text("0 .5 .5 .3 .3")
        os.utime(edited, ns=(before.st_atime_ns, before.st_mtime_ns))
        for policy in ("content", "metadata"):
            value = uyd.load_cache(
                root / f"segment-{policy}.uydcache",
                images,
                labels,
                num_classes=2,
                task="segment",
                fingerprint=policy,
                prefix="val: ",
            )
            assert isinstance(value, CacheMiss) == (policy == "content")
    report = {
        "scope": "installed standalone wheel on the recorded platform; Detect/Segment cold/warm caches, label-oracle parity, restored-mtime invalidation; no framework or platform-matrix claim",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "package": uyd.__file__,
        "extension_sha256": hashlib.sha256(Path(uyd._native.__file__).read_bytes()).hexdigest(),
        "cases": cases,
        "content_edit_detected": True,
        "metadata_weaker_contract_confirmed": True,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
