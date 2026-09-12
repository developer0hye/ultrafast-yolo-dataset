"""Owned byte snapshots for cacheable scans; never infer digests from a reread."""

import hashlib
import io
import json

from PIL import Image

from . import _native

InputChangedError = _native.InputChangedError


def fingerprints(paths, max_bytes, workers, *, content=True):
    if not paths:
        return []
    return json.loads(_native.fingerprint_paths(list(paths), max_bytes, workers, content))


class _ImageBytes(io.BytesIO):
    def __init__(self, data, path):
        super().__init__(data)
        self.path = path

    def __repr__(self):
        # Pillow's unidentified-image diagnostic uses repr(fp) when opening a
        # file-like object. Retain the reference's original path diagnostic.
        return repr(self.path)


def verify_image(args):
    from ._scan import RepairRequired, _check_image, _verify_image

    path, repair, limit = args
    try:
        snapshot = _native.read_snapshot(path, limit)
    except (_native.SnapshotLimitError, OSError) as error:
        # Verification remains useful even if cache capture is unavailable.
        # Retain its result/messages once, including any explicit JPEG repair.
        return (*_verify_image((path, repair)), None, str(error))
    proof = json.loads(snapshot.fingerprint_json())
    if snapshot.kind in ("missing", "directory"):
        try:
            with open(path, "rb"):
                raise InputChangedError(f"non-file input became readable during scan: {path}")
        except (FileNotFoundError, IsADirectoryError) as error:
            if fingerprints([path], limit, 1)[0] != proof:
                raise InputChangedError(f"input changed during scan: {path}") from error
            return None, str(error), "corrupt_image", proof, None
    if snapshot.kind != "file":
        raise _native.SnapshotLimitError(f"cannot snapshot a non-regular image: {path}")
    data = snapshot.data
    del snapshot

    def save_repaired(corrected):
        nonlocal proof
        if fingerprints([path], limit, 1)[0] != proof:
            raise InputChangedError(f"JPEG changed before repair: {path}")
        output = io.BytesIO()
        corrected.save(output, "JPEG", subsampling=0, quality=100)
        repaired = output.getvalue()
        # Record the encoder's actual bytes, not whatever might replace the
        # path between saving and collecting the final fingerprint.
        digest = hashlib.sha256(repaired).hexdigest()
        with open(path, "wb") as stream:
            stream.write(repaired)
        after = fingerprints([path], max(limit, len(repaired)), 1)[0]
        if after["sha256"] != digest:
            raise InputChangedError(f"JPEG changed during repair: {path}")
        proof = after

    try:
        message, shape = _check_image(
            path,
            repair,
            opener=lambda _: Image.open(_ImageBytes(data, path)),
            tail=data[-2:],
            save_repaired=save_repaired,
        )
        return shape, message, "", proof, None
    except (InputChangedError, MemoryError):
        raise
    except RepairRequired as error:
        return None, str(error), "repair_required", proof, None
    except Exception as error:  # noqa: BLE001 - same Pillow record classification as the reference
        return None, str(error), "corrupt_image", proof, None


def fallback_open(data):
    def open_label(_, *, encoding):
        return io.TextIOWrapper(io.BytesIO(data), encoding=encoding)

    return open_label
