"""Experimental label engine and hybrid Pillow scan. Cache/Ultralytics adapter pending.

Statuses: 0 valid, 1 missing, 2 empty, 3 reference_required, 4 resource_limit.
Rows requiring reference processing are never represented as successfully parsed.
"""

import os

from . import _native

__version__ = "0.1.0a1"


def parse_labels(label_paths, *, num_classes, task="detect", single_cls=False, workers=4, max_file_bytes=16 * 1024**2):
    """Read/parse labels into owned packed storage; does not verify images.

    ``to_arrays()`` explicitly copies data into writable NumPy arrays. Inspect
    statuses/reference_required before using results; reference fallback is not
    yet automatic. Both tasks follow the reference's content-based segment detection.
    """
    if task not in ("detect", "segment"):
        raise ValueError("only detect and segment are supported")
    paths = [os.fsdecode(os.fspath(p)) for p in label_paths]
    return _native.parse_files(paths, num_classes, single_cls, workers, max_file_bytes)


def scan(*args, **kwargs):
    """Hybrid Pillow image verification and native labels; see _scan.scan."""
    from ._scan import scan as implementation

    return implementation(*args, **kwargs)
