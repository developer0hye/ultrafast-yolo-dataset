import numpy as np
import pytest
from reference import verify_image_label
from ultrafast_yolo_dataset import _native, parse_labels


def oracle(path, classes=80, single=False):
    return verify_image_label(("image.jpg", str(path), "train: ", False, classes, 0, 0, single))


def compare(text, path, classes=80, single=False):
    path.write_bytes(text)
    expected = oracle(path, classes, single)
    parsed = _native.parse_texts([text], classes, single)
    got = parsed.to_arrays()
    if expected[0] is None:
        assert got["statuses"][0] == 3
        return
    assert got["statuses"][0] in (0, 2), (text, got, expected)
    assert got["labels"].shape == expected[1].shape
    assert got["labels"].dtype == expected[1].dtype
    assert got["labels"].tobytes() == expected[1].tobytes(), (text, got["labels"], expected[1])
    offsets = got["segment_offsets"]
    assert len(offsets) == len(expected[3]) + 1
    for i, segment in enumerate(expected[3]):
        assert got["segment_points"][offsets[i] : offsets[i + 1]].tobytes() == segment.tobytes()
    duplicates = int(got["duplicates"][0])
    assert bool(duplicates) == ("duplicate labels removed" in expected[-1])


@pytest.mark.parametrize(
    "text",
    [
        b"",
        b" \n\t ",
        b"0 .5 .5 .2 .2\r\n",
        b"0 1.0000000596046448 .5 .2 .2",
        b"1 .5 .5 .2 .2\n0 .4 .4 .1 .1",
        b"1 .5 .5 .2 .2\n0 .4 .4 .1 .1\n1 .5 .5 .2 .2",
        b"0 -0.0 .5 .2 .2\n0 0 .5 .2 .2",
        b".5 .5 .5 .2 .2",
        b"-.01 1.01 .5 .2 .2",
        b"0 0 0 1 0 1 1\n0 1 1 0 0 1 0",
        b"0 .5 .5 .2 .2\n \n0 .3 .3 .1 .1",
        b"nan .5 .5 .2 .2",
        b"0 inf .5 .2 .2",
        b"0 1.02 .5 .2 .2",
        b"-1 .5 .5 .2 .2",
        b"80 .5 .5 .2 .2",
        b"0 0 0 1 0 1 1\n0 .5 .5 .2 .2",
        b"0 0 0 1 1 0 1 0",
    ],
)
def test_reference_edges(text, tmp_path):
    compare(text, tmp_path / "labels.txt")


def test_single_class(tmp_path):
    compare(b"999 .5 .5 .2 .2", tmp_path / "a.txt", 1, True)


def test_files_order_lifetime_limits(tmp_path):
    paths = [tmp_path / f"标注 {i}.txt" for i in range(6)]
    for i, p in enumerate(paths[:-1]):
        p.write_text(f"{i} .5 .5 .2 .2")
    paths[2].write_text("")
    paths[3].write_text("invalid")
    for workers in (1, 2, 4):
        packed = parse_labels(paths, num_classes=10, workers=workers)
        a = packed.to_arrays()
        assert a["statuses"].tolist() == [0, 0, 2, 3, 0, 1]
        assert a["labels"][:, 0].tolist() == [0, 1, 4]
        assert a["object_offsets"].tolist() == [0, 1, 2, 2, 2, 3, 3]
        assert packed.reference_required == [3]
        a["labels"][:] = 999
        assert packed.to_arrays()["labels"][0, 0] == 0
        del packed
        assert a["labels"][0, 0] == 999
    limited = parse_labels(paths[:1], num_classes=10, max_file_bytes=2).to_arrays()
    assert limited["statuses"].tolist() == [4]


def test_10000_seeded_differential_cases(tmp_path):
    rng = np.random.default_rng(20260912)
    path = tmp_path / "case.txt"
    for i in range(10000):
        rows = []
        for _ in range(int(rng.integers(1, 20))):
            cls = int(rng.integers(0, 80))
            if i % 2:
                points = rng.uniform(0, 1, (int(rng.integers(3, 20)), 2))
                values = points.ravel()
            else:
                values = rng.uniform(-0.005, 1.005, 4)
            rows.append(str(cls) + " " + " ".join(format(float(v), ".17g") for v in values))
        if i % 3 == 0:
            rows.extend(rows[:2])
        compare(("\n".join(rows)).encode(), path)
