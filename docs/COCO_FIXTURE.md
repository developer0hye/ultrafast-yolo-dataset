# Real COCO validation fixture

`bench/prepare_coco.py` creates separate Detection and Segmentation fixtures from
all 5,000 [official COCO val2017 images](https://cocodataset.org/#download). It uses
the unmodified `convert_coco` from the pinned Ultralytics source and validates the
entire converter file's hash. The image and annotation archives have fixed SHA-256
values in the script; a mismatch stops preparation before creating the output.

```sh
mkdir -p /path/to/archives
curl --fail --location --output /path/to/archives/val2017.zip \
  https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip
curl --fail --location --output /path/to/archives/annotations_trainval2017.zip \
  https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
python bench/prepare_coco.py --archives /path/to/archives --out /path/to/new/coco-yolo
python bench/cache_startup.py --corpus /path/to/new/coco-yolo/detect --mode miss --out bench/out/coco-detect-miss.json
python bench/cache_startup.py --corpus /path/to/new/coco-yolo/segment --mode miss --out bench/out/coco-segment-miss.json
python bench/cache_startup.py --corpus /path/to/new/coco-yolo/detect --mode hit --out bench/out/coco-detect-hit.json
python bench/cache_startup.py --corpus /path/to/new/coco-yolo/segment --mode hit --out bench/out/coco-segment-hit.json
```

The output directory must not already exist because upstream conversion appends
label rows. All original image bytes are retained; each task gets independent
copies so a JPEG repair cannot affect the other task or the extracted source.
Crowd/invalid boxes are omitted, classes use the standard 91-to-80 mapping,
multipart polygons use upstream joining, and upstream box-shaped fallback segments
are retained if needed. No custom geometry conversion is substituted. Images
without a label file remain in the fixture.

Both tasks contain 4,952 TXT files with 36,335 serialized annotation rows and
5,000 distinct images. Subsequent scan deduplication can affect accepted object
counts. There are 48 missing label files; all 5,000 images pass the current measured
scan. The generated source manifest records every original image's hash/dimensions,
the annotation hash and converter identity. Each task manifest records the combined
ordered image/label content hash and the exact source-manifest file hash.

Frozen copies of these manifests are in `bench/results/coco-*-manifest.json` and
`coco-source-manifest.json`. The task content hashes are independent of the parent
directory, allowing the same fixture to be copied to another host and verified.
Cache files are excluded from these hashes. Copy `images`, `labels` and `coco.json`
when moving a task; caches need not be transferred.

COCO runs verify the complete labels, first batch, scan counters and ordered
diagnostic messages against the reference, outside timing and memory sampling.
Hit priming runs in separate processes, and content-checked hit baselines use the
same native hashing backend. Strong validation includes all ~816–830 MB of input
data; the historical synthetic 100k corpus instead contains small repeated-template
JPEG content in distinct files. These workloads exercise different costs.

The fixture is for preprocessing/startup and later throughput measurement.
Training on this validation split would be a performance workload only, not an
accuracy experiment. Real-data correctness/startup evidence does not establish
full GPU training throughput or generalization to other datasets.
