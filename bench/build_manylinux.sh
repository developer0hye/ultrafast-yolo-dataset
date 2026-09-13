#!/usr/bin/env bash
set -euo pipefail
: "${GITHUB_WORKSPACE:?run through the hosted workflow}"
[[ "$PWD" == "$GITHUB_WORKSPACE" ]]
# Checkout and build-step Git see different temporary HOME configuration.
# Trust only this known checkout inside the disposable container.
git config --global --add safe.directory "$GITHUB_WORKSPACE"
export PATH="/opt/python/cp312-cp312/bin:$PATH"
export RUSTUP_HOME="$RUNNER_TEMP/rustup"
export CARGO_HOME="$RUNNER_TEMP/cargo"
export CARGO_TARGET_DIR="$RUNNER_TEMP/manylinux-target"
mkdir -p manylinux-evidence
git rev-parse HEAD > manylinux-evidence/source-commit.txt
python -VV > manylinux-evidence/python-build.txt
getconf GNU_LIBC_VERSION > manylinux-evidence/glibc-build.txt
gcc --version > manylinux-evidence/gcc.txt
cmake --version > manylinux-evidence/cmake.txt
patchelf --version > manylinux-evidence/patchelf.txt
cp /etc/os-release manylinux-evidence/os-release
# Pin the installer bytes as well as the Rust toolchain.
curl --fail --location https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init -o "$RUNNER_TEMP/rustup-init"
python - <<'PY'
import hashlib, os
from pathlib import Path
p = Path(os.environ['RUNNER_TEMP']) / 'rustup-init'
assert hashlib.sha256(p.read_bytes()).hexdigest() == '20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c'
PY
chmod +x "$RUNNER_TEMP/rustup-init"
"$RUNNER_TEMP/rustup-init" -y --no-modify-path --profile minimal --default-toolchain 1.98.0
export PATH="$CARGO_HOME/bin:$PATH"
rustc -vV > manylinux-evidence/rustc.txt
python - <<'PY'
import subprocess
from pathlib import Path
sysroot = Path(subprocess.check_output(['rustc', '--print', 'sysroot'], text=True).strip())
supplied = sysroot / 'share/doc/rust/COPYRIGHT-library.html'
bundled = Path('python/ultrafast_yolo_dataset/licenses/rust-stdlib/1.98.0/COPYRIGHT-library.html')
assert supplied.read_bytes() == bundled.read_bytes(), 'Rust toolchain notices differ'
PY
python -m pip install build==1.3.0 maturin==1.15.0 auditwheel==6.8.2 numpy==2.4.4 pillow==12.1.1
python -m pip freeze > manylinux-evidence/build-environment.txt
python -m build --wheel --sdist --no-isolation
cp -a dist manylinux-evidence/original-dist
python -m auditwheel show dist/*.whl > manylinux-evidence/auditwheel-before.txt
python -m auditwheel repair --plat manylinux_2_28_x86_64 --wheel-dir wheelhouse dist/*.whl > manylinux-evidence/auditwheel-repair.txt 2>&1
python -m auditwheel show wheelhouse/*.whl > manylinux-evidence/auditwheel-after.txt
# Keep original wheel and repaired wheel separate for provenance.
python - <<'PY'
from pathlib import Path
import shutil
original = list(Path('dist').glob('*.whl'))
repaired = list(Path('wheelhouse').glob('*.whl'))
assert len(original) == len(repaired) == 1
original[0].unlink()
shutil.copy2(repaired[0], Path('dist') / repaired[0].name)
PY
python -m venv "$RUNNER_TEMP/manylinux-runtime"
export PATH="$RUNNER_TEMP/manylinux-runtime/bin:$PATH"
export LD_LIBRARY_PATH=""
python -m pip install numpy==2.4.4 pillow==12.1.1
python -m pip install --no-deps dist/*.whl
python bench/ci_wheel_check.py
python bench/check_manylinux_runtime.py
python bench/wheel_smoke.py --out dist/standalone-cache-smoke.json
python -m pip install pytest==9.1.1
python -m pytest -q tests/test_parser.py tests/test_snapshot_hashes.py tests/test_materialize.py tests/test_benchmark_fixture.py tests/test_cache_section_buffers.py --junitxml=dist/core-tests.xml
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.4.4 pillow==12.1.1 opencv-python==4.13.0.92 pi-heif==1.4.0 https://github.com/ultralytics/ultralytics/archive/795a556942a12fe0124cf767888194a1d0b83e2e.tar.gz
python -m pytest -q tests/test_scan.py tests/test_cache.py tests/test_integration.py --junitxml=dist/integration-tests.xml
python -m pip freeze > dist/final-environment.txt
