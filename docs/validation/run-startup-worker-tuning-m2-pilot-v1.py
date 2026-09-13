"""Fail-stop functional pilots, only after the worker phase has passed audit."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT = Path('/Volumes/T7/ultrafast-vision-build')
PREFIX = 'startup-worker-tuning-m2-pilot-v1'
PLAN = ROOT / (PREFIX + '-plan.json')
OUT = ROOT / (PREFIX + '-qualification.json')
STATE = dict(complete=False, passed=False, pid=os.getpid(), started_at_ns=time.time_ns(), records=[])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    temp = OUT.with_suffix('.tmp')
    temp.write_text(json.dumps(STATE, indent=2) + '\n')
    temp.replace(OUT)


def main():
    assert not OUT.exists()
    audit_path = ROOT / 'validation-workers-m2-v1-audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['passed'] and audit['trials'] == 21 and audit['measured_trials'] == 18
    plan = json.loads(PLAN.read_text())
    source = Path(plan['source'])
    assert len(plan['cases']) == 6
    STATE.update(plan_sha256=sha(PLAN), phase_audit_sha256=sha(audit_path), script_sha256=sha(Path(__file__)), scope=plan['scope'])
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    overrides = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', YOLO_OFFLINE='true')
    env.update(overrides)
    STATE['environment_overrides'] = overrides
    save()
    for case in plan['cases']:
        for name, digest in plan['source_files'].items():
            assert sha(source / name) == digest, name
        command = case['command']
        out = Path(command[command.index('--out') + 1])
        assert not out.exists()
        log = out.with_suffix('.log')
        rec = dict(workers=case['workers'], mode=case['mode'], command=command,
                   cwd=str(source), started_at_ns=time.time_ns(), path=str(out))
        STATE['records'].append(rec)
        save()
        print('start', case['workers'], case['mode'], flush=True)
        with log.open('x') as stream:
            proc = subprocess.Popen(command, cwd=source, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            rec['pid'] = proc.pid
            save()
            try:
                code = proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                rec['timed_out'] = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait(timeout=10)
        rec.update(returncode=code, finished_at_ns=time.time_ns(), log_sha256=sha(log))
        if out.exists():
            rec['raw_sha256'] = sha(out)
        save()
        assert code == 0 and not rec.get('timed_out', False)
        raw = json.loads(out.read_text())
        assert raw['complete'] and raw['passed'] and raw['workers'] == case['workers'] and raw['mode'] == case['mode']
        assert raw['result']['workers'] == case['workers'] and raw['result']['loader_workers'] == 0
        assert raw['result']['output_sha256'] == plan['expected_outputs']
        rec.update(validated=True, constructor_s=raw['result']['constructor_s'],
                   cache_sha256=raw['cache']['sha256'])
        save()
        print('passed', case['workers'], case['mode'], flush=True)
    assert sha(Path(plan['original_cache'])) == plan['original_cache_sha256']
    STATE.update(complete=True, passed=True, finished_at_ns=time.time_ns())
    save()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        STATE.update(complete=True, passed=False, finished_at_ns=time.time_ns(), error=traceback.format_exc())
        save()
        raise
