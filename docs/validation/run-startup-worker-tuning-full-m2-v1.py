"""Full-size cache-hit confirmation: prior seven-worker setting versus sixteen.

Two separate primers then five alternating pairs. This uses the exact functional
pilot worker/harness and one immutable, isolated cache copy per worker setting.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import traceback

import psutil

ROOT = Path('/Volumes/T7/ultrafast-vision-build')
PREFIX = 'startup-worker-tuning-full-m2-v1'
OUT = ROOT / (PREFIX + '.json')
STATE = dict(complete=False, passed=False, pid=os.getpid(), started_at_ns=time.time_ns(), records=[])


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save():
    temp = OUT.with_suffix('.tmp')
    temp.write_text(json.dumps(STATE, indent=2) + '\n')
    temp.replace(OUT)


def main():
    phase_path = ROOT / 'validation-workers-m2-v1-audit.json'
    pilot_path = ROOT / 'startup-worker-tuning-m2-pilot-v1-qualification.json'
    phase, pilot = (json.loads(p.read_text()) for p in (phase_path, pilot_path))
    assert phase['passed'] and phase['trials'] == 21 and phase['measured_trials'] == 18
    assert pilot['complete'] and pilot['passed'] and len(pilot['records']) == 6
    assert all(r['validated'] and r['returncode'] == 0 for r in pilot['records'])
    for rec in pilot['records']:
        assert sha(Path(rec['path'])) == rec['raw_sha256']
        assert sha(Path(rec['path']).with_suffix('.log')) == rec['log_sha256']
    pilot_plan_path = ROOT / 'startup-worker-tuning-m2-pilot-v1-plan.json'
    pilot_plan = json.loads(pilot_plan_path.read_text())
    assert sha(pilot_plan_path) == pilot['plan_sha256']
    source = Path('/tmp') / (PREFIX + '-source')
    source.mkdir()
    for name, digest in pilot_plan['source_files'].items():
        original = Path(pilot_plan['source']) / name
        assert sha(original) == digest
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        assert sha(target) == digest
    cache = ROOT / 'snapshot-startup-full-m2-v1.caches/baseline/0000-3bf8910a76e9bbf572c4.uydcache'
    cache_sha = '2e94732eee42009919f1552b702c0ba158e1a0b1b1a49933f7e599eb9a2f1dc6'
    assert sha(cache) == cache_sha
    cache_roots = {}
    for workers in (7, 16):
        folder = ROOT / (PREFIX + f'-cache-w{workers}')
        folder.mkdir()
        shutil.copyfile(cache, folder / cache.name)
        assert sha(folder / cache.name) == cache_sha
        cache_roots[workers] = folder
    corpus = ROOT / 'startup-detect-500k-v1'
    fixture = json.loads((corpus / 'startup.json').read_text())
    assert fixture['count'] == 500000 and fixture['task'] == 'detect'
    assert fixture['fingerprint'] == dict(files=1000000, bytes=3625093750,
                                        sha256='931bd8059aefe560c3601e49d9b5f08ddc87831440bb0a1b4118575edca83f5d')
    original_audit = ROOT / 'dataset-500k-p3-complete-audit-v1.json'
    expected = json.loads(original_audit.read_text())['output_sha256']
    expected_path = ROOT / (PREFIX + '-expected.json')
    with expected_path.open('x') as stream:
        json.dump(expected, stream, indent=2)
    runs = OUT.with_suffix('.runs')
    runs.mkdir()
    plan = [dict(primer=True, pair=-1, position=i, workers=w) for i, w in enumerate((7, 16))]
    plan += [dict(primer=False, pair=pair, position=i, workers=w)
             for pair in range(5) for i, w in enumerate((7, 16) if pair % 2 == 0 else (16, 7))]
    STATE.update(plan=plan, planned_trials=12, planned_measured_trials=10, pairs=500000, mode='hit',
                 source=str(source), source_files=pilot_plan['source_files'], script_sha256=sha(Path(__file__)),
                 phase_audit_sha256=sha(phase_path), pilot_qualification_sha256=sha(pilot_path),
                 original_output_audit_sha256=sha(original_audit), expected_output_sha256=expected,
                 original_cache=str(cache), original_cache_sha256=cache_sha, fixture=fixture,
                 scope='Same tested native wheel, 7 versus 16 scan workers, full content-cache-hit constructor plus first batch. Two unmeasured primers and five alternating pairs. Full fixture checks before/after each child. No miss-startup, new-kernel or general-platform claim; final independent audit required.')
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    overrides = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', YOLO_OFFLINE='true')
    env.update(overrides)
    STATE['environment_overrides'] = overrides
    descriptor = None
    save()
    for index, spec in enumerate(plan):
        for name, digest in STATE['source_files'].items():
            assert sha(source / name) == digest
        out = runs / f'{index:02d}-w{spec["workers"]}.json'
        log = out.with_suffix('.log')
        command = [str(ROOT / 'dataset-cache-read-copy-m2-v1-clean/bin/python'),
                   str(source / 'bench/startup_worker_trial.py'), '--corpus', str(corpus),
                   '--harness-root', str(source), '--library-source', str(ROOT / 'dataset-cache-read-copy-m2-v1-source'),
                   '--wheel', str(ROOT / 'dataset-cache-read-copy-m2-v1-dist/ultrafast_yolo_dataset-0.1.0a1-cp312-cp312-macosx_11_0_arm64.whl'),
                   '--identity', str(ROOT / 'dataset-cache-read-copy-m2-v1-build-identity.json'),
                   '--cache-root', str(cache_roots[spec['workers']]), '--expected-outputs', str(expected_path),
                   '--out', str(out), '--workers', str(spec['workers']), '--mode', 'hit',
                   '--expected-cache-sha256', cache_sha]
        rec = dict(spec=spec, path=str(out), command=command, started_at_ns=time.time_ns())
        STATE['records'].append(rec)
        save()
        print('start', index, spec, flush=True)
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=source, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            rec['pid'] = child.pid
            save()
            host_log = out.with_suffix('.host.jsonl')
            stop = threading.Event()
            observation = dict(samples=0, errors=[])

            def observe():
                try:
                    process = psutil.Process(child.pid)
                    with host_log.open('x') as host_stream:
                        while not stop.is_set():
                            try:
                                sample = dict(at_ns=time.time_ns(), child_rss_bytes=process.memory_info().rss,
                                              child_cpu_times=process.cpu_times()._asdict(),
                                              child_threads=process.num_threads(), loadavg=os.getloadavg(),
                                              host_cpu_times=psutil.cpu_times()._asdict(),
                                              available_memory=psutil.virtual_memory().available,
                                              swap_used=psutil.swap_memory().used,
                                              disk_io=psutil.disk_io_counters()._asdict())
                            except psutil.NoSuchProcess:
                                break
                            host_stream.write(json.dumps(sample) + '\n')
                            host_stream.flush()
                            observation['samples'] += 1
                            stop.wait(1)
                except BaseException:
                    observation['errors'].append(traceback.format_exc())

            observer = threading.Thread(target=observe, daemon=True)
            observer.start()
            try:
                code = child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                rec['timed_out'] = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait(timeout=10)
            finally:
                stop.set()
                observer.join(timeout=10)
            rec['host_observation'] = dict(path=str(host_log), sha256=sha(host_log),
                                           samples=observation['samples'], errors=observation['errors'],
                                           scope='One-second whole-child samples, including pre/post fixture checks. Not isolated timing-window telemetry or an unsampled peak bound.')
            assert not observer.is_alive()
        rec.update(returncode=code, finished_at_ns=time.time_ns(), log_sha256=sha(log))
        if out.exists():
            rec['raw_sha256'] = sha(out)
        save()
        assert code == 0 and not rec.get('timed_out', False)
        assert observation['samples'] > 0 and not observation['errors']
        raw = json.loads(out.read_text())
        assert raw['complete'] and raw['passed'] and raw['workers'] == spec['workers'] and raw['mode'] == 'hit'
        assert raw['result']['workers'] == spec['workers'] and raw['result']['loader_workers'] == 0
        assert raw['result']['output_sha256'] == expected and raw['cache']['sha256'] == cache_sha
        descriptor = raw['descriptor'] if descriptor is None else descriptor
        assert raw['descriptor'] == descriptor
        rec.update(validated=True, result=raw['result'])
        save()
        print('passed', index, raw['result']['first_batch_total_s'], flush=True)
    assert sha(cache) == cache_sha
    STATE.update(complete=True, passed=True, descriptor=descriptor, finished_at_ns=time.time_ns(), final_independent_audit_passed=False)
    save()


if __name__ == '__main__':
    # Reserve before the failure handler, so an accidental second invocation
    # cannot replace an earlier completed or failed campaign.
    with OUT.open('x') as stream:
        json.dump(STATE, stream)
    try:
        main()
    except BaseException:
        STATE.update(complete=True, passed=False, error=traceback.format_exc(), finished_at_ns=time.time_ns())
        save()
        raise
