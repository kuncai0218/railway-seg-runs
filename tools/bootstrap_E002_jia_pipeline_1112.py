"""Jia's authorized foldI: pinned receiver -> verified data -> admission -> training."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import psutil

BASE = 'https://raw.githubusercontent.com/kuncai0218/railway-seg-runs/main/'
KIT_RECEIPT_SHA = '85e65194cacf0fd6e0f86a577e16c6e11a12bb4535e320ae27842a49be84129d'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(relative, limit=2**20):
    assert not relative.startswith('/') and '..' not in Path(relative).parts
    with urllib.request.urlopen(BASE + relative, timeout=20) as r:
        data = r.read(limit + 1)
    assert len(data) <= limit
    return data


def install(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, f'existing different file retained: {path}'
    else:
        temporary = path.with_name(path.name + f'.partial_{os.getpid()}')
        temporary.write_bytes(data)
        os.rename(temporary, path)


def main():
    ap = argparse.ArgumentParser()
    for name in ['work', 'bind', 'sender', 'token']:
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args()
    root = Path(a.work).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'E002_foldI_pipeline.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('existing foldI pipeline owns lock; no duplicate launched', flush=True)
        return
    state_path = root / 'E002_foldI_pipeline_progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def progress(stage, **values):
        state.update(time=dt.datetime.now().astimezone().isoformat(), stage=stage,
                     pipeline_PID=os.getpid(), pipeline_created=psutil.Process().create_time(), **values)
        temporary = state_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        os.replace(temporary, state_path)
        print(json.dumps(state), flush=True)

    def identity(pid, script):
        if not isinstance(pid, int) or pid <= 0:
            return False, None
        try:
            p = psutil.Process(pid)
            return str(script) in p.cmdline(), p.create_time()
        except psutil.Error:
            return False, None

    try:
        # If a prior supervisor ended while its trainer survived, preserve it.
        trainer_script = root / 'author_foldI/e002_train.py'
        active, created = identity(state.get('trainer_PID', -1), trainer_script)
        if active and created == state.get('trainer_created'):
            progress('watching_existing_formal_trainer')
            psutil.Process(state['trainer_PID']).wait()
            return
        for _ in range(180):
            try:
                publication = json.loads(fetch('data/E002_DEN_source55_publication_1112.json'))
                assert publication['complete'] and publication['archive_count'] == 55
                assert publication['kit_receipt_SHA256'] == KIT_RECEIPT_SHA
                break
            except urllib.error.HTTPError as error:
                if error.code not in [404, 429, 500, 502, 503, 504]:
                    raise
                progress('waiting_for_published_manifest', bulk_bytes_received=0)
                time.sleep(20)
            except (urllib.error.URLError, TimeoutError):
                progress('retrying_small_manifest_request', bulk_bytes_received=0)
                time.sleep(20)
        else:
            raise TimeoutError('published manifest not ready in one hour')
        manifest_data = fetch('data/E002_DEN_source55_manifest_1112.json')
        assert digest(manifest_data) == publication['manifest_SHA256']
        manifest_path = root / 'DEN_source55_manifest_1112.json'
        install(manifest_path, manifest_data)
        manifest = json.loads(manifest_data)
        kit = root / 'author_foldI'
        receipt_data = fetch('packages/e002_author_foldI_1112/portable_kit_receipt.json')
        assert digest(receipt_data) == KIT_RECEIPT_SHA
        install(kit / 'portable_kit_receipt.json', receipt_data)
        code_receipt = json.loads(receipt_data)
        for relative, expected in code_receipt['file_SHA256'].items():
            if relative.startswith('packages/e002_author_foldI_1112/'):
                target = kit / Path(relative).name
            elif relative in ['tools/receive_DEN_source55_1112.py', 'tools/prepare_DEN_author_source55_1112.py']:
                target = root / Path(relative).name
            else:
                raise ValueError('unexpected code receipt path')
            content = fetch(relative)
            assert digest(content) == expected
            install(target, content)
        incoming = root / 'incoming_source55'
        incoming.mkdir(exist_ok=True)
        receiver_script = root / 'receive_DEN_source55_1112.py'
        active, created = identity(state.get('receiver_PID', -1), receiver_script)
        if not (active and created == state.get('receiver_created')):
            log = (root / 'DEN_receiver_41987.log').open('ab')
            receiver = subprocess.Popen([sys.executable, '-u', str(receiver_script),
                '--bind', a.bind, '--sender', a.sender, '--token', a.token,
                '--manifest', str(manifest_path), '--manifest-sha', publication['manifest_SHA256'],
                '--out', str(incoming), '--port', '41987'], stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            time.sleep(1)
            assert receiver.poll() is None, 'receiver failed; inspect DEN_receiver_41987.log'
            state.update(receiver_PID=receiver.pid, receiver_created=psutil.Process(receiver.pid).create_time())
        progress('receiving_verified_public_source55', manifest_SHA256=publication['manifest_SHA256'])
        for _ in range(2160):
            completed = 0
            for record in manifest['archives']:
                receipt = incoming / (record['file'] + '.receipt.json')
                if receipt.exists():
                    r = json.loads(receipt.read_text())
                    assert r['accepted'] and r['SHA256'] == record['SHA256'] and r['bytes'] == record['bytes']
                    completed += 1
            if completed == 55:
                break
            progress('receiving_verified_public_source55', completed_archives=completed, total_archives=55)
            time.sleep(10)
        else:
            raise TimeoutError('data receipt incomplete after six hours; inspect transfer')
        env = dict(os.environ)
        raw, cache = root / 'den_raw_source55', root / 'den_cache_foldI/train'
        env.update(E002_AUTHOR_REPO=str(root / 'SitsSCD'), E002_RAW_ROOT=str(raw), E002_SSD_ROOT=str(cache),
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')

        def run_phase(stage, command, log_name):
            progress(stage)
            with (root / log_name).open('ab') as output:
                process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
                progress(stage, phase_PID=process.pid, phase_created=psutil.Process(process.pid).create_time())
                code = process.wait()
            assert code == 0, f'{stage} failed with{code}; inspect{log_name}'

        run_phase('verify_unpack_and_build_exact_fold_cache', [sys.executable, '-u',
            str(root / 'prepare_DEN_author_source55_1112.py'), '--manifest', str(manifest_path),
            '--manifest-sha', publication['manifest_SHA256'], '--incoming', str(incoming),
            '--raw', str(raw), '--cache', str(cache), '--protocol', str(kit / 'protocol.json'),
            '--split-metadata', str(kit / 'source55_original_split_metadata.json')], 'DEN_prepare.log')
        admission = kit / 'admission_accepted.json'
        if not admission.exists():
            run_phase('actual_DEN_native_data_and_GPU_optimizer_admission',
                      [sys.executable, '-u', str(trainer_script), '--admission'], 'DEN_admission.log')
        record = json.loads(admission.read_text())
        assert record['passed'] and record['weights_discarded_after_admission']
        out = root / 'runs/E002_DEN_author_spatial_I_s42_1112'
        command = [sys.executable, '-u', str(trainer_script), '--micro', '2', '--out', str(out)]
        with (root / 'DEN_foldI_training.log').open('ab') as log:
            trainer = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            progress('formal_training_running', trainer_PID=trainer.pid,
                     trainer_created=psutil.Process(trainer.pid).create_time(), train_out=str(out),
                     formal_command=command, phase_PID=None)
            returncode = trainer.wait()
        assert returncode == 0
        result = json.loads((out / 'final.json').read_text())
        assert result['optimizer_updates'] == 500000 and not result['internal_test_evaluated']
        progress('completed', final=result)
    except Exception as error:
        progress('failed', error=repr(error), existing_results_retained=True)
        raise


if __name__ == '__main__':
    main()
