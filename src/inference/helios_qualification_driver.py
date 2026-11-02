"""Bounded native-DICOM qualification of the restored public compact reader.

The author's model and pixel source remain unchanged. This supervisor replaces
the permissive mean-filling controller with complete-coverage checks.
"""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def locate(owner, slug, sentinel):
    roots = [Path('/kaggle/input')/slug,
             Path('/kaggle/input/datasets')/owner/slug]
    matches = [p for p in roots if (p/sentinel).exists()]
    assert len(matches) == 1, f'Expected one attached {slug}: {matches}'
    return matches[0]


def validate(path, expected, labels):
    import numpy as np
    import pandas as pd
    f = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    assert list(f.columns) == ['StudyInstanceUID', *labels]
    assert f.StudyInstanceUID.is_unique and set(f.StudyInstanceUID) == set(expected)
    a = f[labels].to_numpy()
    assert np.isfinite(a).all() and ((0 <= a) & (a <= 1)).all()
    return f.set_index('StudyInstanceUID').loc[expected]


def run(modules):
    work = Path('/kaggle/working')
    lock = (work/'helios_qualification.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = Path(tempfile.mkdtemp(prefix='helios_qualification_attempt_', dir=work))
    write(work/'helios_latest_attempt.json', {'attempt_directory':attempt.name})
    source = attempt/'source'; source.mkdir()
    started = time.monotonic()
    children = []
    try:
        assert not (work/'submission.csv').exists()
        for name, content in modules.items():
            p = Path(name)
            assert not p.is_absolute() and '..' not in p.parts
            d = source/p; d.parent.mkdir(parents=True, exist_ok=True); d.write_text(content)
        contract = json.loads(modules['qualification_contract.json'])
        labels = contract['target_order']
        for name, expected in contract['source_sha256'].items():
            assert sha(source/name) == expected, name
        import torch
        assert sys.version_info[:2] == (3,13)
        assert str(torch.__version__) == contract['torch_version']
        assert torch.cuda.device_count() == 2
        for i in range(2): assert 'T4' in torch.cuda.get_device_name(i)
        compact = locate('heliosli','rsna-knee-compact-v1','manifest.json')
        code_root = locate('heliosli','rsna-orthofoundation-fold0-epoch7-20261004','receipt.json')
        decoder = locate('adityakumarrai0401','rsna-knee-decoder-cp313','decoder_identity.json')
        manifest = json.loads((compact/'manifest.json').read_text())
        assert sha(compact/'manifest.json') == contract['manifest_sha256']
        for name, expected in manifest['sha256'].items():
            assert sha(compact/name) == expected, name
        anatomy = code_root/'anatomy.pt'
        assert sha(anatomy) == manifest['anchor_weights_sha256']
        code = code_root/'cloud_code'
        for name, expected in contract['code_sha256'].items():
            assert sha(code/'training'/name) == expected, name
        assert (code/'training/vendor_runtime/monai060').is_dir()
        wheels = sorted((code_root/'dependencies/wheels').glob('*.whl'))
        assert {p.name for p in wheels} == {'pydicom-3.0.1-py3-none-any.whl','timm-1.0.25-py3-none-any.whl'}
        decoder_wheels = sorted(decoder.glob('*.whl'))
        assert len(decoder_wheels) == 3
        overlay = attempt/'deps'
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(overlay),
                        *map(str,wheels),*map(str,decoder_wheels)],check=True,timeout=180)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=os.pathsep.join(map(str,[overlay,code,source/'author'])),
                   OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MALLOC_ARENA_MAX='2')
        native_candidates = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                             Path('/kaggle/input/rsna-knee-abnormality-detection')]
        native = next(p for p in native_candidates if (p/'test.csv').exists())
        with (native/'test.csv').open() as f:
            public = [r['StudyInstanceUID'] for r in csv.DictReader(f)]
        assert len(public) == 3, 'This qualification job is for the three visible tests only'
        ids = public + contract['resource_ids'] + contract['expert_ids']
        assert len(ids) == len(set(ids))
        shim = attempt/'native_uid_only'; shim.mkdir(); (shim/'test_series').mkdir()
        import pandas as pd
        meta = []
        for split, selected in [('test',public),('train',contract['resource_ids']+contract['expert_ids'])]:
            frame = pd.read_csv(native/f'{split}_series.csv',dtype=str)
            meta.append(frame[frame.StudyInstanceUID.isin(selected)])
            for uid in selected:
                p = native/f'{split}_series'/uid
                assert p.is_dir(), uid
                (shim/'test_series'/uid).symlink_to(p, target_is_directory=True)
        pd.concat(meta,ignore_index=True).to_csv(shim/'test_series.csv',index=False)
        pd.DataFrame({'StudyInstanceUID':ids}).to_csv(shim/'test.csv',index=False)
        pd.DataFrame({'StudyInstanceUID':ids,**{k:[0]*len(ids) for k in labels}}).to_csv(shim/'sample_submission.csv',index=False)
        cap = contract['maximum_parent_seconds']
        samples = []

        def terminate_owned():
            for p in children:
                try: os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError: pass
            deadline = time.monotonic()+8
            for p in children:
                try: p.wait(timeout=max(.1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    try: os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    p.wait(timeout=10)

        def execute(name, selected, chunk, stage_cap):
            stage = attempt/name; stage.mkdir()
            loads = [0,0]; assignments = [[],[]]
            counts = {u:sum(1 for _ in (shim/'test_series'/u).glob('*/*.dcm')) for u in selected}
            assert all(counts.values())
            for uid in sorted(selected,key=lambda u:(-counts[u],u)):
                gi = min(range(2),key=lambda g:(loads[g],g))
                assignments[gi].append(uid); loads[gi] += counts[uid]
            processes, logs = [], []
            stage_start = time.monotonic()
            deadline = time.time()+min(stage_cap, cap-(stage_start-started)-45)
            assert deadline > time.time()+60
            for gi, selected_ids in enumerate(assignments):
                if not selected_ids: continue
                assignment = stage/f'worker{gi}_uids.json'; write(assignment,selected_ids)
                out = stage/f'worker{gi}'
                cmd = [sys.executable,str(source/'worker_entry.py'),'--data',str(shim),'--split','test',
                       '--code',str(code),'--models',str(compact),'--arm',manifest['arm'],
                       '--backbone',manifest['backbone'],'--size',str(manifest['size']),
                       '--crop',str(manifest['crop']),'--views',manifest['views'],
                       '--slices',str(manifest['slices']),'--phys-mm',str(manifest['phys_mm']),
                       '--eval-stride',str(manifest['eval_stride']),
                       '--state-spec',str(compact/manifest['state_spec']),
                       '--side-model',str(compact/manifest['side_model']),
                       '--side-threshold',str(manifest['side_threshold']),
                       '--side-position-mm',str(manifest['side_position_mm']),
                       '--side-max-x',str(manifest['side_max_x']),
                       '--anchor-weights',str(anatomy),'--chunk',str(chunk),'--workers','1',
                       '--assignment',str(assignment),'--output',str(out),'--deadline',str(deadline),
                       '--item-timeout','240','--dump-logits',str(stage/f'worker{gi}_logits.json')]
                log = (stage/f'worker{gi}.log').open('w'); logs.append(log)
                p = subprocess.Popen(cmd,env=dict(env,CUDA_VISIBLE_DEVICES=str(gi)),
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(p); processes.append((p,out,selected_ids))
            last_sample = 0.
            try:
                while any(p.poll() is None for p,_,_ in processes):
                    if time.monotonic()-stage_start > stage_cap or time.monotonic()-started > cap-30:
                        raise TimeoutError(f'Owned {name} processes exceeded fixed resource cap')
                    if any(p.poll() not in (None,0) for p,_,_ in processes):
                        raise RuntimeError(f'{name} worker failed; inspect preserved worker logs')
                    if time.monotonic()-last_sample > 30:
                        r = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                            '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
                        samples.append({'seconds':time.monotonic()-started,'stage':name,'gpu':r.stdout.strip()})
                        write(attempt/'resource_samples.json',samples)
                        print(json.dumps({'stage':name,'elapsed_seconds':round(time.monotonic()-stage_start),
                                          'worker_exit_codes':[p.poll() for p,_,_ in processes]}),flush=True)
                        last_sample = time.monotonic()
                    time.sleep(5)
                frames, receipts = [], []
                for p,out,selected_ids in processes:
                    assert p.returncode == 0
                    r = json.loads((out/'receipt.json').read_text())
                    assert r['status']=='PASS' and not r['fallback'] and not r['timed_out']
                    assert r['studies']==r['scored']==len(selected_ids)
                    assert r['models']==manifest['models'] and r['views']==manifest['views']
                    assert r['anchor_weights_sha256']==manifest['anchor_weights_sha256']
                    assert r['side_check']['model_sha256']==manifest['sha256'][manifest['side_model']]
                    for filename,h in r['sha256'].items(): assert h==contract['source_sha256']['author/'+filename]
                    frames.append(validate(out/'predictions.csv',selected_ids,labels)); receipts.append(r)
                f = pd.concat(frames).loc[selected]
                f.reset_index().to_csv(stage/'predictions.csv',index=False)
                write(stage/'execution.json',{'status':'COMPLETE_ZERO_FALLBACK_NATIVE_PREDICTIONS',
                                             'studies':len(selected),'chunk':chunk,
                                             'elapsed_seconds':time.monotonic()-stage_start,
                                             'prediction_sha256':sha(stage/'predictions.csv'),
                                             'worker_receipts_sha256':[sha(out/'receipt.json') for _,out,_ in processes]})
                return f, receipts
            finally:
                terminate_owned()
                for f in logs: f.close()

        pilot, receipts = execute('resource_pilot',public+contract['resource_ids'],64,900)
        replay, _ = execute('public_replay',public,32,600)
        drift = float(abs(pilot.loc[public].to_numpy()-replay.loc[public].to_numpy()).max())
        assert drift <= contract['prediction_replay_max_absolute_error'], drift
        write(attempt/'pilot_review.json',{'status':'PASS_NATIVE_PILOT_AND_CHUNK_REPLAY',
                                          'prediction_max_abs_error':drift,
                                          'absolute_tolerance':contract['prediction_replay_max_absolute_error'],
                                          'independent_clinical_validation':False})
        clinical, _ = execute('expert_diagnostic',contract['expert_ids'],64,
                              max(60,cap-(time.monotonic()-started)-60))
        write(attempt/'execution_receipt.json',{
            'status':'COMPLETE_BOUNDED_HELIOS_NATIVE_QUALIFICATION',
            'contract_sha256':sha(source/'qualification_contract.json'),
            'elapsed_seconds':time.monotonic()-started,'resource_studies':len(pilot),
            'expert_studies':len(clinical),'public_replay_max_abs_error':drift,
            'expert_predictions_sha256':sha(attempt/'expert_diagnostic/predictions.csv'),
            'manifest_sha256':sha(compact/'manifest.json'),
            'model_sha256':manifest['sha256'],'code_sha256':contract['code_sha256'],
            'torch_version':str(torch.__version__),'python':sys.version,
            'source_files_sha256':{n:sha(source/n) for n in modules},
            'offline_wheel_sha256':{p.name:sha(p) for p in wheels+decoder_wheels},
            'gold_labels_or_reports_read_by_inference':False,
            'test_time_fitting':False,'competition_submission_enabled':False})
        print(json.dumps({'status':'COMPLETE_BOUNDED_HELIOS_NATIVE_QUALIFICATION',
                          'expert_studies':len(clinical),'elapsed_seconds':time.monotonic()-started}),flush=True)
    except BaseException as e:
        write(attempt/'failure.json',{'status':'FAILED_NO_SUBMISSION','error':repr(e),
                                     'traceback':traceback.format_exc(),
                                     'elapsed_seconds':time.monotonic()-started})
        raise
    finally:
        for p in children:
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
        for p in children:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try: os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                p.wait(timeout=10)
        if (attempt/'native_uid_only').exists():
            shutil.rmtree(attempt/'native_uid_only')
        lock.close()
