"""Run exactly one full fold, preserve snapshots, then release the Colab runtime."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import traceback

import e05_fetch_assets as assets
from e05_upload_blob import upload

SETUP=Path('/content/e05_full_setup')
OUTPUT=Path('/content/e05_full_results')


def main():
    started=time.time()
    job=json.loads((SETUP/'job.json').read_text())
    assert job['fold'] in range(5) and job['arm']=='uniform'
    assert job['training_hours_cap']==2.0 and job['competition_submission_enabled'] is False
    OUTPUT.mkdir(exist_ok=False)
    assets.SETUP=SETUP;assets.START=time.monotonic()
    access=json.loads((SETUP/'download_access.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        receipts=list(pool.map(assets.fetch,access))
    (SETUP/'download_access.json').unlink()
    (OUTPUT/'assets.json').write_text(json.dumps(receipts,indent=2)+'\n')
    for name,expected in job['files_sha256'].items():
        assert assets.sha(SETUP/name)==expected,name
    os.environ['E04_TRAIN_FOLD']=str(job['fold']);os.environ['E05_ARM']=job['arm']
    spec=importlib.util.spec_from_file_location('e05_full_definitions',SETUP/'worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Each downloaded byte has already been hashed. Bind the file paths/sizes
    # to those receipts instead of reading the entire 22GB a second time.
    actual=module.discover_wide_files();by_name={r['name']:r for r in receipts}
    for name,item in actual.items():
        assert item['sha256']==by_name[name]['sha256'] and item['bytes']==by_name[name]['bytes']
    verified={'schema':'e04_mounted_cache_verification_v1','status':'PASS','files':actual}
    verification=OUTPUT/'cache_verification.json'
    verification.write_text(json.dumps(verified,indent=2)+'\n')
    train_out=OUTPUT/'training'
    env=os.environ.copy();env.update(E04_TRAIN_OUTPUT=str(train_out),
        E04_PARENT_DEADLINE=str(time.time()+7200-300),
        E04_CACHE_VERIFICATION_PATH=str(verification),
        E04_CACHE_VERIFICATION_SHA256=assets.sha(verification),
        E05_UPLOAD_LEASES=str(SETUP/'upload_leases_private.json'),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    print(json.dumps({'stage':'FULL_FOLD_START','fold':job['fold'],'training_seconds_cap':7200}),flush=True)
    with (OUTPUT/'training.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-u',str(SETUP/'worker.py')],env=env,
                                 stdout=log,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise RuntimeError('Full fold reached runtime cap') from None
    if code!=0:raise RuntimeError('Full worker failed; inspect uploaded snapshots and saved log')
    manifest=train_out/f'e05_uniform_fold{job["fold"]}_manifest.json'
    record=json.loads(manifest.read_text())
    final=json.loads((train_out/'final_durable_upload.json').read_text())
    assert final['status']=='BUNDLE_AND_RECEIPT_UPLOADED'
    result={'status':record['status'],'completed_epoch':record['completed_epoch'],
            'fold':job['fold'],'elapsed_seconds':time.time()-started,
            'final_snapshot_uploaded':True,'competition_submission_enabled':False}
    (OUTPUT/'job_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    terminal={'status':'WORKER_FINISHED'}
    try:main()
    except Exception as error:
        terminal={'status':'FULL_JOB_FAILED','error_type':type(error).__name__,
                  'error_frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}
                                  for f in traceback.extract_tb(error.__traceback__)]}
        print(json.dumps(terminal),flush=True)
    finally:
        preserved=False
        try:
            leases=json.loads((SETUP/'upload_leases_private.json').read_text())
            lease=leases['snapshots']['terminal']['receipt']
            log=OUTPUT/'training.log'
            terminal['training_log_tail']=log.read_text()[-24000:] if log.exists() else ''
            supervisor_log=SETUP/'supervisor.log'
            terminal['supervisor_log_tail']=supervisor_log.read_text()[-12000:] if supervisor_log.exists() else ''
            result=OUTPUT/'job_result.json'
            if result.exists():terminal['job_result']=json.loads(result.read_text())
            path=SETUP/lease['name'];data=(json.dumps(terminal)+'\n').encode()
            assert len(data)<lease['bytes']
            path.write_bytes(data+b' '*(lease['bytes']-len(data)))
            upload(path,{**lease,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},SETUP/'terminal_upload.json')
            preserved=True
        except Exception as error:
            print(json.dumps({'terminal_export_failed':type(error).__name__}),flush=True)
        # Keep failed-transfer diagnostics accessible until the outer release
        # timer if preservation itself failed. Never silently erase them.
        if preserved:
            from google.colab import runtime
            try:runtime.unassign()
            except Exception:pass
