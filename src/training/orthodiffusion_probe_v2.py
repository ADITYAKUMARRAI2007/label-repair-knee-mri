"""V2: short-volume repair and persistent feature recovery; fixed comparison unchanged."""
import gc
import atexit
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import diffusion_probe_support as support
from diffusion_probe_support import find,table,sha,guard,OUT,START,DEADLINE
from orthodiffusion_encoder import load_encoder,feature_with_noise
from orthodiffusion_volume_v2 import series_noise_seed
from orthodiffusion_decode_worker_v2 import initialize,decode
from feature_checkpoint import save_checkpoint,row_digest
from orthodiffusion_fit_worker import evict
from e03_split_contract import freeze_split

BASE=Path(__file__).resolve().parent
CONTRACT=json.loads((BASE/'contract.json').read_text())

def write(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')

def fit_processes(scratch,pilot=False):
    processes=[];begin=time.monotonic()
    try:
        for device in range(2):
            command=[sys.executable,str(BASE/'orthodiffusion_fit_worker.py'),'--scratch',str(scratch),
                     '--device',str(device),'--deadline',str(DEADLINE)]
            if pilot:command.append('--pilot')
            processes.append(subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device))))
        if pilot:
            while not all((scratch/f'ready_{d}').exists() for d in range(2)):
                guard()
                if any(p.poll() is not None for p in processes):raise RuntimeError('pilot failed before barrier')
                time.sleep(.2)
            (scratch/'pilot_go').touch()
        while any(p.poll() is None for p in processes):
            guard()
            if any(p.poll() not in {None,0} for p in processes):raise RuntimeError('head process failed')
            time.sleep(1)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('head process failed')
        return time.monotonic()-begin
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()

def decoded(executor,indices,ids):
    pending={};items=iter(indices)
    def enqueue():
        try:row=int(next(items))
        except StopIteration:return False
        pending[executor.submit(decode,(row,str(ids[row])))]=row
        return True
    for _ in range(8):
        if not enqueue():break
    while pending:
        guard();done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
        for future in done:
            pending.pop(future)
            yield future.result()
            enqueue()

def main():
    torch.set_num_threads(2)
    assert torch.cuda.device_count()==2
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    for name,expected in CONTRACT['source_files'].items():assert sha(BASE/'author'/name)==expected
    assert sha(BASE/'orthodiffusion_encoder.py')==CONTRACT['adapter_sha256']
    assert sha(BASE/'orthodiffusion_volume.py')==CONTRACT['original_volume_sha256']
    assert sha(BASE/'orthodiffusion_volume_v2.py')==CONTRACT['volume_sha256']
    records,official=table(find('train.csv',CONTRACT['train_sha256']))
    series_csv=find('train_series.csv',CONTRACT['train_series_sha256'])
    series_root=series_csv.parent/'train_series';assert series_root.is_dir()
    split=freeze_split(records,CONTRACT['targets'])
    excluded=set(split['gold'])|set(split['excluded_gold_group_weak'])
    training=sorted(set(official)-excluded)
    assert hashlib.sha256(json.dumps(training).encode()).hexdigest()==CONTRACT['train_ids_sha256']
    sources=[table(find(name,expected))[1] for name,expected in CONTRACT['label_sources'].items()]
    previous=CONTRACT['previous_features']
    ids=np.load(find('study_ids.npy',previous['study_ids.npy']['sha256'])).astype(str)
    assert len(ids)==len(set(ids))==4407 and set(ids)==set(official)
    old_mask=np.load(find('token_masks.npy',previous['token_masks.npy']['sha256']))
    old_path=find('features.npy',previous['features.npy']['sha256'])
    control=np.load(old_path,mmap_mode='r')
    assert control.shape==(4407,20,2048) and old_mask.shape==(4407,20)
    train=np.array([i for i,u in enumerate(ids) if u not in excluded],dtype=np.int64)
    expert=np.array([i for i,u in enumerate(ids) if u in set(split['gold'])],dtype=np.int64)
    assert len(train)==4345 and len(expert)==58
    labels=np.zeros((4407,12),np.float32);weights=labels.copy()
    for row in train:
        for j,t in enumerate(CONTRACT['targets']):
            values=np.array([float(s[ids[row]][t]) for s in sources],np.float64)
            assert np.isfinite(values).all() and np.all((values>=0)&(values<=1))
            labels[row,j]=np.clip(np.median(values),.01,.99)
            weights[row,j]=np.clip(1-np.ptp(values),.25,1)
    pilot_ids=np.asarray(sorted(train,key=lambda i:hashlib.sha256(f'20261017:{ids[i]}'.encode()).digest())[:128])
    extra=np.array([int(np.flatnonzero(ids==uid)[0]) for uid in CONTRACT['preflight']['additional_weak_short_studies']],dtype=np.int64)
    pilot_ids=np.concatenate([pilot_ids,extra[~np.isin(extra,pilot_ids)]])
    assert len(pilot_ids)==CONTRACT['preflight']['studies'] and set(pilot_ids).issubset(set(train))
    assert not set(pilot_ids)&set(expert)
    scratch=Path('/tmp/orthodiffusion_probe_features_v2');scratch.mkdir(exist_ok=False)
    assert shutil.disk_usage(OUT).free>4*2**30
    os.symlink(old_path,scratch/'control.npy')
    feature_path=OUT/'diffusion_features.npy'
    assert not feature_path.exists(),'do not overwrite prior feature cache'
    features=np.lib.format.open_memmap(feature_path,mode='w+',dtype=np.float16,shape=(4407,3,256,256))
    os.symlink(feature_path,scratch/'candidate.npy')
    masks=np.zeros((4407,3),bool);complete=np.zeros(4407,bool)
    digests=np.zeros(4407,dtype='S64')
    np.save(OUT/'study_ids.npy',ids)
    def commit_checkpoint():
        path=OUT/'feature_checkpoint.npz'
        save_checkpoint(path,features,ids,masks,complete,digests,CONTRACT['feature_contract_sha256'])
        write('feature_checkpoint_summary.json',dict(completed_rows=int(complete.sum()),total_rows=4407,
              feature_key=CONTRACT['feature_contract_sha256'],checkpoint_sha256=sha(path),
              feature_shape=list(features.shape),seconds=time.monotonic()-START))
    commit_checkpoint();atexit.register(commit_checkpoint)
    for name,array in dict(labels=labels,weights=weights,train=train,expert=expert,pilot_ids=pilot_ids,
                           control_mask=old_mask,candidate_mask=masks).items():np.save(scratch/(name+'.npy'),array)
    (scratch/'fit_config.json').write_text(json.dumps({'seeds':CONTRACT['seeds']}))
    write('contract.json',CONTRACT)
    encoders=[]
    for plane in ['axial','coronal','sagittal']:
        identity=CONTRACT['checkpoints'][plane]
        encoders.append(load_encoder(BASE/'author',find(identity['file']),identity['sha256']).cuda())
    cpu=np.load(BASE/'cpu_fixture.npz')
    with torch.inference_mode():
        fixture=feature_with_noise(encoders[0],torch.from_numpy(cpu['volume']).cuda(),
                                   torch.from_numpy(cpu['noise']).cuda()).cpu().numpy()
    error=float(np.max(np.abs(fixture-cpu['feature'])))
    assert error<=.003;write('encoder_fixture.json',{'cpu_gpu_fp32_max_abs_error':error})
    receipts_path=OUT/'volume_receipts.jsonl'
    def extract(item,replay=False):
        row=item['row'];mask=item['mask'];volume=item['volume'];receipt=item['receipt']
        values=np.zeros((3,256,256),np.float16)
        with torch.inference_mode():
            for plane in np.flatnonzero(mask):
                r=receipt['planes'][plane]
                x=torch.from_numpy(volume[plane:plane+1,None]).cuda()
                gen=torch.Generator(device='cpu').manual_seed(series_noise_seed(item['uid'],r['SeriesInstanceUID']))
                noise=torch.randn(x.shape,generator=gen).cuda()
                value=feature_with_noise(encoders[plane],x,noise,amp=True)
                assert value.shape==(1,256,1,16,16) and torch.isfinite(value).all()
                assert value.abs().max()<65000
                values[plane]=value[0,:,0].permute(1,2,0).reshape(256,256).half().cpu().numpy()
        if replay:
            assert np.array_equal(values,features[row]) and np.array_equal(mask,masks[row])
        else:
            assert not complete[row]
            features[row]=values;masks[row]=mask;digests[row]=row_digest(values,mask);complete[row]=True
            with receipts_path.open('a') as f:f.write(json.dumps(dict(row=row,decode_seconds=item['seconds'],**receipt))+'\n')
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=4,mp_context=context,initializer=initialize,
                             initargs=(str(series_csv),str(series_root))) as executor:
        failures=[];early=time.monotonic();saved=set(map(int,pilot_ids[:3]));short_checks={}
        for item in decoded(executor,pilot_ids,ids):
            if item['error']:
                failures.append(item);continue
            extract(item)
            if item['uid'] in CONTRACT['preflight']['additional_weak_short_studies']:
                planes=item['receipt']['planes']
                assert len(planes)==3 and all(p['state']=='decoded' and p['native_depth']==15
                    and p['edge_padding_used'] and p['effective_depth']==16 for p in planes)
                short_checks[item['uid']]=dict(native_depths=[p['native_depth'] for p in planes],
                                              present_planes=int(item['mask'].sum()))
            if item['row'] in saved:
                np.savez_compressed(OUT/f"volume_fixture_{item['row']}.npz",volume=item['volume'],mask=item['mask'])
            if complete.sum()%32==0:
                commit_checkpoint()
                print(json.dumps(dict(stage='real_volume_preflight',complete=int(complete.sum()),seconds=time.monotonic()-START)),flush=True)
        early_seconds=time.monotonic()-early
        preflight=dict(status='PASS' if not failures else 'REJECT_REAL_DATA_DECODER',
                       pilot_studies=len(pilot_ids),successful_studies=int(complete.sum()),failures=failures,
                       present_planes=masks[pilot_ids].sum(0).tolist(),decode_extract_seconds=early_seconds,
                       pilot_expert_rows=0,additional_short_study_checks=short_checks)
        write('real_volume_preflight.json',preflight)
        if failures:raise RuntimeError(f'real-data decoder failed for{len(failures)} of{len(pilot_ids)} weak-only studies')
        assert set(short_checks)==set(CONTRACT['preflight']['additional_weak_short_studies'])
        for item in decoded(executor,pilot_ids[:3],ids):
            assert not item['error'];extract(item,replay=True)
        preflight['three_study_feature_replay_max_error']=0;write('real_volume_preflight.json',preflight)
        commit_checkpoint();evict(features);np.save(scratch/'candidate_mask.npy',masks)
        pilot_wall=fit_processes(scratch,pilot=True)
        pilot={str(d):json.loads((OUT/f'pilot_device_{d}.json').read_text()) for d in range(2)}
        projected_devices={str(d):sum(pilot[str(d)][arm]['p95_complete_step_seconds']*68*30
                           for arm in ['control','candidate'])*len(CONTRACT['seeds'][d::2])*1.2 for d in range(2)}
        remaining=early_seconds*(4407-len(pilot_ids))/len(pilot_ids)*1.2
        startup=max(60,pilot_wall);heads=max(projected_devices.values())
        projected=time.monotonic()-START+remaining+heads+startup+240
        resource=dict(status='PASS' if projected<6000 else 'REJECT_COMPLETE_RECIPE_DOES_NOT_FIT',
            early_feature_seconds=early_seconds,early_feature_studies=len(pilot_ids),pilot=pilot,
            projected_remaining_extraction_seconds=remaining,projected_device_head_seconds=projected_devices,
            projected_all_head_seconds=heads,pilot_wall_seconds=pilot_wall,process_startup_margin_seconds=startup,
            finalization_margin_seconds=240,projected_worker_elapsed_seconds=projected,worker_deadline_seconds=6000,
            elapsed_at_projection_seconds=time.monotonic()-START,pilot_expert_rows=0)
        write('resource_pilot.json',resource);print(json.dumps(dict(stage='resource_gate',**resource)),flush=True)
        if resource['status']!='PASS':raise RuntimeError('complete fixed recipe exceeds resource cap')
        for item in decoded(executor,np.flatnonzero(~complete),ids):
            if item['error']:
                write('decode_failure.json',item);raise RuntimeError('DICOM decode failed after preflight')
            extract(item)
            if complete.sum()%64==0:
                commit_checkpoint();evict(features);guard()
                print(json.dumps(dict(stage='features',complete=int(complete.sum()),seconds=time.monotonic()-START)),flush=True)
    assert complete.all()
    commit_checkpoint();atexit.unregister(commit_checkpoint)
    del encoders;gc.collect();torch.cuda.empty_cache()
    features.flush();evict(features);np.save(scratch/'candidate_mask.npy',masks)
    for name,value in dict(study_ids=ids,expert_ids=ids[expert],train_indices=train,expert_indices=expert,
                           pilot_indices=pilot_ids,labels=labels,weights=weights,candidate_mask=masks,
                           control_mask=old_mask,expert_features=np.asarray(features[expert])).items():np.save(OUT/(name+'.npy'),value)
    fit_processes(scratch)
    results=[json.loads((OUT/f'{arm}_{seed}_fit.json').read_text()) for seed in CONTRACT['seeds'] for arm in ['control','candidate']]
    truth=np.asarray([[float(official[ids[i]][t]) for t in CONTRACT['targets']] for i in expert])
    for result in results:
        pred=np.load(OUT/f"{result['arm']}_{result['seed']}_expert.npy")
        aucs={t:float(roc_auc_score(truth[:,j],pred[:,j])) for j,t in enumerate(CONTRACT['targets'])}
        result.update(expert_per_target_auc=aucs,expert_macro_auc=float(np.mean(list(aucs.values()))))
    evict(features);del features,control
    # Persistent feature cache is already in the output directory.
    shutil.rmtree(scratch)
    artifacts={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.iterdir() if p.is_file()}
    manifest=dict(status='COMPLETE',contract=CONTRACT,results=results,resource_pilot=resource,
                  real_volume_preflight=preflight,elapsed_seconds=time.monotonic()-START,
                  artifacts=artifacts,all_expert_cases_excluded_from_training=True,
                  competition_submission_enabled=False,public_score=None)
    write('manifest.json',manifest)
    print(json.dumps(dict(stage='complete',scores=[{k:r[k] for k in ['seed','arm','expert_macro_auc']} for r in results])),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        write('failure.json',dict(status='INCOMPLETE_INELIGIBLE',error_type=type(error).__name__,
                                 message=str(error),elapsed_seconds=time.monotonic()-START,
                                 host_memory=support.host_memory()))
        raise
