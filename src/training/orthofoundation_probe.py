"""Bounded paired knee-pretrained encoder probe; fixed labels and head recipe."""
import gc
import hashlib
import json
import math
import mmap
import os
import subprocess
import sys
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from spatial_pooling_head import SpatialHead
from ortho_probe_support import find, table, sha, guard, host_memory, OUT, START, DEADLINE
from orthofoundation_encoder import OrthoEncoder, normalize_pixels
from e03_split_contract import freeze_split
from e03_window_contract import select_slot_windows

# __PROBE_CONTRACT__


def evict(array):
    if hasattr(array,'_mmap') and hasattr(array._mmap,'madvise'):
        array._mmap.madvise(mmap.MADV_DONTNEED)


def head_step(model,optimizer,features,mask,labels,weights,indices):
    x=torch.from_numpy(np.asarray(features[indices])).cuda()
    m=torch.from_numpy(mask[indices]).cuda()
    y=torch.from_numpy(labels[indices]).cuda()
    w=torch.from_numpy(weights[indices]).cuda()
    optimizer.zero_grad(set_to_none=True)
    loss=(F.binary_cross_entropy_with_logits(model(x,m),y,reduction='none')*w).mean()
    if not torch.isfinite(loss):raise RuntimeError('nonfinite head loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
    if not torch.isfinite(norm):raise RuntimeError('nonfinite gradients')
    optimizer.step()
    return float(loss.detach())


def fit_processes(scratch,pilot=False):
    processes=[];before=time.monotonic()
    try:
        for device in range(2):
            command=[sys.executable,str(Path(__file__).with_name('orthofoundation_fit_worker.py')),
                     '--scratch',str(scratch),'--device',str(device),'--deadline',str(DEADLINE)]
            if pilot:command.append('--pilot')
            processes.append(subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device))))
        if pilot:
            while not all((scratch/f'ready_{d}').exists() for d in range(2)):
                guard()
                if any(p.poll() is not None for p in processes):raise RuntimeError('pilot process failed before barrier')
                time.sleep(.2)
            (scratch/'pilot_go').touch()
        while any(p.poll() is None for p in processes):
            guard()
            if any(p.poll() not in {None,0} for p in processes):raise RuntimeError('independent fit process failed')
            time.sleep(1)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('independent fit process failed')
        return time.monotonic()-before
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()


def main():
    torch.set_num_threads(2)
    if torch.cuda.device_count()!=2:raise RuntimeError('two T4s required')
    records,official=table(find('train.csv',CONTRACT['train_sha256']))
    split=freeze_split(records,CONTRACT['targets'])
    excluded=set(split['gold'])|set(split['excluded_gold_group_weak'])
    training=sorted(set(official)-excluded)
    if hashlib.sha256(json.dumps(training).encode()).hexdigest()!=CONTRACT['train_ids_sha256']:
        raise RuntimeError('training cohort/exclusions changed')
    sources=[table(find(name,expected))[1] for name,expected in CONTRACT['label_sources'].items()]
    arrays=[];masks=[];ordered=[]
    for part,expected in CONTRACT['cache'].items():
        manifest=json.loads(find(f'e04_wide_{part}_manifest.json',expected).read_text())
        if manifest['status']!='COMPLETE' or manifest['span']!=[.06,.94]:raise RuntimeError('cache contract drift')
        mounted={}
        for name,receipt in manifest['artifacts'].items():
            p=find(name)
            if p.stat().st_size!=receipt['bytes'] or sha(p)!=receipt['sha256']:raise RuntimeError('cache bytes differ')
            mounted[name]=p
        arrays.append(np.load(mounted[f'e04_wide_{part}_vols.npy'],mmap_mode='r'))
        masks.append(np.load(mounted[f'e04_wide_{part}_masks.npy'],allow_pickle=False))
        ordered.extend(np.load(mounted[f'e04_wide_{part}_ids.npy'],allow_pickle=False).astype(str).tolist())
    masks=np.concatenate(masks);ids=np.asarray(ordered)
    if len(set(ids))!=4407 or set(ids)!=set(official):raise RuntimeError('cache identity drift')
    old_ids=np.load(find('study_ids.npy',CONTRACT['previous_features']['study_ids.npy']['sha256']),allow_pickle=False)
    old_mask=np.load(find('token_masks.npy',CONTRACT['previous_features']['token_masks.npy']['sha256']),allow_pickle=False)
    old_mean=np.load(find('features.npy',CONTRACT['previous_features']['features.npy']['sha256']),mmap_mode='r')
    if not np.array_equal(ids,old_ids):raise RuntimeError('reference feature row order differs')
    labels=np.zeros((4407,12),np.float32);weights=labels.copy()
    for row,uid in enumerate(ids):
        if uid in excluded:continue
        for j,target in enumerate(CONTRACT['targets']):
            values=np.asarray([float(source[uid][target]) for source in sources],np.float64)
            if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise RuntimeError('invalid weak labels')
            labels[row,j]=np.clip(np.median(values),.01,.99)
            weights[row,j]=np.clip(1-np.ptp(values),.25,1)
    train=np.asarray([i for i,u in enumerate(ids) if u not in excluded],np.int64)
    expert=np.asarray([i for i,u in enumerate(ids) if u in set(split['gold'])],np.int64)
    if len(train)!=4345 or len(expert)!=58:raise RuntimeError('cohort counts differ')
    scratch=Path('/tmp/rsna_orthofoundation_probe_features');scratch.mkdir(exist_ok=False)
    if shutil.disk_usage(scratch).free<3*(1<<30):raise RuntimeError('insufficient scratch space')
    os.symlink(find('features.npy',CONTRACT['previous_features']['features.npy']['sha256']),scratch/'control.npy')
    features={'control':old_mean,
              'candidate':np.lib.format.open_memmap(scratch/'candidate.npy',mode='w+',dtype=np.float16,shape=(4407,20,2048))}
    token_mask=np.zeros((4407,20),bool)
    bundle=find('dinov3_source.json',CONTRACT['assets']['dinov3_source.json']['sha256'])
    source=scratch/'architecture';source.mkdir()
    for name,code in json.loads(bundle.read_text()).items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise RuntimeError('invalid architecture source path')
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(code)
    sys.path.insert(0,str(source))
    checkpoint=find('OrthoFoudation-L.pth',CONTRACT['assets']['OrthoFoudation-L.pth']['sha256'])
    fixture=np.load(find('orthofoundation_cpu_fixture.npz',CONTRACT['assets']['orthofoundation_cpu_fixture.npz']['sha256']))
    encoders=[];parity=[]
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for device in range(2):
        with torch.cuda.device(device),torch.inference_mode():
            model=OrthoEncoder(checkpoint).to(f'cuda:{device}')
            image=normalize_pixels(torch.from_numpy(fixture['pixels']).to(f'cuda:{device}'))
            fp32=model(image).float().cpu().numpy()
            with torch.autocast('cuda',dtype=torch.float16):amp=model(image).float().cpu().numpy()
            e32=float(np.max(np.abs(fp32-fixture['features'])))
            e16=float(np.max(np.abs(amp-fp32)))
            if not np.isfinite(fp32).all() or not np.isfinite(amp).all() or e32>.003 or e16>.05:
                raise RuntimeError(f'fixed encoder fixture failed: fp32={e32}, fp16={e16}')
            parity.append({'device':device,'cpu_gpu_fp32_max_abs_error':e32,'amp_fp32_max_abs_error':e16})
            encoders.append(model)
    del model
    (OUT/'encoder_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    def extract(indices,device):
        with torch.cuda.device(device),torch.inference_mode():
            for start in range(0,len(indices),4):
                guard();batch=indices[start:start+4]
                pixels=np.zeros((len(batch),20,336,336),np.uint8);valid=np.zeros((len(batch),20),bool)
                for bi,row in enumerate(batch):
                    centers,_,_=select_slot_windows(masks[row].tolist(),4,False)
                    volume=arrays[0][row] if row<3200 else arrays[1][row-3200]
                    for slot,picks in enumerate(centers):
                        for k,c in enumerate(picks):
                            if c is not None:pixels[bi,slot*4+k]=volume[c];valid[bi,slot*4+k]=True
                keep=valid.ravel()
                image=torch.from_numpy(pixels.reshape(-1,336,336)[keep]).to(f'cuda:{device}').float()[:,None]
                image=normalize_pixels(F.interpolate(image,size=(224,224),mode='bilinear',align_corners=False))
                chunks=[]
                for offset in range(0,len(image),16):
                    guard()
                    with torch.autocast('cuda',dtype=torch.float16):
                        values=encoders[device](image[offset:offset+16])
                    if not torch.isfinite(values).all() or values.abs().max().float()>65000:
                        raise RuntimeError('feature overflow')
                    chunks.append(values.half().cpu().numpy())
                values=np.concatenate(chunks)
                block=np.zeros((len(batch)*20,2048),np.float16);block[keep]=values
                if not np.array_equal(valid,old_mask[batch]):raise RuntimeError('slice mask differs from control')
                features['candidate'][batch]=block.reshape(len(batch),20,2048)
                token_mask[batch]=valid
                if start%64==0:
                    for arr in arrays+[old_mean]+list(features.values()):evict(arr)
                    print(json.dumps({'stage':'features','device':device,'completed':start+len(batch),
                                      'partition_total':len(indices),'seconds':time.monotonic()-START}),flush=True)
    def extract_pair(partitions):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(extract,part,device) for device,part in enumerate(partitions)]
            for future in futures:future.result()
    all_parts=[np.arange(d,4407,2) for d in range(2)]
    before=time.monotonic();extract_pair([part[:128] for part in all_parts]);early_seconds=time.monotonic()-before
    pilot_ids=np.intersect1d(train,np.arange(256))
    if len(pilot_ids)<64:raise RuntimeError('insufficient weak-only pilot rows')
    for array in features.values():array.flush();evict(array)
    for name,value in dict(token_mask=token_mask,labels=labels,weights=weights,train=train,expert=expert,pilot_ids=pilot_ids).items():
        np.save(scratch/(name+'.npy'),value)
    (scratch/'fit_config.json').write_text(json.dumps({'seeds':CONTRACT['seeds']}))
    pilot_wall=fit_processes(scratch,pilot=True)
    pilot={str(d):json.loads((OUT/f'pilot_device_{d}.json').read_text()) for d in range(2)}
    steps_per_fit=math.ceil(4345/64)*30
    # Two separate processes: device0 runs seeds0/2, device1 runs seed1.
    projected_devices={str(d):sum(pilot[str(d)][arm]['p95_complete_step_seconds']*steps_per_fit
                        for arm in ['control','candidate'])*len(CONTRACT['seeds'][d::2])*1.2 for d in range(2)}
    projected_heads=max(projected_devices.values())
    process_margin=max(60.,pilot_wall)
    projected_remaining_extraction=early_seconds*(4407-256)/256*1.2
    projection=time.monotonic()+projected_heads+projected_remaining_extraction+240+process_margin
    resource={'status':'PASS' if projection<DEADLINE else 'REJECT_COMPLETE_RECIPE_DOES_NOT_FIT',
              'pilot':pilot,'projected_device_head_seconds':projected_devices,'pilot_wall_seconds':pilot_wall,
              'process_startup_margin_seconds':process_margin,'early_feature_studies':256,'early_feature_seconds':early_seconds,
              'projected_remaining_extraction_seconds':projected_remaining_extraction,
              'projected_all_head_seconds':projected_heads,'finalization_margin_seconds':240,
              'projected_worker_elapsed_seconds':projection-START,'worker_deadline_seconds':3300,
              'pilot_expert_rows':0,'host_memory':host_memory()}
    (OUT/'resource_pilot.json').write_text(json.dumps(resource,indent=2)+'\n')
    print(json.dumps({'stage':'resource_gate',**resource}),flush=True)
    if resource['status']!='PASS':raise RuntimeError('resource projection rejects full fixed schedule')
    extract_pair([part[128:] for part in all_parts])
    del encoders,arrays,old_mean
    gc.collect();torch.cuda.empty_cache()
    feature_receipts={}
    for arm,array in features.items():
        array.flush();evict(array);path=scratch/(arm+'.npy')
        feature_receipts[arm]={'shape':list(array.shape),'bytes':path.stat().st_size,'sha256':sha(path),'persisted':arm=='candidate'}
    np.save(OUT/'expert_ids.npy',ids[expert])
    np.save(OUT/'study_ids.npy',ids)
    np.save(OUT/'token_masks.npy',token_mask)
    np.save(scratch/'token_mask.npy',token_mask)
    fit_processes(scratch)
    outputs=[json.loads((OUT/f'{arm}_{seed}_fit.json').read_text())
             for seed in CONTRACT['seeds'] for arm in ['control','candidate']]
    # Score only after all six fits. No epoch/seed/target weight search.
    truth=np.asarray([[float(official[ids[i]][target]) for target in CONTRACT['targets']] for i in expert])
    for result in outputs:
        prediction=np.load(OUT/f"{result['arm']}_{result['seed']}_expert.npy")
        aucs={t:float(roc_auc_score(truth[:,j],prediction[:,j])) for j,t in enumerate(CONTRACT['targets'])}
        result.update(expert_per_target_auc=aucs,expert_macro_auc=float(np.mean(list(aucs.values()))))
    for array in features.values():evict(array)
    del features
    shutil.move(scratch/'candidate.npy',OUT/'orthofoundation_features.npy')
    shutil.rmtree(scratch)
    manifest={'schema':'orthofoundation_paired_probe_v1','status':'COMPLETE','contract':CONTRACT,'results':outputs,
              'resource_pilot':resource,'feature_receipts':feature_receipts,'global_feature_reference_exact':True,'encoder_parity':parity,
              'elapsed_seconds':time.monotonic()-START,'public_score':None,'competition_submission_enabled':False,
              'all_expert_cases_excluded_from_training':True,
              'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file()}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'stage':'complete','scores':[{k:r[k] for k in ['seed','arm','expert_macro_auc']} for r in outputs]}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        (OUT/'failure.json').write_text(json.dumps({'status':'INCOMPLETE_INELIGIBLE','error_type':type(error).__name__,
            'message':str(error),'elapsed_seconds':time.monotonic()-START,'host_memory':host_memory()})+'\n')
        raise
