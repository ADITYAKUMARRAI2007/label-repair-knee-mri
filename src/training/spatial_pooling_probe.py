"""Bounded full-grid paired probe; no training schedule or score rescue."""
import gc
import hashlib
import json
import math
import mmap
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from spatial_pooling_head import SpatialHead
from spatial_probe_support import Encoder, find, table, sha, guard, host_memory, OUT, START, DEADLINE
from e03_split_contract import freeze_split
from e03_window_contract import select_slot_windows

# __PROBE_CONTRACT__


def evict(array):
    if hasattr(array,'_mmap') and hasattr(array._mmap,'madvise'):
        array._mmap.madvise(mmap.MADV_DONTNEED)


def head_step(model,optimizer,features,mask,labels,weights,indices):
    x=torch.from_numpy(np.array(features[indices],copy=True)).cuda()
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


def resource_pilot(features,mask,labels,weights,indices):
    results={}
    for arm in ['control','candidate']:
        torch.manual_seed(20261011);torch.cuda.manual_seed_all(20261011)
        model=SpatialHead().cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        measured=[]
        for step in range(50):
            guard()
            batch=np.random.default_rng(20261011+step).choice(indices,64,replace=False)
            torch.cuda.synchronize();before=time.monotonic()
            head_step(model,optimizer,features[arm],mask,labels,weights,batch)
            torch.cuda.synchronize()
            if step>=10:measured.append(time.monotonic()-before)
            if step%10==0:evict(features[arm])
        results[arm]={'measured_steps':len(measured),'mean_complete_step_seconds':float(np.mean(measured)),
                      'p95_complete_step_seconds':float(np.quantile(measured,.95))}
        del model,optimizer
        evict(features[arm]);gc.collect();torch.cuda.empty_cache()
    return results


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
    scratch=Path('/tmp/rsna_spatial_probe_features');scratch.mkdir(exist_ok=False)
    if shutil.disk_usage(scratch).free<22*(1<<30):raise RuntimeError('insufficient feature scratch space')
    features={'control':np.lib.format.open_memmap(scratch/'control.npy',mode='w+',dtype=np.float16,shape=(4407,20,2048)),
              'candidate':np.lib.format.open_memmap(scratch/'candidate.npy',mode='w+',dtype=np.float16,shape=(4407,20,49,2048))}
    token_mask=np.zeros((4407,20),bool)
    state=torch.load(find('ResNet50.pt',CONTRACT['encoder_sha256']),map_location='cpu',weights_only=True)
    encoders=[]
    for device in range(2):
        model=Encoder();model.load_state_dict(state,strict=True)
        if sum(p.numel() for p in model.parameters())!=23508032:raise RuntimeError('encoder identity differs')
        encoders.append(model.eval().requires_grad_(False).to(f'cuda:{device}'))
    del model,state
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
                image=F.interpolate(image,size=(224,224),mode='bilinear',align_corners=False).div_(127.5).sub_(1)
                with torch.autocast('cuda',dtype=torch.float16):
                    spatial=encoders[device](image.expand(-1,3,-1,-1).contiguous())
                    mean=spatial.mean((2,3))
                if spatial.shape[1:]!=(2048,7,7):raise RuntimeError('encoder spatial shape differs')
                mean=mean.float().cpu().numpy()
                spatial=spatial.flatten(2).transpose(1,2).contiguous().float().cpu().numpy()
                if not np.isfinite(spatial).all() or np.abs(spatial).max()>65000:raise RuntimeError('feature overflow')
                global_block=np.zeros((len(batch)*20,2048),np.float16);global_block[keep]=mean
                global_block=global_block.reshape(len(batch),20,2048)
                if not np.array_equal(global_block,old_mean[batch]) or not np.array_equal(valid,old_mask[batch]):
                    raise RuntimeError('global feature/mask parity with prior encoder failed')
                grid=np.zeros((len(batch)*20,49,2048),np.float16);grid[keep]=spatial
                features['control'][batch]=global_block
                features['candidate'][batch]=grid.reshape(len(batch),20,49,2048)
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
    pilot=resource_pilot(features,token_mask,labels,weights,pilot_ids)
    per_arm_steps=math.ceil(4345/64)*30*3
    projected_heads=sum(pilot[arm]['p95_complete_step_seconds']*per_arm_steps for arm in pilot)*1.2
    projected_remaining_extraction=early_seconds*(4407-256)/256*1.2
    projection=time.monotonic()+projected_heads+projected_remaining_extraction+240
    resource={'status':'PASS' if projection<DEADLINE else 'REJECT_COMPLETE_RECIPE_DOES_NOT_FIT',
              'pilot':pilot,'early_feature_studies':256,'early_feature_seconds':early_seconds,
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
        feature_receipts[arm]={'shape':list(array.shape),'bytes':path.stat().st_size,'sha256':sha(path),'persisted':False}
    np.save(OUT/'expert_ids.npy',ids[expert])
    outputs=[]
    for seed in CONTRACT['seeds']:
        for arm in ['control','candidate']:
            guard();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=SpatialHead().cuda()
            initial=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            history=[]
            for epoch in range(30):
                torch.manual_seed(seed+epoch*100);torch.cuda.manual_seed_all(seed+epoch*100)
                order=np.random.default_rng(seed+epoch).permutation(train);model.train();losses=[]
                for start in range(0,len(order),64):
                    guard();losses.append(head_step(model,optimizer,features[arm],token_mask,labels,weights,order[start:start+64]))
                    if start%(64*16)==0:evict(features[arm])
                history.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
                print(json.dumps({'stage':'head','arm':arm,'seed':seed,**history[-1],'seconds':time.monotonic()-START}),flush=True)
            model.eval()
            def predict(head):
                blocks=[]
                with torch.inference_mode():
                    for start in range(0,58,16):
                        batch=expert[start:start+16]
                        x=torch.from_numpy(np.array(features[arm][batch],copy=True)).cuda()
                        mask=torch.from_numpy(token_mask[batch]).cuda()
                        blocks.append(head(x,mask).sigmoid().cpu().numpy())
                return np.concatenate(blocks)
            prediction=predict(model)
            if not np.isfinite(prediction).all():raise RuntimeError('invalid expert prediction')
            np.save(OUT/f'{arm}_{seed}_expert.npy',prediction)
            torch.save(model.state_dict(),OUT/f'{arm}_{seed}_head.pt')
            replay=SpatialHead().cuda();replay.load_state_dict(torch.load(OUT/f'{arm}_{seed}_head.pt',weights_only=True));replay.eval()
            error=float(np.max(np.abs(predict(replay)-prediction)))
            if error>1e-6:raise RuntimeError('checkpoint replay failed')
            outputs.append({'seed':seed,'arm':arm,'epochs':30,'initial_state_sha256':initial,'history':history,'checkpoint_replay_max_abs_error':error})
            del replay,model,optimizer
            evict(features[arm]);gc.collect();torch.cuda.empty_cache()
    # Score only after all six fits. No epoch/seed/target weight search.
    truth=np.asarray([[float(official[ids[i]][target]) for target in CONTRACT['targets']] for i in expert])
    for result in outputs:
        prediction=np.load(OUT/f"{result['arm']}_{result['seed']}_expert.npy")
        aucs={t:float(roc_auc_score(truth[:,j],prediction[:,j])) for j,t in enumerate(CONTRACT['targets'])}
        result.update(expert_per_target_auc=aucs,expert_macro_auc=float(np.mean(list(aucs.values()))))
    for array in features.values():evict(array)
    del features
    shutil.rmtree(scratch)
    manifest={'schema':'spatial_pooling_paired_probe_v1','status':'COMPLETE','contract':CONTRACT,'results':outputs,
              'resource_pilot':resource,'feature_receipts':feature_receipts,'global_feature_reference_exact':True,
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
