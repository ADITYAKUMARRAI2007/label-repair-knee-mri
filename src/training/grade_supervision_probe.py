"""One bounded paired frozen-feature supervision probe; no submission output."""
import csv
import hashlib
import json
import mmap
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet50

from grade_aware_supervision import adjusted_probability
from e03_split_contract import freeze_split
from e03_window_contract import select_slot_windows

# __PROBE_CONTRACT__

START=time.monotonic()
DEADLINE=START+3300
OUT=Path('/kaggle/working/grade_probe');OUT.mkdir(exist_ok=True)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def guard():
    if time.monotonic()>DEADLINE:raise TimeoutError('bounded probe deadline; incomplete results are ineligible')
    if Path('/sys/fs/cgroup/memory.current').exists():
        if int(Path('/sys/fs/cgroup/memory.current').read_text())>18*(1<<30):
            raise RuntimeError('host memory guard')
    if any(torch.cuda.max_memory_allocated(i)>12*(1<<30) for i in range(torch.cuda.device_count())):
        raise RuntimeError('GPU memory guard')


def find(name,expected=None):
    matches=[]
    for root,dirs,files in os.walk('/kaggle/input'):
        dirs[:]=[d for d in dirs if d not in {'train_series','test_series'}]
        if name in files:
            p=Path(root)/name
            if expected is None or sha(p)==expected:matches.append(p)
    if len(matches)!=1:raise RuntimeError(f'expected one pinned asset: {name}; got {len(matches)}')
    return matches[0]


def table(path):
    with path.open(newline='') as handle:records=list(csv.DictReader(handle))
    d={r['StudyInstanceUID']:r for r in records}
    if len(d)!=len(records):raise RuntimeError('duplicate source identity')
    return records,d


class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(self,x):return self.backbone(x).mean((2,3))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256),nn.GELU())
        self.slot=nn.Parameter(torch.randn(5,256)*.01)
        self.position=nn.Parameter(torch.randn(4,256)*.01)
        self.query=nn.Parameter(torch.randn(12,256)*.01)
        self.weight=nn.Parameter(torch.randn(12,256)*.01)
        self.bias=nn.Parameter(torch.zeros(12))
        self.dropout=nn.Dropout(.1)
    def forward(self,x,mask):
        if not mask.any(1).all():raise RuntimeError('all-empty study')
        token=self.project(x.float()).view(-1,5,4,256)
        token=(token+self.slot[None,:,None]+self.position[None,None]).flatten(1,2)
        score=torch.einsum('bkd,td->btk',token,self.query)/16
        attention=score.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
        pooled=torch.einsum('btk,bkd->btd',attention,token)
        return (self.dropout(pooled)*self.weight[None]).sum(-1)+self.bias


def main():
    torch.set_num_threads(2)
    if torch.cuda.device_count()!=2:raise RuntimeError('paired T4 requirement')
    train_path=find('train.csv',CONTRACT['train_sha256']); records,official=table(train_path)
    split=freeze_split(records,CONTRACT['targets'])
    excluded=set(split['gold'])|set(split['excluded_gold_group_weak'])
    train_ids=sorted(set(official)-excluded)
    if hashlib.sha256(json.dumps(train_ids).encode()).hexdigest()!=CONTRACT['train_ids_sha256']:
        raise RuntimeError('expert/report-group training exclusion drift')
    source=[]
    for name,expected in CONTRACT['label_sources'].items():source.append(table(find(name,expected))[1])
    arrays=[];masks=[];ids=[]
    for part,expected in CONTRACT['cache'].items():
        manifest_path=find(f'e04_wide_{part}_manifest.json',expected)
        manifest=json.loads(manifest_path.read_text())
        if manifest['status']!='COMPLETE' or manifest['span']!=[.06,.94]:raise RuntimeError('cache contract drift')
        mounted={}
        for name,receipt in manifest['artifacts'].items():
            p=find(name)
            if p.stat().st_size!=receipt['bytes'] or sha(p)!=receipt['sha256']:
                raise RuntimeError('mounted cache bytes failed')
            mounted[name]=p
        arrays.append(np.load(mounted[f'e04_wide_{part}_vols.npy'],mmap_mode='r'))
        masks.append(np.load(mounted[f'e04_wide_{part}_masks.npy'],allow_pickle=False))
        ids.extend(np.load(mounted[f'e04_wide_{part}_ids.npy'],allow_pickle=False).astype(str).tolist())
    masks=np.concatenate(masks); ids=np.asarray(ids)
    if len(set(ids))!=4407 or set(ids)!=set(official):raise RuntimeError('cache identity drift')
    y=np.zeros((len(ids),12),np.float32);candidate=y.copy();weight=y.copy()
    changed={t:0 for t in CONTRACT['targets']}
    for row,uid in enumerate(ids):
        if uid in excluded:continue
        for j,t in enumerate(CONTRACT['targets']):
            values=np.asarray([float(s[uid][t]) for s in source],np.float64)
            if not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise RuntimeError('invalid labels')
            y[row,j]=np.clip(np.median(values),.01,.99)
            # Use float64 base before casting: exactly the frozen local recipe.
            value,_=adjusted_probability(official[uid]['Report'],t,float(np.clip(np.median(values),.01,.99)))
            candidate[row,j]=value
            weight[row,j]=np.clip(1-np.ptp(values),.25,1)
            changed[t]+=int(candidate[row,j]!=y[row,j])
    if {t:n for t,n in changed.items() if n}!=CONTRACT['changed_cells']:raise RuntimeError('candidate policy count drift')
    encoder_path=find('ResNet50.pt',CONTRACT['encoder_sha256'])
    state=torch.load(encoder_path,map_location='cpu',weights_only=True)
    encoders=[]
    for device in range(2):
        encoder=Encoder();encoder.load_state_dict(state,strict=True)
        if sum(p.numel() for p in encoder.parameters())!=23508032:raise RuntimeError('encoder identity')
        if not all(torch.isfinite(p).all() for p in encoder.parameters()):raise RuntimeError('nonfinite encoder')
        encoders.append(encoder.eval().requires_grad_(False).to(f'cuda:{device}'))
    features=np.lib.format.open_memmap(OUT/'features.npy',mode='w+',dtype=np.float16,shape=(4407,20,2048))
    token_masks=np.zeros((4407,20),bool)

    def extract(indices,device):
        with torch.cuda.device(device),torch.inference_mode():
            for start in range(0,len(indices),4):
                guard(); batch=indices[start:start+4];pixels=np.zeros((len(batch),20,336,336),np.uint8)
                valid=np.zeros((len(batch),20),bool)
                for bi,row in enumerate(batch):
                    centers,present,_=select_slot_windows(masks[row].tolist(),4,False)
                    volume=arrays[0][row] if row<3200 else arrays[1][row-3200]
                    for slot,picks in enumerate(centers):
                        for k,c in enumerate(picks):
                            if c is not None:pixels[bi,slot*4+k]=volume[c];valid[bi,slot*4+k]=True
                flat=valid.ravel();images=torch.from_numpy(pixels.reshape(-1,336,336)[flat]).to(f'cuda:{device}').float()[:,None]
                images=F.interpolate(images,size=(224,224),mode='bilinear',align_corners=False).div_(127.5).sub_(1)
                with torch.autocast('cuda',dtype=torch.float16):values=encoders[device](images.expand(-1,3,-1,-1).contiguous())
                values=values.float().cpu().numpy()
                if not np.isfinite(values).all() or np.abs(values).max()>65000:raise RuntimeError('feature overflow')
                block=np.zeros((len(batch)*20,2048),np.float16);block[flat]=values
                features[batch]=block.reshape(len(batch),20,2048);token_masks[batch]=valid
                if start%64==0:
                    print(json.dumps({'stage':'features','device':device,'completed':start+len(batch),'total':len(indices),'seconds':time.monotonic()-START}),flush=True)
                    for arr in arrays:
                        if hasattr(arr._mmap,'madvise'):arr._mmap.madvise(mmap.MADV_DONTNEED)
                # Measured early projection leaves room for six fixed head fits.
                if start==124:
                    rate=(time.monotonic()-feature_start)/(start+len(batch))
                    if feature_start+rate*len(indices)+1200>DEADLINE:
                        raise TimeoutError('measured projection cannot fit bounded comparison')
    feature_start=time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(extract,np.arange(d,4407,2),d) for d in range(2)]
        for future in futures:future.result()
    feature_end=time.monotonic()
    features.flush();np.save(OUT/'token_masks.npy',token_masks);np.save(OUT/'study_ids.npy',ids)
    del encoders,state,encoder
    torch.cuda.empty_cache()
    train_id_set=set(train_ids);gold_id_set=set(split['gold'])
    train=np.asarray([i for i,u in enumerate(ids) if u in train_id_set],np.int64)
    expert=np.asarray([i for i,u in enumerate(ids) if u in gold_id_set],np.int64)
    if len(train)!=4345 or len(expert)!=58:raise RuntimeError('cohort counts drift')
    outputs=[]
    for seed in CONTRACT['seeds']:
        for arm,target_values in [('control',y),('candidate',candidate)]:
            guard();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=Head().cuda();initial=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            history=[]
            for epoch in range(30):
                guard();torch.manual_seed(seed+epoch*100);torch.cuda.manual_seed_all(seed+epoch*100)
                order=np.random.default_rng(seed+epoch).permutation(train);model.train();losses=[]
                for start in range(0,len(order),64):
                    guard();b=order[start:start+64]
                    x=torch.from_numpy(np.array(features[b],copy=True)).cuda();mask=torch.from_numpy(token_masks[b]).cuda()
                    labels=torch.from_numpy(target_values[b]).cuda();w=torch.from_numpy(weight[b]).cuda()
                    optimizer.zero_grad(set_to_none=True);logits=model(x,mask)
                    loss=(F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*w).mean()
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                    loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),3.)
                    if not torch.isfinite(norm):raise RuntimeError('nonfinite gradients')
                    optimizer.step();losses.append(float(loss.detach()))
                history.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
                print(json.dumps({'stage':'head','seed':seed,'arm':arm,**history[-1],'seconds':time.monotonic()-START}),flush=True)
            model.eval();pred=[]
            with torch.inference_mode():
                for start in range(0,len(expert),32):
                    b=expert[start:start+32]
                    pred.append(model(torch.from_numpy(np.array(features[b],copy=True)).cuda(),torch.from_numpy(token_masks[b]).cuda()).sigmoid().cpu().numpy())
            prediction=np.concatenate(pred)
            if not np.isfinite(prediction).all():raise RuntimeError('invalid expert predictions')
            np.save(OUT/f'{arm}_{seed}_expert.npy',prediction)
            torch.save(model.state_dict(),OUT/f'{arm}_{seed}_head.pt')
            replay=Head().cuda();replay.load_state_dict(torch.load(OUT/f'{arm}_{seed}_head.pt',weights_only=True));replay.eval()
            replay_predictions=[]
            with torch.inference_mode():
                for start in range(0,len(expert),32):
                    b=expert[start:start+32]
                    replay_predictions.append(replay(torch.from_numpy(np.array(features[b],copy=True)).cuda(),torch.from_numpy(token_masks[b]).cuda()).sigmoid().cpu().numpy())
            replay_error=float(np.max(np.abs(np.concatenate(replay_predictions)-prediction)))
            if replay_error>1e-6:raise RuntimeError('saved head replay failed')
            outputs.append({'seed':seed,'arm':arm,'initial_state_sha256':initial,'epochs':30,'history':history,'checkpoint_replay_max_abs_error':replay_error})
            del replay
            del model,optimizer;torch.cuda.empty_cache()
    # Expert finding values are accessed only after every fixed fit is complete.
    truth=np.asarray([[float(official[ids[i]][t]) for t in CONTRACT['targets']] for i in expert])
    for output in outputs:
        pred=np.load(OUT/f"{output['arm']}_{output['seed']}_expert.npy")
        aucs={t:float(roc_auc_score(truth[:,j],pred[:,j])) for j,t in enumerate(CONTRACT['targets'])}
        output.update(expert_per_target_auc=aucs,expert_macro_auc=float(np.mean(list(aucs.values()))))
    np.save(OUT/'expert_ids.npy',ids[expert])
    manifest={'schema':'grade_supervision_paired_probe_v1','status':'COMPLETE','contract':CONTRACT,
              'results':outputs,'changed_cells':changed,'all_expert_cases_excluded_from_training':True,
              'feature_seconds':feature_end-feature_start,
              'elapsed_seconds':time.monotonic()-START,'public_score':None,'competition_submission_enabled':False,
              'artifacts':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file()}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'stage':'complete','scores':[{k:r[k] for k in ['seed','arm','expert_macro_auc']} for r in outputs]}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        (OUT/'failure.json').write_text(json.dumps({'status':'INCOMPLETE_INELIGIBLE','error_type':type(error).__name__,'message':str(error),'elapsed_seconds':time.monotonic()-START})+'\n')
        raise
