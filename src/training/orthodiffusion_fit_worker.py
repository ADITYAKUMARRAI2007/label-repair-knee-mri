"""Fixed independent heads; no label-based epoch selection or scoring here."""
import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from spatial_pooling_head import SpatialHead
from orthodiffusion_head import DiffusionHead
import diffusion_probe_support as support

def evict(array):
    import mmap
    if hasattr(array,'_mmap') and hasattr(array._mmap,'madvise'):
        array._mmap.madvise(mmap.MADV_DONTNEED)

def head_for(arm):return SpatialHead() if arm=='control' else DiffusionHead()

def step(model,optimizer,features,mask,y,w,rows):
    x=torch.from_numpy(np.asarray(features[rows])).cuda()
    valid=torch.from_numpy(mask[rows]).cuda()
    target=torch.from_numpy(y[rows]).cuda();weight=torch.from_numpy(w[rows]).cuda()
    optimizer.zero_grad(set_to_none=True)
    loss=(F.binary_cross_entropy_with_logits(model(x,valid),target,reduction='none')*weight).mean()
    if not torch.isfinite(loss):raise RuntimeError('nonfinite head loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
    if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
    optimizer.step()
    return float(loss.detach())

def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--device',type=int,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--pilot',action='store_true');args=p.parse_args()
    support.DEADLINE=args.deadline;torch.set_num_threads(2)
    assert torch.cuda.device_count()==1
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    scratch=args.scratch
    features={arm:np.load(scratch/(arm+'.npy'),mmap_mode='r') for arm in ['control','candidate']}
    masks={arm:np.load(scratch/(arm+'_mask.npy')) for arm in features}
    data={name:np.load(scratch/(name+'.npy')) for name in ['labels','weights','train','expert','pilot_ids']}
    config=json.loads((scratch/'fit_config.json').read_text());out=support.OUT
    if args.pilot:
        (scratch/f'ready_{args.device}').touch()
        while not (scratch/'pilot_go').exists():support.guard();time.sleep(.1)
        result={}
        for arm in features:
            torch.manual_seed(20261011);torch.cuda.manual_seed_all(20261011)
            model=head_for(arm).cuda().train()
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            durations=[]
            for k in range(50):
                support.guard()
                rows=np.random.default_rng(20261011+k).choice(data['pilot_ids'],64,replace=False)
                torch.cuda.synchronize();begin=time.monotonic()
                step(model,opt,features[arm],masks[arm],data['labels'],data['weights'],rows)
                torch.cuda.synchronize()
                if k>=10:durations.append(time.monotonic()-begin)
                if k%10==0:evict(features[arm])
            result[arm]=dict(measured_steps=40,p95_complete_step_seconds=float(np.quantile(durations,.95)),
                             mean_complete_step_seconds=float(np.mean(durations)))
            del model,opt;evict(features[arm]);gc.collect();torch.cuda.empty_cache()
        (out/f'pilot_device_{args.device}.json').write_text(json.dumps(result,indent=2)+'\n')
        return
    for seed in config['seeds'][args.device::2]:
        for arm in features:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=head_for(arm).cuda()
            initial=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            history=[]
            for epoch in range(30):
                model.train();torch.manual_seed(seed+epoch*100);torch.cuda.manual_seed_all(seed+epoch*100)
                order=np.random.default_rng(seed+epoch).permutation(data['train']);losses=[]
                for start in range(0,len(order),64):
                    support.guard()
                    losses.append(step(model,opt,features[arm],masks[arm],data['labels'],data['weights'],order[start:start+64]))
                    if start%(64*16)==0:evict(features[arm])
                history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
                print(json.dumps(dict(stage='head',device=args.device,seed=seed,arm=arm,**history[-1])),flush=True)
            def predict(head):
                head.eval();blocks=[]
                with torch.inference_mode():
                    for start in range(0,58,16):
                        batch=data['expert'][start:start+16]
                        x=torch.from_numpy(np.asarray(features[arm][batch])).cuda()
                        mask=torch.from_numpy(masks[arm][batch]).cuda()
                        blocks.append(head(x,mask).sigmoid().cpu().numpy())
                return np.concatenate(blocks)
            pred=predict(model);assert np.isfinite(pred).all()
            np.save(out/f'{arm}_{seed}_expert.npy',pred)
            torch.save(model.state_dict(),out/f'{arm}_{seed}_head.pt')
            replay=head_for(arm).cuda()
            replay.load_state_dict(torch.load(out/f'{arm}_{seed}_head.pt',weights_only=True))
            error=float(np.max(np.abs(predict(replay)-pred)))
            assert error<=1e-6
            result=dict(seed=seed,arm=arm,epochs=30,history=history,initial_state_sha256=initial,
                        trainable_parameters=sum(p.numel() for p in model.parameters()),
                        physical_device=args.device,checkpoint_replay_max_abs_error=error)
            (out/f'{arm}_{seed}_fit.json').write_text(json.dumps(result,indent=2)+'\n')
            del model,replay,opt;evict(features[arm]);gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
