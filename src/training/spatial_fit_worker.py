"""Independent single-GPU process for the unchanged fixed spatial head recipe."""
import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import spatial_probe_support as support
from spatial_pooling_head import SpatialHead
from spatial_pooling_probe_v2 import evict, head_step


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    support.DEADLINE=args.deadline
    torch.set_num_threads(2)
    if torch.cuda.device_count()!=1:raise RuntimeError('worker requires exactly one isolated GPU')
    scratch=args.scratch
    features={arm:np.load(scratch/(arm+'.npy'),mmap_mode='r') for arm in ['control','candidate']}
    data={name:np.load(scratch/(name+'.npy'),allow_pickle=False)
          for name in ['token_mask','labels','weights','train','expert','pilot_ids']}
    config=json.loads((scratch/'fit_config.json').read_text())
    out=support.OUT
    if args.pilot:
        # Both processes begin together so measured rates include shared CPU/I/O contention.
        (scratch/f'ready_{args.device}').touch()
        while not (scratch/'pilot_go').exists():support.guard();time.sleep(.1)
        result={}
        for arm in ['control','candidate']:
            torch.manual_seed(20261011);torch.cuda.manual_seed_all(20261011)
            model=SpatialHead().cuda().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            measured=[]
            for step in range(50):
                support.guard()
                indices=np.random.default_rng(20261011+step).choice(data['pilot_ids'],64,replace=False)
                torch.cuda.synchronize();before=time.monotonic()
                head_step(model,optimizer,features[arm],data['token_mask'],data['labels'],data['weights'],indices)
                torch.cuda.synchronize()
                if step>=10:measured.append(time.monotonic()-before)
                if step%10==0:evict(features[arm])
            result[arm]={'measured_steps':40,'mean_complete_step_seconds':float(np.mean(measured)),
                         'p95_complete_step_seconds':float(np.quantile(measured,.95))}
            del model,optimizer
            evict(features[arm]);gc.collect();torch.cuda.empty_cache()
        (out/f'pilot_device_{args.device}.json').write_text(json.dumps(result,indent=2)+'\n')
        return
    for seed in config['seeds'][args.device::2]:
        for arm in ['control','candidate']:
            support.guard();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=SpatialHead().cuda()
            initial=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            history=[]
            for epoch in range(30):
                torch.manual_seed(seed+epoch*100);torch.cuda.manual_seed_all(seed+epoch*100)
                order=np.random.default_rng(seed+epoch).permutation(data['train']);model.train();losses=[]
                for start in range(0,len(order),64):
                    support.guard()
                    losses.append(head_step(model,optimizer,features[arm],data['token_mask'],data['labels'],data['weights'],order[start:start+64]))
                    if start%(64*16)==0:evict(features[arm])
                history.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
                print(json.dumps({'stage':'head','device':args.device,'arm':arm,'seed':seed,**history[-1]}),flush=True)
            model.eval()
            def predict(head):
                blocks=[]
                with torch.inference_mode():
                    for start in range(0,58,16):
                        batch=data['expert'][start:start+16]
                        x=torch.from_numpy(np.asarray(features[arm][batch])).cuda()
                        mask=torch.from_numpy(data['token_mask'][batch]).cuda()
                        blocks.append(head(x,mask).sigmoid().cpu().numpy())
                return np.concatenate(blocks)
            prediction=predict(model)
            if not np.isfinite(prediction).all():raise RuntimeError('invalid expert prediction')
            np.save(out/f'{arm}_{seed}_expert.npy',prediction)
            torch.save(model.state_dict(),out/f'{arm}_{seed}_head.pt')
            replay=SpatialHead().cuda();replay.load_state_dict(torch.load(out/f'{arm}_{seed}_head.pt',weights_only=True));replay.eval()
            error=float(np.max(np.abs(predict(replay)-prediction)))
            if error>1e-6:raise RuntimeError('checkpoint replay failed')
            result={'seed':seed,'arm':arm,'epochs':30,'initial_state_sha256':initial,'history':history,
                    'checkpoint_replay_max_abs_error':error,'physical_device':args.device}
            (out/f'{arm}_{seed}_fit.json').write_text(json.dumps(result,indent=2)+'\n')
            del replay,model,optimizer
            evict(features[arm]);gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
