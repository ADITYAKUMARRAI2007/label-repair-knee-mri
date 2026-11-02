"""Check missing-plane invariance, finite updates and exact checkpoint replay."""
import hashlib
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from orthodiffusion_head import DiffusionHead

def main():
    torch.set_num_threads(2);torch.manual_seed(20261017)
    out=ROOT/'runs/orthodiffusion_trial_v1';out.mkdir(exist_ok=True)
    model=DiffusionHead().eval()
    x=torch.randn(2,3,256,256);mask=torch.tensor([[True,False,True],[False,True,False]])
    with torch.no_grad():
        original=model(x,mask)
        changed=x.clone();changed[~mask]=torch.randn_like(changed[~mask])*100
        mask_error=float((model(changed,mask)-original).abs().max())
        batch_error=float((model(x[:1],mask[:1])-original[:1]).abs().max())
    assert mask_error==0 and batch_error<1e-6
    try:model(x,torch.zeros_like(mask))
    except ValueError:pass
    else:raise AssertionError('empty mask accepted')
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    model.train();opt.zero_grad(set_to_none=True)
    loss=F.binary_cross_entropy_with_logits(model(x,mask),torch.rand(2,12))
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
    assert torch.isfinite(norm)
    opt.step();model.eval()
    torch.save(model.state_dict(),out/'synthetic_head.pt')
    replay=DiffusionHead();replay.load_state_dict(torch.load(out/'synthetic_head.pt',weights_only=True));replay.eval()
    with torch.no_grad():error=float((replay(x,mask)-model(x,mask)).abs().max())
    assert error==0
    result=dict(status='PASS_HEAD_MASK_UPDATE_REPLAY',masked_plane_max_error=mask_error,
                batch_invariance_max_error=batch_error,checkpoint_replay_max_error=error,
                trainable_parameters=sum(p.numel() for p in model.parameters()),
                source_sha256=hashlib.sha256((ROOT/'training/orthodiffusion_head.py').read_bytes()).hexdigest(),
                clinical_data_used=False)
    (out/'head_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
