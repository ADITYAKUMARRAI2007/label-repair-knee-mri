"""Audit six complete fits, training exclusions, saved-head replay and paired AUC."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'runs/orthodiffusion_trial_v1'
sys.path.insert(0,str(ROOT/'training'))
from spatial_pooling_head import SpatialHead
from orthodiffusion_head import DiffusionHead
from e03_split_contract import freeze_split

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def table(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    out=args.directory;manifest=json.loads((out/'manifest.json').read_text())
    build=json.loads((LOCAL/'build.json').read_text());contract=build['contract']
    assert manifest['status']=='COMPLETE' and manifest['contract']==contract
    assert not manifest['competition_submission_enabled'] and manifest['public_score'] is None
    for name,expected in {**build['sources'],**build['gates']}.items():assert sha(ROOT/name)==expected,name
    for name,receipt in manifest['artifacts'].items():
        assert Path(name).name==name
        if name=='diffusion_features.npy':continue  # Full training cache remains on Kaggle.
        assert (out/name).stat().st_size==receipt['bytes'] and sha(out/name)==receipt['sha256'],name
    pre=manifest['real_volume_preflight'];resource=manifest['resource_pilot']
    assert pre['status']=='PASS' and not pre['failures'] and pre['successful_studies']==128
    assert pre['pilot_expert_rows']==0 and pre['three_study_feature_replay_max_error']==0
    assert resource['status']=='PASS' and resource['pilot_expert_rows']==0
    projections={str(d):sum(resource['pilot'][str(d)][a]['p95_complete_step_seconds']*68*30
                   for a in ['control','candidate'])*len(contract['seeds'][d::2])*1.2 for d in range(2)}
    for d,value in projections.items():assert abs(value-resource['projected_device_head_seconds'][d])<1e-8
    assert abs(resource['projected_remaining_extraction_seconds']-pre['decode_extract_seconds']*(4407-128)/128*1.2)<1e-8
    projected=resource['elapsed_at_projection_seconds']+max(projections.values())+resource['projected_remaining_extraction_seconds']+resource['process_startup_margin_seconds']+240
    assert abs(projected-resource['projected_worker_elapsed_seconds'])<.05 and projected<6000
    assert manifest['elapsed_seconds']<6000
    assert sha(ROOT/'data/train.csv')==contract['train_sha256']
    official=table(ROOT/'data/train.csv');split=freeze_split(official,contract['targets'])
    by_id={r['StudyInstanceUID']:r for r in official}
    ids=np.load(out/'study_ids.npy').astype(str);expert_ids=np.load(out/'expert_ids.npy').astype(str)
    train=np.load(out/'train_indices.npy');expert=np.load(out/'expert_indices.npy');pilot=np.load(out/'pilot_indices.npy')
    excluded=set(split['gold'])|set(split['excluded_gold_group_weak'])
    assert len(ids)==len(set(ids))==4407 and set(ids)==set(by_id)
    assert np.array_equal(expert_ids,ids[expert]) and set(expert_ids)==set(split['gold'])
    assert np.array_equal(train,np.array([i for i,u in enumerate(ids) if u not in excluded]))
    assert len(train)==4345 and len(expert)==58
    assert hashlib.sha256(json.dumps(sorted(ids[train].tolist())).encode()).hexdigest()==contract['train_ids_sha256']
    wanted=np.array(sorted(train,key=lambda i:hashlib.sha256(f'20261017:{ids[i]}'.encode()).digest())[:128])
    assert np.array_equal(pilot,wanted) and not set(pilot)&set(expert)
    sources=[]
    for name,expected in contract['label_sources'].items():
        matches=[p for p in (ROOT/'labels').rglob(name) if sha(p)==expected];assert matches,name
        sources.append({r['StudyInstanceUID']:r for r in table(matches[0])})
    y=np.zeros((4407,12),np.float32);w=y.copy()
    for i in train:
        for j,t in enumerate(contract['targets']):
            v=np.array([float(s[ids[i]][t]) for s in sources],np.float64)
            y[i,j]=np.clip(np.median(v),.01,.99);w[i,j]=np.clip(1-np.ptp(v),.25,1)
    assert np.array_equal(y,np.load(out/'labels.npy')) and np.array_equal(w,np.load(out/'weights.npy'))
    reference=ROOT/'runs/grade_supervision_probe_outputs_v2/grade_probe'
    for name,receipt in contract['previous_features'].items():assert sha(reference/name)==receipt['sha256']
    assert np.array_equal(ids,np.load(reference/'study_ids.npy'))
    control_mask=np.load(out/'control_mask.npy');candidate_mask=np.load(out/'candidate_mask.npy')
    assert np.array_equal(control_mask,np.load(reference/'token_masks.npy'))
    assert candidate_mask.shape==(4407,3) and candidate_mask.any(1).all()
    receipts=[json.loads(line) for line in (out/'volume_receipts.jsonl').read_text().splitlines()]
    assert len(receipts)==4407 and len({r['row'] for r in receipts})==4407
    for r in receipts:
        row=r['row'];assert r['StudyInstanceUID']==ids[row] and not r['prediction_fallback_used']
        present=[v['state']=='decoded' for v in r['planes']]
        assert np.array_equal(present,candidate_mask[row])
        for v in r['planes']:
            if v['state']=='decoded':
                assert v['decoded_slices']==16 and v['decode_failures']==v['geometry_fallbacks']==0
                assert v['central_stop']-v['central_start']==16 and not v['depth_resampling'] and not v['spatial_crop']
    arrays={'control':np.load(reference/'features.npy',mmap_mode='r')[expert],
            'candidate':np.load(out/'expert_features.npy')}
    masks={'control':control_mask[expert],'candidate':candidate_mask[expert]}
    assert arrays['candidate'].shape==(58,3,256,256) and arrays['candidate'].dtype==np.float16
    assert np.isfinite(arrays['candidate']).all() and not arrays['candidate'][~masks['candidate']].any()
    truth=np.array([[float(by_id[u][t]) for t in contract['targets']] for u in expert_ids])
    results={(r['seed'],r['arm']):r for r in manifest['results']};assert len(results)==len(manifest['results'])==6
    torch.set_num_threads(2);predictions={};scores={a:[] for a in arrays};replay_errors={}
    for seed in contract['seeds']:
        for arm in arrays:
            r=results[seed,arm]
            assert r['epochs']==30 and [h['epoch'] for h in r['history']]==list(range(1,31))
            assert np.isfinite([h['loss'] for h in r['history']]).all()
            assert r['physical_device']==contract['seeds'].index(seed)%2 and r['checkpoint_replay_max_abs_error']<=1e-6
            pred=np.load(out/f'{arm}_{seed}_expert.npy');assert pred.shape==(58,12) and np.isfinite(pred).all()
            auc=np.array([roc_auc_score(truth[:,j],pred[:,j]) for j in range(12)])
            assert np.allclose(auc,[r['expert_per_target_auc'][t] for t in contract['targets']],atol=1e-12,rtol=0)
            assert abs(auc.mean()-r['expert_macro_auc'])<1e-12
            predictions[seed,arm]=pred;scores[arm].append(auc)
            model=(SpatialHead() if arm=='control' else DiffusionHead()).eval()
            model.load_state_dict(torch.load(out/f'{arm}_{seed}_head.pt',weights_only=True,map_location='cpu'),strict=True)
            assert sum(p.numel() for p in model.parameters())==r['trainable_parameters']
            replay=[]
            with torch.inference_mode():
                for start in range(0,58,16):
                    replay.append(model(torch.from_numpy(arrays[arm][start:start+16]),
                                        torch.from_numpy(masks[arm][start:start+16])).sigmoid().numpy())
            error=float(np.abs(np.concatenate(replay)-pred).max());assert error<=1e-4
            replay_errors[f'{arm}_{seed}']=error
    control,candidate=np.asarray(scores['control']),np.asarray(scores['candidate']);delta=candidate-control
    counts=np.random.default_rng(20261010).multinomial(58,np.full(58,1/58),size=2000)
    boot=np.zeros(2000);valid=np.ones(2000,bool)
    for j in range(12):
        pos,neg=truth[:,j]==1,truth[:,j]==0
        positive,negative=counts[:,pos],counts[:,neg];denom=positive.sum(1)*negative.sum(1)
        valid &= denom>0
        for seed in contract['seeds']:
            difference=np.zeros((pos.sum(),neg.sum()))
            for arm,sign in [('candidate',1),('control',-1)]:
                p=predictions[seed,arm][:,j];d=p[pos,None]-p[None,neg]
                difference+=sign*((d>0).astype(float)+.5*(d==0))
            numerator=np.einsum('bi,ij,bj->b',positive,difference,negative,optimize=True)
            boot+=np.divide(numerator,denom,out=np.zeros(2000),where=denom>0)/36
    assert valid.sum()>=1900
    lower,upper=np.quantile(boot[valid],[.025,.975])
    gates=dict(candidate_macro_at_least_090=float(candidate.mean())>=.90,
               gain_at_least_0003=float(delta.mean())>=.003,
               positive_in_two_seeds=int((delta.mean(1)>0).sum())>=2,bootstrap_lower_positive=float(lower)>0)
    report=dict(status='PASS_SCREENING_ONLY' if all(gates.values()) else 'REJECT_FOR_ADVANCEMENT',
        control_mean_macro=float(control.mean()),candidate_mean_macro=float(candidate.mean()),mean_delta=float(delta.mean()),
        per_seed_macro_gains=delta.mean(1).tolist(),paired_95pct=[float(lower),float(upper)],
        per_target_gain=dict(zip(contract['targets'],delta.mean(0).tolist())),gates=gates,
        independent_cpu_replay_errors=replay_errors,worker_seconds=manifest['elapsed_seconds'],
        public_score=None,new_competition_submissions=0,eligible_for_competition_submission=False,
        training_cache_hash_verified_locally=False,expert_feature_hash_verified_locally=True,
        limitations=contract['limitations']+['Full1.7GB training feature cache retained remotely; local replay verifies the persisted58-study expert slice'],
        manifest_sha256=sha(out/'manifest.json'),auditor_sha256=sha(Path(__file__)))
    (LOCAL/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (LOCAL/'decision.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
