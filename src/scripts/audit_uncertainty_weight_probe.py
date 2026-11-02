#!/usr/bin/env python3
"""Verify UW01 arrays, saved fits, replay and statistics independently of its runner."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/uncertainty_weight_probe_v1'
SOURCE = ROOT / 'runs/grade_supervision_probe_outputs_v2/grade_probe'
sys.path.insert(0,str(ROOT/'training'))
from spatial_pooling_head import SpatialHead


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def rows(path):
    with path.open() as f:return {r['StudyInstanceUID']:r for r in csv.DictReader(f)}


def main():
    contract=json.loads((OUT/'contract.json').read_text())
    result=json.loads((OUT/'result.json').read_text())
    assert result['all_six_fits_complete']
    assert result['contract_sha256']==sha(OUT/'contract.json')
    for name,expected in contract['inputs'].items():assert sha(ROOT/name)==expected,name
    for name,expected in result['artifacts'].items():assert sha(OUT/name)==expected,name
    # Compare actual source hashes to the earlier independently audited feature manifest.
    prior=json.loads((ROOT/'runs/orthofoundation_probe_source.json').read_text())['contract']['previous_features']
    for name in ['features.npy','token_masks.npy','study_ids.npy']:
        assert sha(SOURCE/name)==prior[name]['sha256']
    ids=np.load(SOURCE/'study_ids.npy').astype(str)
    features=np.load(SOURCE/'features.npy',mmap_mode='r');masks=np.load(SOURCE/'token_masks.npy')
    split=json.loads((ROOT/'runs/E03_contract/split.json').read_text())
    excluded=set(split['gold'])|set(split['excluded_gold_group_weak'])
    train=np.load(OUT/'train_indices.npy')
    assert np.array_equal(train,np.array([i for i,u in enumerate(ids) if u not in excluded]))
    expert_ids=np.load(OUT/'expert_ids.npy').astype(str)
    assert len(expert_ids)==58 and set(expert_ids)==set(split['gold'])
    lookup={u:i for i,u in enumerate(ids)};positions=np.array([lookup[u] for u in expert_ids])
    target=rows(ROOT/'runs/grade_supervision_trial_v1/control_targets.csv')
    weight=rows(ROOT/'runs/grade_supervision_trial_v1/shared_weights.csv')
    verdict=rows(ROOT/'labels/pilkwang_rsna-knee-llm-labels/report_labels_v2.csv')
    y=np.load(OUT/'labels.npy');w=np.load(OUT/'control_weights.npy');v=np.load(OUT/'candidate_weights.npy')
    modified=0
    for i in train:
        for j,t in enumerate(contract['targets']):
            assert y[i,j]==np.float32(target[ids[i]][t])
            expected=float(weight[ids[i]][t])
            assert w[i,j]==np.float32(expected)
            if verdict[ids[i]][t+'__verdict']=='UNK':expected=min(expected,.25)
            assert v[i,j]==np.float32(expected)
            modified+=int(w[i,j]!=v[i,j])
    assert not np.count_nonzero(y[positions]) and not np.count_nonzero(v[positions])
    truth_rows=rows(ROOT/'data/train.csv')
    truth=np.array([[float(truth_rows[u][t]) for t in contract['targets']] for u in expert_ids])
    torch.set_num_threads(2)
    scores={a:[] for a in ['control','candidate']};predictions={};replay_errors={}
    for seed in contract['seeds']:
        initial=[]
        for arm in scores:
            fit=json.loads((OUT/f'{arm}_{seed}_fit.json').read_text())
            assert [r['epoch'] for r in fit['history']]==list(range(1,31))
            assert np.isfinite([r['loss'] for r in fit['history']]).all()
            initial.append(fit['initial_state_sha256'])
            model=SpatialHead().eval()
            model.load_state_dict(torch.load(OUT/f'{arm}_{seed}_head.pt',map_location='cpu',weights_only=True),strict=True)
            blocks=[]
            with torch.inference_mode():
                for start in range(0,58,16):
                    b=positions[start:start+16]
                    blocks.append(model(torch.from_numpy(np.asarray(features[b])),torch.from_numpy(masks[b])).sigmoid().numpy())
            saved=np.load(OUT/f'{arm}_{seed}_expert.npy')
            assert saved.shape==(58,12) and np.isfinite(saved).all()
            error=float(np.abs(np.concatenate(blocks)-saved).max());assert error<=1e-6
            replay_errors[f'{arm}_{seed}']=error
            auc=np.array([roc_auc_score(truth[:,j],saved[:,j]) for j in range(12)])
            scores[arm].append(auc);predictions[seed,arm]=saved
        assert len(set(initial))==1
    scores={a:np.array(v) for a,v in scores.items()}
    for a,v in scores.items():assert np.allclose(v,result['scores'][a],atol=1e-12,rtol=0)
    # Independent sampled-index sklearn bootstrap, using a separate fixed RNG.
    # Check direction/interval agreement, not exact reproduction of producer multinomial draws.
    rng=np.random.default_rng(20261015);gains=[]
    # 200 draws are an independent sanity check; producer's frozen 2000-draw CI remains the decision statistic.
    for draw in range(200):
        indices=rng.integers(0,58,58)
        if any(len(np.unique(truth[indices,j]))<2 for j in range(12)):continue
        seed_gains=[]
        for seed in contract['seeds']:
            ca=roc_auc_score(truth[indices],predictions[seed,'candidate'][indices],average='macro')
            co=roc_auc_score(truth[indices],predictions[seed,'control'][indices],average='macro')
            seed_gains.append(ca-co)
        gains.append(float(np.mean(seed_gains)))
    assert len(gains)>=180
    delta=scores['candidate']-scores['control']
    assert abs(float(delta.mean())-result['mean_delta'])<1e-12
    gates={'candidate_macro_at_least_090':float(scores['candidate'].mean())>=.90,
           'gain_at_least_0003':float(delta.mean())>=.003,
           'positive_in_two_seeds':int((delta.mean(1)>0).sum())>=2,
           'bootstrap_lower_positive':result['paired_95pct'][0]>0}
    assert gates==result['gates']
    audit={'status':'PASS_ARRAY_FIT_REPLAY_AND_AUC_CHECKS','candidate_disposition':result['status'],
           'changed_training_weights':modified,'all_six_heads_replayed':True,'replay_errors':replay_errors,
           'control_mean_macro':float(scores['control'].mean()),'candidate_mean_macro':float(scores['candidate'].mean()),
           'mean_delta':float(delta.mean()),'independent_200_draw_95pct':np.quantile(gains,[.025,.975]).tolist(),
           'independent_bootstrap_note':'Separate RNG sanity check, not replacement of frozen 2000-draw decision statistic.',
           'expert_training_rows':0,'new_competition_submissions':0,'auditor_sha256':sha(Path(__file__)),
           'result_sha256':sha(OUT/'result.json')}
    (OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
