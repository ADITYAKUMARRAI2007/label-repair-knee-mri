#!/usr/bin/env python3
"""One fixed CPU comparison on cached features; no Kaggle submission capability."""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from spatial_pooling_head import SpatialHead

OUT = ROOT / 'runs/uncertainty_weight_probe_v1'
FEATURES = ROOT / 'runs/grade_supervision_probe_outputs_v2/grade_probe'
CONTRACT = OUT / 'contract.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def table(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    result = {r['StudyInstanceUID']: r for r in rows}
    assert len(result) == len(rows)
    return result


def prepare():
    if CONTRACT.exists():
        raise RuntimeError('Preserve existing frozen comparison')
    audit = json.loads((ROOT / 'runs/report_uncertainty_audit_v1/audit.json').read_text())
    timing = json.loads((ROOT / 'runs/report_uncertainty_audit_v1/cpu_timing.json').read_text())
    assert timing['projected_six_fits_seconds_with_20pct_margin'] < 1200
    paths = [Path(__file__), ROOT / 'training/spatial_pooling_head.py',
             ROOT / 'runs/report_uncertainty_audit_v1/audit.json',
             ROOT / 'runs/report_uncertainty_audit_v1/cpu_timing.json',
             ROOT / 'runs/E03_contract/split.json', ROOT / 'data/train.csv',
             ROOT / 'runs/grade_supervision_trial_v1/control_targets.csv',
             ROOT / 'runs/grade_supervision_trial_v1/shared_weights.csv',
             ROOT / 'labels/pilkwang_rsna-knee-llm-labels/report_labels_v2.csv']
    paths += [FEATURES / n for n in ['features.npy', 'token_masks.npy', 'study_ids.npy']]
    contract = {
        'schema': 'uncertainty_weight_probe_v1', 'seeds': [20261008,20261009,20261010],
        'epochs': 30, 'batch_size': 64, 'device': 'cpu', 'threads': 2,
        'seconds_cap': 1200, 'targets': list(audit['by_target'])[:-1],
        'candidate': 'For source verdict UNK only, min(original weight,0.25); all target probabilities unchanged.',
        'cap': .25, 'cap_rationale': 'Existing recipe minimum confidence weight, fixed without expert-score selection.',
        'loss': 'mean(weight * BCEWithLogits); no per-target or global renormalization',
        'control': 'Original four-source median probabilities and disagreement weights.',
        'checkpoint_selection': 'Final epoch30 only; all six fits before accessing expert scores.',
        'optimizer': {'name':'AdamW','lr':3e-4,'weight_decay':.01,'gradient_clip':3},
        'gates': {'candidate_macro_min':.90,'paired_macro_gain_min':.003,
                  'positive_seed_count_min':2,'paired_bootstrap_lower_min':0},
        'bootstrap': {'seed':20261010,'draws':2000},
        'limitations': ['Reused Gold58 is a rejection screen, not independent performance validation.',
                       'Source UNK combines unmentioned and uncertain and can be mistaken.',
                       'Weight attenuation also changes total gradient magnitude; no isolated semantic-causality claim.',
                       'Frozen-feature head accuracy does not establish full fine-tuning performance or B03 complementarity.'],
        'competition_submission_enabled': False,
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write(CONTRACT, contract)
    print(json.dumps({'status':'FROZEN_NOT_RUN','contract':str(CONTRACT),'seconds_cap':1200}),flush=True)


def run():
    started = time.monotonic()
    c = json.loads(CONTRACT.read_text())
    receipt = OUT / 'launch.json'
    with receipt.open('x') as f:
        import os
        json.dump({'pid':os.getpid(),'contract_sha256':sha(CONTRACT),'started_unix':time.time()},f)
    for name, expected in c['inputs'].items():
        assert sha(ROOT / name) == expected, name
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    def guard():
        if time.monotonic() - started > c['seconds_cap']:
            raise RuntimeError('CPU comparison exceeded frozen time cap')
    features = np.load(FEATURES / 'features.npy', mmap_mode='r')
    masks = np.load(FEATURES / 'token_masks.npy')
    ids = np.load(FEATURES / 'study_ids.npy').astype(str)
    assert features.shape == (4407,20,2048) and features.dtype == np.float16
    assert masks.shape == (4407,20) and masks.dtype == bool and masks.any(1).all()
    assert len(set(ids)) == 4407
    split = json.loads((ROOT / 'runs/E03_contract/split.json').read_text())
    gold = set(split['gold']); excluded = gold | set(split['excluded_gold_group_weak'])
    train = np.array([i for i,u in enumerate(ids) if u not in excluded])
    expert = np.array([i for i,u in enumerate(ids) if u in gold])
    assert len(train) == 4345 and len(expert) == 58
    target_rows = table(ROOT / 'runs/grade_supervision_trial_v1/control_targets.csv')
    weight_rows = table(ROOT / 'runs/grade_supervision_trial_v1/shared_weights.csv')
    verdict_rows = table(ROOT / 'labels/pilkwang_rsna-knee-llm-labels/report_labels_v2.csv')
    labels = np.zeros((4407,12),np.float32); control = labels.copy(); candidate = labels.copy()
    unknown = np.zeros((4407,12),bool)
    for i in train:
        for j,t in enumerate(c['targets']):
            labels[i,j] = float(target_rows[ids[i]][t])
            control[i,j] = float(weight_rows[ids[i]][t])
            unknown[i,j] = verdict_rows[ids[i]][t+'__verdict'] == 'UNK'
            candidate[i,j] = min(control[i,j],c['cap']) if unknown[i,j] else control[i,j]
    assert unknown.sum() == 14256
    assert np.array_equal(candidate[~unknown],control[~unknown])
    assert (candidate <= control).all() and (candidate[train] > 0).all()
    for name, array in {'labels':labels,'control_weights':control,'candidate_weights':candidate,
                        'train_indices':train,'expert_ids':ids[expert]}.items():
        np.save(OUT / (name+'.npy'),array)
    fits = []; predictions = {}
    def predict(model):
        blocks = []
        model.eval()
        with torch.inference_mode():
            for start in range(0,len(expert),16):
                b = expert[start:start+16]
                blocks.append(model(torch.from_numpy(np.asarray(features[b])),torch.from_numpy(masks[b])).sigmoid().numpy())
        return np.concatenate(blocks)
    for seed in c['seeds']:
        initial_hashes = []
        for arm,w in [('control',control),('candidate',candidate)]:
            guard();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            model = SpatialHead()
            initial = hashlib.sha256(b''.join(t.numpy().tobytes() for t in model.state_dict().values())).hexdigest()
            initial_hashes.append(initial)
            opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            history = []
            for epoch in range(c['epochs']):
                torch.manual_seed(seed+epoch*100)
                order = np.random.default_rng(seed+epoch).permutation(train)
                model.train();losses=[]
                for start in range(0,len(order),64):
                    guard(); b=order[start:start+64]
                    opt.zero_grad(set_to_none=True)
                    logits = model(torch.from_numpy(np.asarray(features[b])),torch.from_numpy(masks[b]))
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(labels[b]),reduction='none')*torch.from_numpy(w[b])).mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
                    losses.append(float(loss.detach()))
                history.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
                if (epoch+1)%5 == 0:
                    progress={'arm':arm,'seed':seed,'epoch':epoch+1,'elapsed_seconds':time.monotonic()-started}
                    write(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
            pred = predict(model); assert np.isfinite(pred).all()
            path = OUT/f'{arm}_{seed}_head.pt'; torch.save(model.state_dict(),path)
            replay = SpatialHead();replay.load_state_dict(torch.load(path,weights_only=True))
            error = float(np.abs(predict(replay)-pred).max());assert error <= 1e-6
            predictions[seed,arm] = pred
            np.save(OUT/f'{arm}_{seed}_expert.npy',pred)
            fit={'arm':arm,'seed':seed,'history':history,'initial_state_sha256':initial,'replay_max_error':error}
            fits.append(fit);write(OUT/f'{arm}_{seed}_fit.json',fit)
        assert len(set(initial_hashes)) == 1
    # Expert finding values first accessed only after all six fixed fits are complete.
    official = table(ROOT / 'data/train.csv')
    truth = np.array([[float(official[u][t]) for t in c['targets']] for u in ids[expert]])
    assert np.isin(truth,[0,1]).all()
    scores={arm:np.array([[roc_auc_score(truth[:,j],predictions[seed,arm][:,j]) for j in range(12)] for seed in c['seeds']]) for arm in ['control','candidate']}
    counts=np.random.default_rng(c['bootstrap']['seed']).multinomial(58,np.full(58,1/58),size=2000)
    boot=np.zeros(2000);valid=np.ones(2000,bool)
    for j in range(12):
        pos,neg=truth[:,j]==1,truth[:,j]==0
        p,n=counts[:,pos],counts[:,neg];den=p.sum(1)*n.sum(1);valid &= den>0
        for seed in c['seeds']:
            difference=np.zeros((pos.sum(),neg.sum()))
            for arm,sign in [('candidate',1),('control',-1)]:
                pred=predictions[seed,arm][:,j];d=pred[pos,None]-pred[None,neg]
                difference+=sign*((d>0)+.5*(d==0))
            numerator=np.einsum('bi,ij,bj->b',p,difference,n,optimize=True)
            boot+=np.divide(numerator,den,out=np.zeros(2000),where=den>0)/36
    assert valid.sum() >= 1900
    ci=np.quantile(boot[valid],[.025,.975]);delta=scores['candidate']-scores['control']
    gates={'candidate_macro_at_least_090':float(scores['candidate'].mean())>=.90,
           'gain_at_least_0003':float(delta.mean())>=.003,
           'positive_in_two_seeds':int((delta.mean(1)>0).sum())>=2,
           'bootstrap_lower_positive':float(ci[0])>0}
    result={'status':'SCREEN_PASS_REQUIRES_INDEPENDENT_AUDIT' if all(gates.values()) else 'REJECT_FOR_ADVANCEMENT',
            'contract_sha256':sha(CONTRACT),'control_mean_macro':float(scores['control'].mean()),
            'candidate_mean_macro':float(scores['candidate'].mean()),'mean_delta':float(delta.mean()),
            'seed_deltas':delta.mean(1).tolist(),'paired_95pct':ci.tolist(),'gates':gates,
            'per_target_gain':dict(zip(c['targets'],delta.mean(0).tolist())),
            'scores':{a:v.tolist() for a,v in scores.items()},'all_six_fits_complete':True,
            'worker_seconds':time.monotonic()-started,'public_score':None,'new_submissions':0,
            'artifacts':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in {'.npy','.pt'}},
            'limitations':c['limitations']}
    write(OUT/'result.json',result);print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:
        try:run()
        except BaseException as error:
            write(OUT/'failure.json',{'error':type(error).__name__,'message':str(error),'no_accuracy_claim':True})
            raise
