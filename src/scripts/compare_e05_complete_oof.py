"""Compare complete, independently audited E04/E05 held-out predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    'comparison': 'E05 uniform weak weights versus frozen E04 controls',
    'required_folds': [0,1,2,3,4], 'required_experts': 58,
    'prediction_rule': 'One independently held-out fold prediction per study',
    'metric': 'Unweighted mean of 12 target ROC AUCs on identical studies',
    'bootstrap': 'Paired report-group resampling; fixed predictions and labels',
    'bootstrap_seed': 20261010, 'bootstrap_draws': 2000,
    'minimum_valid_draws': 1000, 'interval_quantiles': [.025,.975],
    'owned_model_screening_floor': .90,
    'checkpoint_selection': 'Fixed final epoch14 EMA',
    'model_or_target_weight_search': False, 'competition_submission_enabled': False,
    'limitations': ['Gold58 has been repeatedly reused and is not independent validation',
                   'Original controls and Colab candidates use different GPU/CUDA environments',
                   'Report groups do not prove patient/site separation',
                   'Conditional bootstrap intervals do not capture training variation or leaderboard uncertainty'],
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def audited_predictions(path, expected_schema):
    audit = json.loads(path.read_text())
    allowed = {expected_schema}
    if expected_schema == 'e05_uniform_independent_fold_training_audit_v1':
        allowed.add('e05_uniform_qualified_reuse_fold_training_audit_v1')
    assert audit['schema'] in allowed and audit['status'] == 'PASS_COMPLETE_OOF_SMALL_OUTPUTS'
    if audit['schema'] == 'e05_uniform_qualified_reuse_fold_training_audit_v1':
        amendment = ROOT / 'runs/e05_confidence_ablation/e06_control_reuse_amendment.json'
        assert audit['reuse_provenance']['amendment_sha256'] == sha(amendment)
        assert audit['reuse_provenance']['fold'] == 2
    assert audit['complete_expert_oof'] and audit['expert_oof_rows'] == 58
    assert sorted(f['fold'] for f in audit['folds']) == list(range(5))
    receipt = audit['expert_oof_artifact']
    assert receipt['studies'] == 58 and receipt['prediction_rule'] == 'each study uses its independently heldout fold only'
    data = path.parent / receipt['filename']
    assert data.stat().st_size == receipt['bytes'] and sha(data) == receipt['sha256']
    frame = pd.read_csv(data, dtype={'StudyInstanceUID':str})
    assert len(frame) == 58 and frame.StudyInstanceUID.is_unique
    return frame, audit


def paired_interval(y, control, candidate, groups):
    unique = np.unique(groups); clusters = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(PROTOCOL['bootstrap_seed']); deltas = []
    for _ in range(PROTOCOL['bootstrap_draws']):
        ix = np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))])
        if ((y[ix].sum(0) == 0) | (y[ix].sum(0) == len(ix))).any(): continue
        deltas.append(float(roc_auc_score(y[ix],candidate[ix])-roc_auc_score(y[ix],control[ix])))
    assert len(deltas) >= PROTOCOL['minimum_valid_draws']
    return np.quantile(deltas,PROTOCOL['interval_quantiles']).tolist(),len(deltas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control-audit',type=Path)
    p.add_argument('--candidate-audit',type=Path)
    p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'runs/e05_confidence_ablation/paired_result.json')
    a = p.parse_args(); protocol_path=ROOT/'runs/e05_confidence_ablation/comparison_protocol.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text()) == PROTOCOL
    else: protocol_path.write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    if a.freeze_only:
        print(json.dumps({'status':'COMPARISON_FROZEN_BEFORE_ACCURACY','protocol_sha256':sha(protocol_path)}));return
    assert a.control_audit and a.candidate_audit, 'Both completed five-fold audits are required'
    control, ca = audited_predictions(a.control_audit,'e04_independent_fold_training_audit_v1')
    candidate, ea = audited_predictions(a.candidate_audit,'e05_uniform_independent_fold_training_audit_v1')
    frozen = json.loads((ROOT/'runs/E04_contract/e04_contract.json').read_text())
    labels=frozen['target_order']; experts={r['StudyInstanceUID']:r for r in frozen['rows'] if r['expert']}
    ids=sorted(experts)
    assert len(ids)==58 and set(control.StudyInstanceUID)==set(candidate.StudyInstanceUID)==set(ids)
    for frame in [control,candidate]: assert frame.columns.tolist()==['StudyInstanceUID',*labels]
    c=control.set_index('StudyInstanceUID').loc[ids,labels].to_numpy()
    e=candidate.set_index('StudyInstanceUID').loc[ids,labels].to_numpy()
    y=pd.read_csv(ROOT/'data/train.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids,labels].to_numpy()
    assert np.isin(y,[0,1]).all()
    for pred in [c,e]: assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    ac=roc_auc_score(y,c,average=None);ae=roc_auc_score(y,e,average=None)
    assert abs(ac.mean()-ca['expert_macro_auc'])<1e-9 and abs(ae.mean()-ea['expert_macro_auc'])<1e-9
    groups=np.array([experts[u]['report_group'] for u in ids])
    interval,draws=paired_interval(y,c,e,groups)
    result={'status':'COMPLETE_PAIRED_DIAGNOSTIC','control_auc':float(ac.mean()),'candidate_auc':float(ae.mean()),
            'delta':float((ae-ac).mean()),'paired_group_interval':interval,'valid_bootstrap_draws':draws,
            'report_groups':len(np.unique(groups)),
            'per_target':{t:{'control_auc':float(ac[i]),'candidate_auc':float(ae[i]),'delta':float(ae[i]-ac[i])}
                          for i,t in enumerate(labels)},
            'folds':{'control':ca['folds'],'candidate':ea['folds']},
            'screening_floor_met':float(ae.mean())>=.90,
            'positive_difference_interval':interval[0]>0,
            'decision':'REVIEW_FULL_EVIDENCE_BEFORE_ANY_SUBMISSION' if ae.mean()>=.90 and ae.mean()>ac.mean()
                       else 'DO_NOT_PROMOTE_ON_THIS_COMPARISON',
            'checkpoint_integrity_review_still_required':True,'competition_submission_enabled':False,
            'public_score_prediction':None,'protocol_sha256':sha(protocol_path),
            'control_audit_sha256':sha(a.control_audit),'candidate_audit_sha256':sha(a.candidate_audit),
            'limitations':PROTOCOL['limitations']}
    if 'reuse_provenance' in ea:
        keep = np.array([experts[u]['fold'] != 2 for u in ids])
        sensitivity_c = float(roc_auc_score(y[keep], c[keep]))
        sensitivity_e = float(roc_auc_score(y[keep], e[keep]))
        result['reuse_provenance'] = ea['reuse_provenance']
        result['sensitivity_excluding_reused_fold'] = {
            'studies': int(keep.sum()), 'control_auc': sensitivity_c,
            'candidate_auc': sensitivity_e, 'delta': sensitivity_e - sensitivity_c,
            'purpose': 'Descriptive sensitivity only; does not replace frozen five-fold decision'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_target','folds']},indent=2))


if __name__=='__main__': main()
