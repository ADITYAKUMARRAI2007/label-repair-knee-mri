"""Describe existing fracture ranking errors without fitting another ensemble."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/phase5b_dual_sota_20261011'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def credit(m):return (m>0).astype(float)+.5*(m==0)

def main():
    contract=read(ROOT/'research/public_refresh_20261011/qualification_protocol.json')
    ids=contract['expert_ids'];target='Fracture'
    paths={'A':RUN/'candidate_gold58_diagnostics/A.csv','C':RUN/'candidate_gold58_diagnostics/C.csv',
           'regional':ROOT/'research/competitive_phase45_20261011/model_analysis/regional_reader_oof.csv',
           'labels':ROOT/'data/train.csv','coverage':ROOT/'research/side_baseline_audit_20261010/metadata_coverage.csv'}
    frames={k:pd.read_csv(p,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids] for k,p in paths.items()}
    y=frames['labels'][target].to_numpy();assert np.isin(y,[0,1]).all()
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    score={k:frames[k][target].to_numpy() for k in ['A','C','regional']}
    credits={k:credit(v[pos,None]-v[None,neg]) for k,v in score.items()}
    for k,v in score.items():assert abs(credits[k].mean()-roc_auc_score(y,v))<1e-12
    delta=credits['C']-credits['A'];lost=delta<0;gained=delta>0
    base=ROOT/'research/public_refresh_20261011/qualification_output'
    native=base/read(base/'helios_latest_attempt.json')['attempt_directory']/'expert_diagnostic'
    logits={};log_paths=[native/f'worker{i}_logits.json' for i in range(2)]
    for p in log_paths:
        d=read(p);assert not set(d)&set(logits);logits.update(d)
    j=contract['target_order'].index(target)
    z=np.array([logits[u] for u in ids])[:,:,j]
    assert z.shape==(58,4) and np.isfinite(z).all()
    pairs=[]
    for pi,ni in np.argwhere(lost|gained):
        p,n=pos[pi],neg[ni];fold_margin=z[p]-z[n]
        pairs.append({'positive_uid':ids[p],'negative_uid':ids[n],
                      'change':'worsened' if lost[pi,ni] else 'improved',
                      'A_credit':credits['A'][pi,ni],'C_credit':credits['C'][pi,ni],
                      'regional_credit':credits['regional'][pi,ni],
                      'compact_folds_correct':int((fold_margin>0).sum()),
                      'compact_folds_wrong':int((fold_margin<0).sum()),
                      'positive_metadata_slots':int(frames['coverage'].iloc[p]['metadata_slots_present']),
                      'negative_metadata_slots':int(frames['coverage'].iloc[n]['metadata_slots_present'])})
    pd.DataFrame(pairs).to_csv(RUN/'fracture_changed_pairs.csv',index=False)
    cases=[]
    for index,uid in enumerate(ids):
        d=delta[np.where(pos==index)[0][0],:] if y[index] else delta[:,np.where(neg==index)[0][0]]
        cases.append({'StudyInstanceUID':uid,'expert_fracture_label':int(y[index]),
                      'A_score':score['A'][index],'C_score':score['C'][index],
                      'net_pair_credit_change':float(d.sum()),'worsened_pairs':int((d<0).sum()),
                      'improved_pairs':int((d>0).sum()),'metadata_slots':int(frames['coverage'].iloc[index]['metadata_slots_present'])})
    pd.DataFrame(cases).sort_values('net_pair_credit_change').to_csv(RUN/'fracture_case_triage.csv',index=False)
    result={'status':'DESCRIPTIVE_FRACTURE_ERROR_TRIAGE_COMPLETE','positive_studies':len(pos),'negative_studies':len(neg),
            'positive_negative_pairs':int(delta.size),'auc':{k:float(v.mean()) for k,v in credits.items()},
            'worsened_pair_count':int(lost.sum()),'improved_pair_count':int(gained.sum()),
            'net_pair_credit_change':float(delta.sum()),
            'regional_correct_on_worsened_pairs':int((credits['regional'][lost]==1).sum()),
            'all_four_compact_folds_wrong_on_worsened_pairs':sum(r['compact_folds_wrong']==4 for r in pairs if r['change']=='worsened'),
            'affected_cases':sum(c['worsened_pairs']>0 for c in cases),
            'inputs_sha256':{k:sha(p) for k,p in paths.items()}|{p.name:sha(p) for p in log_paths},
            'weight_or_threshold_search':False,'new_model_inference':False,'new_submission':False,
            'interpretation_limits':['Post hoc analysis on reused Gold58; cannot select a fracture override or prove generalization.',
                                     'Metadata slot coverage does not prove actual pixel visibility or a cause of model error.',
                                     'Pair counts are dependent: the same study participates in multiple pairs.',
                                     'Regional model correctness on selected lost pairs is not evidence that adding it improves overall AUC.'],
            'next_experiment_gate':'Use independent grouped validation to compare a targeted fracture reader or supervision change; no per-class weight tuning on these cases.'}
    (RUN/'fracture_error_triage.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs_sha256','interpretation_limits']}))
if __name__=='__main__':main()
