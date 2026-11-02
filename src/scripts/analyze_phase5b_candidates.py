"""Preserve fixed A--E candidates and descriptive complementarity on reused Gold58."""
from datetime import datetime,timezone
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import roc_auc_score
from compare_e05_complete_oof import paired_interval

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/phase5b_dual_sota_20261011'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rank(a):return pd.DataFrame(a).rank(method='average',pct=True).to_numpy()

def main():
    qc=ROOT/'research/public_refresh_20261011';contract=read(qc/'qualification_protocol.json')
    native=qc/'qualification_output'/read(qc/'qualification_output/helios_latest_attempt.json')['attempt_directory']
    ids=contract['expert_ids'];labels=contract['target_order']
    paths={'N02':ROOT/contract['baseline_path'],
           'same_raptor_anchor_rank':ROOT/'runs/colab_n03/output/fixed_reader_blend/anchor.csv',
           'old_convnext_reader':ROOT/'runs/colab_n03/output/fixed_reader_blend/reader.csv',
           'compact':native/'expert_diagnostic/predictions.csv',
           'regional_phase5':ROOT/'research/competitive_phase45_20261011/model_analysis/regional_reader_oof.csv'}
    protocol={'recorded_utc':datetime.now(timezone.utc).isoformat(),
              'inputs_sha256':{k:sha(p) for k,p in paths.items()},'expert_ids':ids,'target_order':labels,
              'cohort':'Repeatedly reusedGold58; descriptive comparisons only; NOT eligible independent validation',
              'candidates':{'A':'N02 unchanged saved scores','B':'0.7 rank(sameCoAtNetanchor)+0.3 rank(compact)',
                            'C':'0.7 rank(N02)+0.3 rank(compact); primary frozen before prior qualification results',
                            'D':'0.7 rank(N02)+0.3 rank(B); secondary fixed published-ratio composition, descriptive only',
                            'E':'Identical to B: published recipe already excludes unused0945 and OFLclassifier; no duplicateCoAtNet'},
              'weight_search':False,'weight_sensitivity_sweep':False,
              'reason_no_sweep':'No independent eligible validation available',
              'B_status':'Published V4 fixed recipe applied to preserved samecheckpointanchor ranking; exactV3 and our public0.954 not reproduced',
              'B_native_raw_probabilities_for_experts_available':False,
              'candidate_choice':'C primary selected by its original fixed protocol; secondary Gold scores do not select weights',
              'automatic_submission':False}
    pp=OUT/'candidate_protocol.json'
    if pp.exists():assert read(pp)==protocol, 'Preserve the fixed candidate protocol'
    else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
    data={k:pd.read_csv(p,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids,labels].to_numpy() for k,p in paths.items()}
    for a in data.values():assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    raw=data['N02'];r=rank(data['same_raptor_anchor_rank']);c=rank(data['compact'])
    published=.7*r+.3*c
    candidates={'A':raw,'B':published,'C':.7*rank(raw)+.3*c,'D':.7*rank(raw)+.3*rank(published),'E':published.copy()}
    cohort=OUT/'candidate_gold58_diagnostics';cohort.mkdir(exist_ok=True)
    for name,a in candidates.items():
        f=pd.DataFrame(a,columns=labels);f.insert(0,'StudyInstanceUID',ids);f.to_csv(cohort/(name+'.csv'),index=False)
    y=pd.read_csv(ROOT/'data/train.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids,labels].to_numpy()
    assert np.isin(y,[0,1]).all()
    rows=read(ROOT/'runs/E04_contract/e04_contract.json')['rows'];lookup={r['StudyInstanceUID']:r for r in rows}
    groups=np.array([lookup[u]['report_group'] for u in ids])
    base_auc=roc_auc_score(y,raw,average=None);metrics={}
    for k,a in candidates.items():
        auc=roc_auc_score(y,a,average=None)
        if k=='A':ci,draws=[0.,0.],0
        elif k=='E':ci,draws=metrics['B']['paired_delta_ci95'],metrics['B']['valid_bootstrap_draws']
        else:ci,draws=paired_interval(y,raw,a,groups)
        metrics[k]={'macro_auc':float(auc.mean()),'delta_vs_N02':float(auc.mean()-base_auc.mean()),
                    'paired_delta_ci95':ci,'valid_bootstrap_draws':draws,
                    'per_class':{l:float(auc[j]) for j,l in enumerate(labels)},
                    'filename':str((cohort/(k+'.csv')).relative_to(ROOT)),'sha256':sha(cohort/(k+'.csv'))}
    assert abs(metrics['C']['macro_auc']-read(qc/'qualification_result.json')['fixed_blend_macro_auc'])<1e-12
    correlations=[]
    for other in ['N02','same_raptor_anchor_rank','old_convnext_reader','regional_phase5']:
        a=data[other];b=data['compact'];ar=rank(a);br=rank(b)
        for j,l in enumerate(labels):
            correlations.append({'comparator':other,'target':l,'studies':58,
                                 'pearson_stored_scores':float(pearsonr(a[:,j],b[:,j]).statistic),
                                 'spearman':float(spearmanr(a[:,j],b[:,j]).statistic),
                                 'mean_absolute_percentile_rank_disagreement':float(np.mean(abs(ar[:,j]-br[:,j]))),
                                 'comparator_mean':float(a[:,j].mean()),'compact_mean':float(b[:,j].mean()),
                                 'comparator_std':float(a[:,j].std()),'compact_std':float(b[:,j].std()),
                                 'score_scale':'N02/anchor are ranking scores; compact/reader/ROI are probabilities. Pearson/distributions do not establish calibration or accuracy.'})
    pd.DataFrame(correlations).to_csv(OUT/'component_correlations.csv',index=False)
    phase5=read(ROOT/'research/competitive_phase45_20261011/model_analysis/results.json')
    result={'status':'COMPLETE_FIXED_CANDIDATES_AND_COMPLEMENTARITY_DIAGNOSTICS',
            'recorded_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(pp),'metrics':metrics,
            'descriptive_reused_gold_ranking':sorted(candidates,key=lambda k:metrics[k]['macro_auc'],reverse=True),
            'independent_validation_ranking':None,'primary_recommendation':'C; retainB as unchanged published-recipe comparator; E aliasesB',
            'N02_vs_compact_mean_spearman':float(np.mean([r['spearman'] for r in correlations if r['comparator']=='N02'])),
            'phase5_status':'COMPLETE; ten checkpoints and completeOOF preserved',
            'phase5_summary':phase5['summary'],
            'no_private_test_labels':True,'no_test_disagreement_accuracy_claim':True,
            'public0_954_reproduced':False,'actual_new_submission':False,
            'OFL_classifier_decision':'Not used in this published recipe, so no evidence of its complementary predictive contribution. Do not launchOFLtraining on that premise.',
            'targeted_next_direction':'Anatomy-guidedmultiscaleConvNeXtSmall and severity-state readout; evaluated sourceexactnewreader before any training extension.',
            'limitations':contract['limitations']}
    (OUT/'candidate_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'metrics':{k:{n:v for n,v in m.items() if n not in ['per_class','filename','sha256']} for k,m in metrics.items()},'mean_N02_compact_spearman':result['N02_vs_compact_mean_spearman'],'public0_954_reproduced':False}))

if __name__=='__main__':main()
