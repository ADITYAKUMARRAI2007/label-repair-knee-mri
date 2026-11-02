"""Describe preserved four-fold logits; no refitting or fold selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/phase5b_dual_sota_20261011'
def read(p):return json.loads(p.read_text())
def main():
    qc=ROOT/'research/public_refresh_20261011';c=read(qc/'qualification_protocol.json')
    folder=qc/'qualification_output'/read(qc/'qualification_output/helios_latest_attempt.json')['attempt_directory']/'expert_diagnostic'
    logits={}
    for gi in range(2):
        a=read(folder/f'worker{gi}_logits.json');assert not set(a)&set(logits);logits.update(a)
    ids=c['expert_ids'];labels=c['target_order'];assert set(logits)==set(ids)
    z=np.array([logits[u] for u in ids]);assert z.shape==(58,4,12) and np.isfinite(z).all()
    names=read(qc/'compact_docs/manifest.json')['models'];assert names==sorted(names)
    base=pd.read_csv(ROOT/c['baseline_path'],dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids,labels].to_numpy()
    observed=pd.read_csv(folder/'predictions.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids,labels].to_numpy()
    err=float(abs(expit(z.mean(1))-observed).max());assert err<.001
    y=pd.read_csv(ROOT/'data/train.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids,labels].to_numpy()
    target=OUT/'compact_fold_diagnostics';target.mkdir(exist_ok=True);metrics={};rows=[]
    for i,n in enumerate(names):
        p=expit(z[:,i]);f=pd.DataFrame(p,columns=labels);f.insert(0,'StudyInstanceUID',ids);f.to_csv(target/(n+'.csv'),index=False)
        auc=roc_auc_score(y,z[:,i],average=None)
        metrics[n]={'macro_auc_from_native_fold_logits':float(auc.mean()),'per_class_auc':dict(zip(labels,map(float,auc)))}
        for j,label in enumerate(labels):rows.append({'checkpoint':n,'target':label,'pearson_N02_vs_fold_probability':float(pearsonr(base[:,j],p[:,j]).statistic),'spearman_N02_vs_fold':float(spearmanr(base[:,j],z[:,i,j]).statistic)})
    pd.DataFrame(rows).to_csv(OUT/'compact_fold_correlations.csv',index=False)
    result={'status':'PRESERVED_NATIVE_INDIVIDUAL_FOLD_LOGITS_ANALYZED','studies':58,'folds':4,'checkpoint_order_proof':'Original compact_infer.load_group uses sorted glob; filenames exactly match strict worker receipts.','mean_logit_sigmoid_vs_saved_FP16_probabilities_max_error':err,'metrics':metrics,'independent_validation':False,'fold_selection_or_weight_search':False,'primary_C_unchanged':True,'limitation':'Only reused Gold58; small numeric difference is half-precision mean/sigmoid rounding. Individual N02 old-reader folds were not separately dumped.'}
    (OUT/'compact_fold_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'fold_auc':{k:v['macro_auc_from_native_fold_logits'] for k,v in metrics.items()},'reconstruction_error':err,'fold_selection':False}))
if __name__=='__main__':main()
