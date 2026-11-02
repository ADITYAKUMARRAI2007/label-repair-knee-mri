"""Verify the native execution chain and score the single frozen N02/compact blend."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from compare_e05_complete_oof import paired_interval

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/public_refresh_20261011'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    output=RUN/'qualification_output'
    current=read(output/'helios_latest_attempt.json')['attempt_directory']
    assert Path(current).name==current
    attempt=output/current
    assert not (attempt/'failure.json').exists(), 'Failed execution cannot qualify'
    observation=read(RUN/'qualification_observation.json')
    assert observation['native_status']=='COMPLETE' and observation['code_cell_sha256_verified']
    contract=read(RUN/'qualification_protocol.json')
    receipt=read(attempt/'execution_receipt.json')
    assert receipt['status']=='COMPLETE_BOUNDED_HELIOS_NATIVE_QUALIFICATION'
    assert receipt['contract_sha256']==sha(RUN/'qualification_protocol.json')
    assert receipt['source_files_sha256']['qualification_contract.json']==receipt['contract_sha256']
    for name,h in contract['source_sha256'].items(): assert receipt['source_files_sha256'][name]==h
    assert sha(attempt/'source/qualification_contract.json')==receipt['contract_sha256']
    assert receipt['manifest_sha256']==contract['manifest_sha256']
    assert receipt['code_sha256']==contract['code_sha256']
    assert not receipt['competition_submission_enabled'] and not receipt['test_time_fitting']
    assert not receipt['gold_labels_or_reports_read_by_inference']
    assert 0<receipt['elapsed_seconds']<=contract['maximum_parent_seconds']
    assert receipt['resource_studies']==9 and receipt['expert_studies']==58
    assert 0<=receipt['public_replay_max_abs_error']<=contract['prediction_replay_max_absolute_error']
    labels=contract['target_order']; ids=contract['expert_ids']; mem=[]; times=[]
    for stage,n in [('resource_pilot',9),('public_replay',3),('expert_diagnostic',58)]:
        folder=attempt/stage; execution=read(folder/'execution.json')
        assert execution['status']=='COMPLETE_ZERO_FALLBACK_NATIVE_PREDICTIONS'
        assert execution['studies']==n and execution['prediction_sha256']==sha(folder/'predictions.csv')
        assigned=[]
        for gi in range(2):
            if not (folder/f'worker{gi}').exists():continue
            wr=read(folder/f'worker{gi}/receipt.json')
            uids=read(folder/f'worker{gi}_uids.json');assigned+=uids
            assert wr['status']=='PASS' and not wr['fallback'] and not wr['timed_out']
            assert wr['scored']==wr['studies']==len(uids)
            assert wr['models']==list(read(RUN/'compact_docs/manifest.json')['models'])
            assert sha(folder/f'worker{gi}/receipt.json') in execution['worker_receipts_sha256']
            assert len(wr['per_study'])==len(uids)
            for p in wr['per_study']:
                assert p[0] in uids and p[1]>0 and all(np.isfinite(p[2:]))
                if stage=='expert_diagnostic':times.append(sum(p[2:]))
            mem.append(read(folder/f'worker{gi}/gpu_memory.json')['peak_allocated_bytes'])
            f=pd.read_csv(folder/f'worker{gi}/predictions.csv',dtype={'StudyInstanceUID':str})
            assert f.StudyInstanceUID.is_unique and set(f.StudyInstanceUID)==set(uids)
        assert len(assigned)==len(set(assigned))==n
    base=ROOT/contract['baseline_path']; assert sha(base)==contract['baseline_sha256']
    candidate=attempt/'expert_diagnostic/predictions.csv'
    assert sha(candidate)==receipt['expert_predictions_sha256']
    c=pd.read_csv(candidate,dtype={'StudyInstanceUID':str})
    assert list(c.columns)==['StudyInstanceUID',*labels] and c.StudyInstanceUID.is_unique
    assert set(c.StudyInstanceUID)==set(ids)
    c=c.set_index('StudyInstanceUID').loc[ids,labels].to_numpy()
    b=pd.read_csv(base,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids,labels].to_numpy()
    y=pd.read_csv(ROOT/'data/train.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids,labels].to_numpy()
    assert np.isin(y,[0,1]).all()
    for p in [b,c]: assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert (c.std(0)>1e-6).all()
    br=pd.DataFrame(b).rank(method='average',pct=True).to_numpy()
    cr=pd.DataFrame(c).rank(method='average',pct=True).to_numpy()
    recipe=contract['primary_blend']; assert recipe['baseline_weight']==.7 and recipe['compact_weight']==.3
    mix=.7*br+.3*cr
    ba=roc_auc_score(y,b,average=None);ca=roc_auc_score(y,c,average=None);ma=roc_auc_score(y,mix,average=None)
    rows=read(ROOT/'runs/E04_contract/e04_contract.json')['rows'];byid={r['StudyInstanceUID']:r for r in rows}
    groups=np.array([byid[u]['report_group'] for u in ids])
    ci,draws=paired_interval(y,br,mix,groups)
    delta=float(ma.mean()-ba.mean())
    per={}
    for j,k in enumerate(labels):
        pos=np.flatnonzero(y[:,j]==1);neg=np.flatnonzero(y[:,j]==0)
        a=b[pos,j,None]-b[neg,j][None,:];d=mix[pos,j,None]-mix[neg,j][None,:]
        per[k]={'baseline_auc':float(ba[j]),'compact_auc':float(ca[j]),'fixed_blend_auc':float(ma[j]),
                'blend_delta':float(ma[j]-ba[j]),'positive_negative_pairs':int(a.size),
                'baseline_wrong_pairs_repaired':int(((a<0)&(d>0)).sum()),
                'baseline_correct_pairs_broken':int(((a>0)&(d<0)).sum()),
                'ties':'Reported AUC gives ties half credit; repair counts above are strict reversals only.'}
    qualifies=delta>.001 and ci[0]>0
    result={'status':'COMPLETE_SOURCE_VERIFIED_FIXED_HELIOS_BLEND_DIAGNOSTIC',
            'observed_utc':datetime.now(timezone.utc).isoformat(),'expert_studies':58,
            'baseline_macro_auc':float(ba.mean()),'compact_macro_auc':float(ca.mean()),
            'fixed_blend_macro_auc':float(ma.mean()),'fixed_blend_delta':delta,
            'paired_report_group_95_interval':ci,'valid_bootstrap_draws':draws,'per_target':per,
            'resource':{'expert_compute_seconds_mean':float(np.mean(times)),
                        'expert_compute_seconds_p95':float(np.quantile(times,.95)),
                        'maximum_main_worker_cuda_peak_bytes':int(max(mem)),
                        'aggregate_gpu_memory_samples':'qualification_output/'+current+'/resource_samples.json',
                        'scope':'Worker compute excludes some loader waiting and startup; whole-run wall time is in execution receipt.'},
            'protocol_sha256':receipt['contract_sha256'],'execution_receipt_sha256':sha(attempt/'execution_receipt.json'),
            'candidate_predictions_sha256':sha(candidate),'baseline_sha256':sha(base),
            'decision':'REVIEW_NATIVE_COMBINED_SERVING_BEFORE_SUBMISSION' if qualifies else 'REJECT_THIS_FIXED_BLEND_NO_SUBMISSION',
            'weight_search_performed':False,'independent_clinical_validation':False,
            'expected_public_score':None,'competition_submission_enabled':False,
            'limitations':contract['limitations']}
    (RUN/'qualification_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_target','limitations','resource']}))

if __name__=='__main__':main()
