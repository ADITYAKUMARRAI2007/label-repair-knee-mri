"""Check collected native combined inference and preserve a reviewable candidate."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/phase5b_dual_sota_20261011'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    observation=read(RUN/'serving_observation.json')
    assert observation['native_status']=='COMPLETE' and observation['code_cell_sha256_verified']
    output=RUN/'serving_output';marker=read(output/'phase5b_latest_attempt.json')
    name=marker['attempt_directory'];assert Path(name).name==name and name.startswith('phase5b_attempt_')
    attempt=output/name
    assert not (attempt/'failure.json').exists()
    receipt=read(attempt/'execution_receipt.json');launch=read(RUN/'serving_launch.json');contract=read(RUN/'serving_contract.json')
    assert receipt['source_contract_sha256']==launch['protocol_sha256']==sha(RUN/'serving_contract.json')
    assert receipt['zero_study_fallbacks'] and receipt['CoAtNet_executions']==1 and not receipt['OFL_classifier_used']
    assert not receipt['official_evaluation_submitted'] and receipt['rows']==3
    assert receipt['visible_N02_parity_max_abs_error']<=1e-6
    assert sha(output/'submission.csv')==receipt['candidate_sha256']['C_N02_compact.csv']
    ids=[r['StudyInstanceUID'] for r in contract['visible_N02_reference']];labels=contract['target_order']
    for n,h in receipt['candidate_sha256'].items():
        p=attempt/n;assert sha(p)==h,n
        f=pd.read_csv(p,dtype={'StudyInstanceUID':str})
        assert list(f.columns)==['StudyInstanceUID',*labels] and f.StudyInstanceUID.is_unique and set(f.StudyInstanceUID)==set(ids)
        a=f[labels].to_numpy();assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    earlier=ROOT/'research/public_refresh_20261011/qualification_output/helios_qualification_attempt_tkewxzd0/resource_pilot/predictions.csv'
    old=pd.read_csv(earlier,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids,labels]
    current=pd.read_csv(attempt/'compact_raw.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[ids,labels]
    error=float(np.max(abs(old.to_numpy()-current.to_numpy())))
    assert error<=.001,'Serving compact differs from qualified compact reader'
    peak={}
    for sample in read(attempt/'resource_samples.json'):
        for line in sample['gpus'].splitlines():
            gpu,used,total,util=[int(x.strip()) for x in line.split(',')]
            peak[gpu]=max(peak.get(gpu,0),used);assert used<total
    baseline=ROOT/'public_kernels/fixed_reader_blend_v3/rsna-knee-n02-fixed-reader-blend-v3.py'
    assert sha(baseline)==contract['baseline_package_sha256']
    warm=re.search(r'READER_BATCH4_ALL6VIEWS_PASS peak_allocated_bytes=(\d+)',(attempt/'old_reader.log').read_text())
    assert warm is not None
    warm_peak=int(warm.group(1));assert 0<warm_peak<16*1024**3
    result={'status':'PASS_VISIBLE_COMBINED_INFERENCE_READY_FOR_REVIEW','observed_utc':datetime.now(timezone.utc).isoformat(),
            'kernel':launch['kernel'],'version':launch['version'],'rows':receipt['rows'],
            'elapsed_seconds':receipt['elapsed_seconds'],'stages':receipt['stages'],
            'baseline_unchanged':True,'N02_max_absolute_error':receipt['visible_N02_parity_max_abs_error'],
            'compact_vs_prior_qualification_max_absolute_error':error,'sampled_total_GPU_peak_MiB':peak,
            'reader_batch4_all6views_peak_allocated_bytes':warm_peak,
            'zero_study_fallbacks':True,'primary_candidate':str((output/'submission.csv').relative_to(ROOT)),
            'primary_candidate_sha256':sha(output/'submission.csv'),'notebook':str((ROOT/'public_kernels/phase5b_fixed_compact_blend/inference.ipynb').relative_to(ROOT)),
            'official_score':None,'author0_954_reproduced_on_our_account':False,'full_hidden_runtime_measured':False,
            'submission_approval_required':True,'competition_submission_made':False,
            'limitations':['Only the three visible test studies are available before an approved official evaluation.',
                           'Gold58 diagnostic is reused and cannot establish0.960 or0.970 performance.',
                           'Official rules permit public pretrained models; the submission export must stay within the nine-hour runtime limit.',
                           'Training exclusions and provenance are author statements rather than independently audited training logs.']}
    (RUN/'serving_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
