#!/usr/bin/env python3
"""Freeze a narrow label comparison; never evaluate official disease labels."""
import csv
import hashlib
import importlib.util
import json
import statistics
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/grade_supervision_trial_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open(newline='') as handle:
        records=list(csv.DictReader(handle))
    mapped={r['StudyInstanceUID']:r for r in records}
    if len(mapped)!=len(records):raise RuntimeError('duplicate study identity')
    return mapped


def main():
    spec=importlib.util.spec_from_file_location('policy',ROOT/'training/grade_aware_supervision.py')
    policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    review=json.loads((ROOT/'runs/supervision_review_v1/split.json').read_text())
    frozen=json.loads((ROOT/'runs/E04_contract/e04_contract.json').read_text())
    for relative,expected in review['input_sha256'].items():
        if sha(ROOT/relative)!=expected:raise RuntimeError('review inputs changed')
    excluded=set(review['excluded_expert_and_connected_ids'])
    meta={r['StudyInstanceUID']:r for r in frozen['rows']}
    official=rows(ROOT/'data/train.csv')
    targets=frozen['target_order']
    uids=sorted(set(meta)-excluded)
    sources=[rows(ROOT/r['path']) for r in frozen['fusion']['source_receipts'].values()]
    changed,states=Counter(),Counter();controls=[];candidates=[];weights=[];evidence_rows=[]
    for uid in uids:
        control={'StudyInstanceUID':uid};candidate={'StudyInstanceUID':uid};weight={'StudyInstanceUID':uid}
        for target in targets:
            values=[float(source[uid][target]) for source in sources]
            base=min(.99,max(.01,statistics.median(values)))
            control[target]=base
            candidate[target],finding=policy.adjusted_probability(official[uid]['Report'],target,base)
            weight[target]=min(1.,max(.25,1-(max(values)-min(values))))
            if target in policy.NOUNS:states[target+'::'+finding['state']]+=1
            if candidate[target]!=base:
                changed[target]+=1
                evidence_rows.append({'StudyInstanceUID':uid,'target':target,'control':base,
                                      'candidate':candidate[target],'evidence':finding,
                                      'report_review_holdout':uid in review['holdout_ids']})
            if target not in policy.NOUNS and candidate[target]!=base:
                raise RuntimeError('unrelated target changed')
        controls.append(control);candidates.append(candidate);weights.append(weight)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'contract.json').exists():raise RuntimeError('trial already frozen; do not overwrite it')
    for name,records in [('control_targets.csv',controls),('candidate_targets.csv',candidates),('shared_weights.csv',weights)]:
        with (OUT/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['StudyInstanceUID',*targets]);writer.writeheader();writer.writerows(records)
    (OUT/'changed_evidence.json').write_text(json.dumps(evidence_rows,indent=2,ensure_ascii=False)+'\n')
    policy_sha=sha(ROOT/'training/grade_aware_supervision.py')
    result={'schema':'grade_supervision_trial_v1','status':'FROZEN_NOT_TRAINED',
            'observed_utc':datetime.now(timezone.utc).isoformat(),'policy':policy.POLICY_VERSION,
            'target_order':targets,'training_studies':len(uids),'excluded_expert_and_connected':len(excluded),
            'ceiling':policy.LOW_PROBABILITY_CEILING,'changed_cells':dict(changed),'state_counts':dict(states),
            'variable':'only two report-size-qualified target probabilities; shared weights unchanged',
            'control':'exact prior four-file median recipe, not four independent label families',
            'source_family_correlation_correction_in_this_trial':False,
            'expert_finding_values_used':False,'expert_score':None,'public_score':None,
            'host_criteria_url':'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343',
            'policy_source_sha256':policy_sha,'review_split_sha256':sha(ROOT/'runs/supervision_review_v1/split.json'),
            'input_sha256':review['input_sha256'],
            'artifacts':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
            'limitations':['English phrase coverage only; abstentions unchanged',
                           'report size is not adjudicated MRI image truth',
                           'control sources include correlated and previously Gold-informed labeling',
                           '0.15 is a fixed hypothesis, not a calibrated probability',
                           'Gold58 is reused and is only a coarse rejection check'],
            'remaining_gates':['untouched report sample extraction review','feature extraction feasibility within1.5h',
                               'fixed paired image-head comparison','raw-DICOM parity','official standalone evaluation'],
            'gpu_jobs_launched':0,'submissions_launched':0}
    (OUT/'contract.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','training_studies','changed_cells','state_counts','expert_score','public_score']},indent=2))


if __name__=='__main__':main()
