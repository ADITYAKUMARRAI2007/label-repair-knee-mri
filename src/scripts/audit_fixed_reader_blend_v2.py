"""Independently audit the executed components and published fixed blend."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'inference'))
from nartaa_reproduction_guard import sha,validate_submission,reported_failure


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    local=ROOT/'runs/fixed_reader_blend_v2';build=json.loads((local/'build.json').read_text())
    for name,expected in {**build['sources'],**build['gates']}.items():assert sha(ROOT/name)==expected,name
    out=args.directory;folder=out/'fixed_reader_blend';root=out/'fixed_blend_source'
    assert not (folder/'failure.json').exists()
    receipt=json.loads((folder/'receipt.json').read_text());contract=build['contract']
    assert receipt['status']=='PASS_EXECUTION_ONLY' and receipt['contract']==contract
    assert not receipt['accuracy_measured'] and not receipt['competition_submission_enabled']
    for name,expected in contract['module_sha256'].items():assert sha(root/name)==expected,name
    anchor=receipt['anchor_receipt'];expected_n01=json.loads((ROOT/'runs/nartaa_reproduction_v1/contract.json').read_text())
    assert anchor['status']=='PASS_EXECUTION_ONLY' and anchor['contract']==expected_n01
    assert anchor['checkpoint_sha256']==expected_n01['checkpoint_sha256']
    assert anchor['source_sha256']==expected_n01['source_sha256']
    assert receipt['anchor']['reported_failure_count']==receipt['reader']['reported_failure_count']==0
    assert receipt['decoder_check']['reported_failure_count']==0
    assert 'DECODER_ABI_PASS cp313' in (folder/'decoder.log').read_text()
    assert (folder/'anchor.csv').read_bytes()==(ROOT/'runs/nartaa_reproduction_v1/outputs/submission.csv').read_bytes()
    assert receipt['elapsed_seconds']<contract['visible_seconds_cap']
    for name in ['anchor','reader']:
        log=(folder/(name+'.log')).read_text();assert not any(reported_failure(line) for line in log.splitlines())
        assert sha(folder/(name+'.csv'))==receipt['components'][name+'.csv']
        validate_submission(folder/(name+'.csv'),ROOT/'data/test.csv',ROOT/'data/sample_submission.csv')
    log=(folder/'reader.log').read_text()
    peak=int(re.search(r'READER_BATCH4_ALL6VIEWS_PASS peak_allocated_bytes=(\d+)',log)[1])
    assert peak<14*1024**3,'batch4 qualification leaves insufficient memory margin'
    def read(p):
        with p.open() as f:r=csv.DictReader(f);return r.fieldnames,list(r)
    fields,aa=read(folder/'anchor.csv');_,bb=read(folder/'reader.csv');_,cc=read(out/'submission.csv')
    assert f'READER_COMPLETE rows={len(cc)}' in log
    maxerr=0
    for label in fields[1:]:
        w=contract['weights'][label];n=len(aa)
        a=rankdata([float(r[label]) for r in aa],method='average')/n
        b=rankdata([float(r[label]) for r in bb],method='average')/n
        expected=rankdata((1-w)*a+w*b,method='average')/n
        actual=np.array([float(r[label]) for r in cc]);assert np.array_equal(actual,expected)
        maxerr=max(maxerr,float(np.max(np.abs(actual-expected))))
    submission=validate_submission(out/'submission.csv',ROOT/'data/test.csv',ROOT/'data/sample_submission.csv')
    assert submission==receipt['output']
    report=dict(status='PASS_VISIBLE_EXECUTION_ONLY',submission=submission,elapsed_seconds=receipt['elapsed_seconds'],
       reader_batch4_all6views_peak_bytes=peak,blend_max_abs_error=maxerr,accuracy_measured=False,
       author_reported_score='0.950',our_score=None,receipt_sha256=sha(folder/'receipt.json'),
       auditor_sha256=sha(Path(__file__)),limitations=contract['limitations'])
    (local/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
