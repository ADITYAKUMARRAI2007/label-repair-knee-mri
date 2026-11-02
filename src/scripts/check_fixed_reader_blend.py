"""Check blend arithmetic, row alignment and failure rejection without model inference."""
import csv
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inference'))
from fixed_reader_blend import blend, checked_view
from nartaa_reproduction_guard import run_source, sha


def main():
    identity = json.loads((ROOT / 'research/new_public/after_n01_20261008/fixed_v3_components/identity.json').read_text())
    weights = identity['fixed_convnext_weights']
    fields = ['StudyInstanceUID'] + list(weights)
    values_a = [0.1,0.1,0.5,0.9,0.2,0.3,0.8,0.8,0.7]
    values_b = [0.9,0.5,0.5,0.1,0.2,0.4,0.7,0.8,0.3]
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory)
        def write(name, values, reverse=False):
            rows = [dict(StudyInstanceUID=str(i), **{label:value for label in weights})
                    for i,value in enumerate(values)]
            if reverse: rows.reverse()
            with (p/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        write('a.csv',values_a);write('b.csv',values_b);write('test.csv',values_a)
        result=blend(p/'a.csv',p/'b.csv',p/'out.csv',weights,p/'test.csv',p/'a.csv')
        with (p/'out.csv').open() as f: rows=list(csv.DictReader(f))
        for label,w in weights.items():
            ranks_a, ranks_b = rankdata(values_a)/9, rankdata(values_b)/9
            expected=rankdata((1-w)*ranks_a+w*ranks_b)/9
            assert np.array_equal(expected,np.array([float(r[label]) for r in rows]))
        for values,reverse in [(values_b,True),([float('nan')]+values_b[1:],False),([1.1]+values_b[1:],False)]:
            write('bad.csv',values,reverse)
            try:blend(p/'a.csv',p/'bad.csv',p/'badout.csv',weights,p/'test.csv',p/'a.csv')
            except AssertionError:pass
            else:raise AssertionError('invalid component was accepted')
        try:checked_view((None,np.zeros(6,dtype=bool),None))
        except AssertionError:pass
        else:raise AssertionError('empty study was accepted')
        checked_view((None,np.array([True,False]),None))
        (p/'failed_reader.py').write_text("print('[pre] FAILED reader selected series: AssertionError')\n")
        try:run_source(p/'failed_reader.py',p/'failure.log',10,p)
        except AssertionError:pass
        else:raise AssertionError('silently skipped acquisition was accepted')
    out=ROOT/'runs/fixed_reader_blend_v1';out.mkdir(exist_ok=True)
    report=dict(status='PASS_BLEND_AND_FAILURE_GUARDS', runtime_sha256=sha(ROOT/'inference/fixed_reader_blend.py'),
                checker_sha256=sha(Path(__file__)), checks=['independent_scipy_rank_replay_with_ties',
                'reject_reordered_UIDs','reject_NaN','reject_out_of_range','reject_no_valid_views',
                'reject_reported_decode_failure_after_zero_exit'],model_inference_performed=False)
    (out/'guard_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
