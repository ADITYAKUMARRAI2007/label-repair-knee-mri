"""Audit one frozen blend on reused Gold58; no weight search or submission."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/n03_paired_diagnostic'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(x):
    return pd.DataFrame(x).rank(method='average',pct=True).to_numpy()


def audit(output, bootstrap_draws=1000):
    protocol=json.loads((RUN/'protocol.json').read_text())
    receipt=json.loads((output/'diagnostic_receipt.json').read_text())
    assert receipt['status']=='PREDICTIONS_READY_NOT_SCORED'
    assert receipt['rows']==len(protocol['cohort'])==78
    assert not receipt['competition_submission_enabled']
    assert not (output/'submission.csv').exists(), 'Unexpected submission artifact'
    for name,expected in receipt['files'].items():
        assert sha(output/name)==expected, name
    columns=['StudyInstanceUID']+list(protocol['reader_weights'])
    # Dict order in the original recipe need not match official submission order.
    official=pd.read_csv(ROOT/'data/train.csv',dtype={'StudyInstanceUID':str})
    labels=[c for c in official.columns if c in protocol['reader_weights']]
    assert len(labels)==12
    def read(name):
        frame=pd.read_csv(output/name,dtype={'StudyInstanceUID':str})
        assert set(frame.columns)==set(columns)
        assert frame.StudyInstanceUID.tolist()==protocol['cohort'], name
        values=frame[labels].to_numpy(dtype=float)
        assert values.shape==(78,12) and np.isfinite(values).all()
        assert ((values>=0)&(values<=1)).all()
        return values
    anchor=read('fixed_reader_blend/anchor.csv')
    reader=read('fixed_reader_blend/reader.csv')
    crop=read('n03_diagnostic.csv')
    baseline=read('n02_diagnostic.csv')
    candidate=read('candidate_diagnostic.csv')
    raw=read('n03_raw.csv')
    w=np.asarray([protocol['reader_weights'][c] for c in labels])[None,:]
    mix=protocol['candidate_anchor_weight']
    # Published Nartaa uses ordinal argsort ranks on [0,1], including its tie
    # ordering. Replay that serving code exactly; the outer blend uses average ranks.
    published_crop=raw.astype(np.float32).argsort(0).argsort(0)/max(1,len(raw)-1)
    np.testing.assert_allclose(crop,published_crop.astype(np.float32),atol=1e-7,rtol=0)
    np.testing.assert_allclose(baseline,rank((1-w)*rank(anchor)+w*rank(reader)),atol=1e-7,rtol=0)
    spatial=rank((1-mix)*rank(anchor)+mix*rank(crop))
    np.testing.assert_allclose(candidate,rank((1-w)*rank(spatial)+w*rank(reader)),atol=1e-7,rtol=0)
    expected_sha=protocol['candidate_checkpoint_sha256']
    log=(output/'n03_diagnostic.log').read_text()
    assert ('N03_CHECKPOINT_SHA256 '+expected_sha) in log
    assert 'FALLBACK' not in log and ' FAILED ' not in log
    ids=protocol['diagnostic_ids'];positions=[protocol['cohort'].index(u) for u in ids]
    truth=official.set_index('StudyInstanceUID').loc[ids,labels].to_numpy(dtype=int)
    assert truth.shape==(58,12) and np.isin(truth,[0,1]).all()
    b=baseline[positions];c=candidate[positions]
    bs=roc_auc_score(truth,b,average=None);cs=roc_auc_score(truth,c,average=None)
    rng=np.random.default_rng(20261009);deltas=[];discarded=0
    for _ in range(bootstrap_draws):
        sample=rng.integers(0,len(ids),size=len(ids));y=truth[sample]
        if ((y.sum(0)==0)|(y.sum(0)==len(ids))).any():
            discarded+=1;continue
        deltas.append(float(roc_auc_score(y,c[sample])-roc_auc_score(y,b[sample])))
    assert len(deltas)>=bootstrap_draws//2, 'Too few valid paired bootstrap samples'
    report={'status':'DESCRIPTIVE_DIAGNOSTIC_COMPLETE','baseline_macro':float(bs.mean()),
            'candidate_macro':float(cs.mean()),'delta':float((cs-bs).mean()),
            'per_target':{label:{'baseline_auc':float(bs[i]),'candidate_auc':float(cs[i]),'delta':float(cs[i]-bs[i]),'positives':int(truth[:,i].sum()),'negatives':int(len(ids)-truth[:,i].sum())} for i,label in enumerate(labels)},
            'paired_bootstrap':{'seed':20261009,'attempted':bootstrap_draws,'valid':len(deltas),'discarded_single_class':discarded,'delta_percentile_95_interval':np.quantile(deltas,[.025,.975]).tolist()},
            'decision':'REJECT_THIS_FIXED_BLEND' if cs.mean()<=bs.mean() else 'POSITIVE_DIAGNOSTIC_REQUIRES_REVIEW',
            'limitations':'Gold58 was repeatedly reused and informed public models. This interval describes resampling these studies only; it does not correct model-selection bias or predict public/private leaderboard gain. Invalid single-class resamples are omitted.',
            'weight_search_performed':False,'competition_submission_enabled':False,
            'protocol_sha256':sha(RUN/'protocol.json'),'receipt_sha256':sha(output/'diagnostic_receipt.json'),
            'evaluation_ids':ids,'source_prediction_files':receipt['files']}
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.output)
    (RUN/'result_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'per_target','evaluation_ids','source_prediction_files'}},indent=2))


if __name__=='__main__':main()
