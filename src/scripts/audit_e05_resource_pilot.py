"""Audit completed epoch/log evidence; no accuracy or full-training approval."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(output):
    build=json.loads((ROOT/'runs/e05_colab_pilot/build.json').read_text())
    assert sha(output/'setup/worker.py')==build['code_sha256']
    result=json.loads((output/'result.json').read_text())
    assert result['status']=='PAIRED_RESOURCE_EPOCHS_COMPLETE'
    assert result['accuracy_measured'] is False and result['competition_submission_enabled'] is False
    assert len(result['reports'])==2
    evidence=[]
    for arm,parent in zip(['control','uniform'],result['reports']):
        receipt=json.loads((output/arm/'receipt.json').read_text())
        assert parent==receipt
        assert receipt['status']=='COMPLETE_RESOURCE_EPOCH_ONLY' and receipt['arm']==arm
        assert receipt['rows']==3525 and receipt['batches']==1763 and receipt['heldout_rows']==882
        assert receipt['accuracy_measured'] is False and receipt['checkpoint_reusable_for_training'] is False
        assert receipt['checkpoint_replay_max_abs']<=1e-6 and receipt['overflows']<=10
        assert 0<receipt['gpu_peak_gib']<35
        assert receipt['epoch_seconds']>0 and np.isfinite(receipt['epoch_seconds'])
        assert receipt['versions']['timm']=='1.0.29'
        heartbeats=[]
        for line in (output/f'{arm}.log').read_text().splitlines():
            try:item=json.loads(line)
            except json.JSONDecodeError:continue
            if item.get('arm')==arm and 'step' in item:heartbeats.append(item)
        assert [p['step'] for p in heartbeats]==list(range(100,1763,100))+[1763]
        assert all(p['total_steps']==1763 and p['rss_gib']<30 for p in heartbeats)
        assert all(b['elapsed_seconds']>a['elapsed_seconds'] for a,b in zip(heartbeats,heartbeats[1:]))
        assert heartbeats[-1]['elapsed_seconds']<=receipt['epoch_seconds']
        evidence.append({'arm':arm,'epoch_seconds':receipt['epoch_seconds'],
                         'rss_peak_gib':max(p['rss_gib'] for p in heartbeats),
                         'gpu_peak_gib':receipt['gpu_peak_gib'],
                         'receipt_sha256':sha(output/arm/'receipt.json')})
    assert result['reports'][0]['training_order_sha256']==result['reports'][1]['training_order_sha256']
    worst=max(x['epoch_seconds'] for x in evidence)
    hours=worst*140/3600
    return {'status':'PASS_PAIRED_RESOURCE_EPOCHS','arms':evidence,
            'ten_models_14_epochs_training_hours':hours,
            'training_hours_with_20percent_margin':hours*1.2,
            'extra_unmeasured_costs':['Final validation','Full optimizer checkpoint persistence','Startup and transfers'],
            'accuracy_measured':False,'full_training_authorized_by_this_audit':False,
            'next_gate':'Compare measured cost plus persistence/validation with fresh available compute; do not assume enough units.',
            'limitations':'One initial epoch per arm on one runtime. This is resource evidence, not convergence or expected leaderboard score.',
            'result_sha256':sha(output/'result.json')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=audit(args.output)
    (ROOT/'runs/e05_colab_pilot/resource_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
