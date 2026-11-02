"""One fixed diagnostic using saved predictions; never select submission weights."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/mavit_complement_20261009'
WEIGHT = 0.10


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    baseline_path = ROOT/'runs/colab_n03/output/n02_diagnostic.csv'
    candidate_path = ROOT/'runs/mavit_gold_screen_v1/mavit_gold58_predictions.csv'
    receipt = json.loads((candidate_path.parent/'mavit_gold58_receipt.json').read_text())
    assert sha(candidate_path) == receipt['prediction_sha256']
    baseline_receipt = json.loads((baseline_path.parent/'diagnostic_receipt.json').read_text())
    assert sha(baseline_path) == baseline_receipt['files'][baseline_path.name]
    protocol = {
        'candidate_weight': WEIGHT,
        'purpose': 'Descriptive complementarity screen only; no promotion or weight search',
        'cohort': 'Exact 58 expert IDs from existing MaViT receipt',
        'ranking': 'Both saved arms reranked on identical 58 rows; not a 78-study serving replay',
        'bootstrap_seed': 20261010, 'bootstrap_draws': 1000,
        'baseline_sha256': sha(baseline_path), 'candidate_sha256': sha(candidate_path),
        'independent_validation': False, 'competition_submission_enabled': False,
    }
    frozen = OUT/'protocol.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2)+'\n')
    candidate = pd.read_csv(candidate_path, dtype={'StudyInstanceUID': str})
    ids = candidate.StudyInstanceUID.tolist()
    assert len(ids) == len(set(ids)) == 58
    labels = candidate.columns[1:].tolist()
    assert len(labels) == 12
    baseline = pd.read_csv(baseline_path, dtype={'StudyInstanceUID': str})
    assert baseline.StudyInstanceUID.is_unique
    baseline = baseline.set_index('StudyInstanceUID').loc[ids, labels]
    candidate = candidate.set_index('StudyInstanceUID').loc[ids, labels]
    truth = pd.read_csv(ROOT/'data/train.csv', dtype={'StudyInstanceUID': str})
    y = truth.set_index('StudyInstanceUID').loc[ids, labels].to_numpy()
    assert np.isin(y, [0, 1]).all()
    for frame in [baseline, candidate]:
        assert np.isfinite(frame.to_numpy()).all()
        assert ((frame.to_numpy() >= 0) & (frame.to_numpy() <= 1)).all()
    b = baseline.rank(pct=True).to_numpy()
    c = candidate.rank(pct=True).to_numpy()
    mixed = (1-WEIGHT)*b + WEIGHT*c
    auc_b = roc_auc_score(y, b, average=None)
    auc_c = roc_auc_score(y, c, average=None)
    auc_mix = roc_auc_score(y, mixed, average=None)
    assert abs(auc_c.mean()-receipt['mean_auc']) < 1e-12
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    deltas = []
    for _ in range(protocol['bootstrap_draws']):
        ix = rng.integers(len(ids), size=len(ids))
        if ((y[ix].sum(0) == 0) | (y[ix].sum(0) == len(ids))).any():
            continue
        deltas.append(float(roc_auc_score(y[ix], mixed[ix])-roc_auc_score(y[ix], b[ix])))
    assert len(deltas) >= 500
    report = {
        'status': 'DESCRIPTIVE_SCREEN_COMPLETE',
        'baseline_auc': float(auc_b.mean()), 'mavit_auc': float(auc_c.mean()),
        'fixed_mix_auc': float(auc_mix.mean()), 'delta': float((auc_mix-auc_b).mean()),
        'bootstrap_valid': len(deltas), 'delta_interval': np.quantile(deltas, [.025, .975]).tolist(),
        'per_target': {label: {
            'baseline_auc': float(auc_b[i]), 'mavit_auc': float(auc_c[i]),
            'mix_auc': float(auc_mix[i]), 'rank_correlation': float(np.corrcoef(b[:,i], c[:,i])[0,1]),
        } for i, label in enumerate(labels)},
        'weight_search_performed': False, 'competition_submission_enabled': False,
        'decision': 'HOLD_FOR_PROVENANCE_AND_INDEPENDENT_EVIDENCE',
        'limitations': 'Repeatedly reused Gold58; MaViT training exclusion unknown. Different source runtimes. Rank mixing uses 58 rows only. Neither the point estimate nor bootstrap interval predicts leaderboard gain.',
        'protocol_sha256': sha(frozen),
    }
    (OUT/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'per_target'}, indent=2))


if __name__ == '__main__':
    main()
