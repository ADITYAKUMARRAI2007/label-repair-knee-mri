"""One explicit interpretation of the diagram's unreleased probit fusion."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/tri_reader_probit_20261010'
BASE = ROOT / 'runs/colab_n03/output'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    old = json.loads((ROOT / 'runs/tri_reader_20261010/protocol.json').read_text())
    paths = {'anchor': BASE / 'fixed_reader_blend/anchor.csv',
             'reader': BASE / 'fixed_reader_blend/reader.csv', 'crop': BASE / 'n03_diagnostic.csv',
             'baseline': BASE / 'n02_diagnostic.csv',
             'published_linear': ROOT / 'runs/tri_reader_20261010/diagnostic_predictions.csv'}
    protocol = {'schema': 'fixed_tri_reader_probit_interpretation_v1',
                'transform': 'ndtri((average_rank-0.5)/cohort_size);finite midpoint quantiles',
                'fusion': '70%normal(anchor)+30%normal(crop);rerank;normal transform;published per-target reader mix;final average percentile rank',
                'weights': old['recipe'], 'source_sha256': {k: sha(p) for k, p in paths.items()},
                'draws': 2000, 'seed': 20261010, 'parameter_search': False,
                'private_author_method_reproduced': False, 'competition_submission_enabled': False,
                'context': 'Public linear blend results already known. Single transform interpretation frozen before its accuracy.',
                'limitations': ['Reused58-expert diagnostic;no independent validation or leaderboard forecast',
                               'Diagram does not specify the private probit formula;this is our implementation',
                               'Ranks use78 saved studies;hidden-test cohort may differ']}
    pp = OUT / 'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol
    else:
        pp.write_text(json.dumps(protocol, indent=2) + '\n')
    labels = list(old['recipe']['reader_weights'])
    cohort = json.loads((ROOT / 'runs/n03_paired_diagnostic/protocol.json').read_text())
    frames = {k: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[cohort['cohort'], labels]
              for k, p in paths.items()}
    assert all(f.index.is_unique and np.isfinite(f.to_numpy()).all() for f in frames.values())

    def normal_rank(frame):
        values = ndtri((frame.rank(method='average').to_numpy()-.5)/len(frame))
        assert np.isfinite(values).all()
        return pd.DataFrame(values, index=frame.index, columns=frame.columns)

    mix = old['recipe']['crop_weight']
    coat = (1-mix)*normal_rank(frames['anchor'])+mix*normal_rank(frames['crop'])
    weights = pd.Series(old['recipe']['reader_weights'])
    candidate = ((1-weights)*normal_rank(coat)+weights*normal_rank(frames['reader'])).rank(method='average', pct=True)
    candidate.to_csv(OUT / 'diagnostic_predictions.csv')
    ids = cohort['diagnostic_ids']
    truth = pd.read_csv(ROOT / 'data/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, labels].to_numpy(dtype=int)
    assert np.isin(truth, [0, 1]).all()
    arrays = {k: frames[k].loc[ids].to_numpy() for k in ['baseline', 'published_linear']}
    arrays['probit'] = candidate.loc[ids].to_numpy()
    metrics = {k: roc_auc_score(truth, a, average=None) for k, a in arrays.items()}
    group_contract = json.loads((ROOT / 'runs/E04_contract/e04_contract.json').read_text())
    groups = {r['StudyInstanceUID']: r['report_group'] for r in group_contract['rows'] if r['expert']}
    clusters = [[i for i, uid in enumerate(ids) if groups[uid] == g] for g in sorted({groups[u] for u in ids})]
    spec = importlib.util.spec_from_file_location('auc_helpers', ROOT / 'scripts/bootstrap_gold58.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    ordered = {k: [helper.grouped_scores(a[:, t].tolist()) for t in range(12)] for k, a in arrays.items()}
    deltas = {k: [] for k in ['baseline', 'published_linear']}
    rng = np.random.default_rng(protocol['seed'])
    for _ in range(protocol['draws']):
        counts = [0]*len(ids)
        for group_index in rng.integers(len(clusters), size=len(clusters)):
            for row_index in clusters[group_index]:
                counts[row_index] += 1
        scores = {k: [helper.auc(truth[:, t].tolist(), ordered[k][t], counts) for t in range(12)] for k in arrays}
        if any(v is None for v in scores['probit']):
            continue
        for k in deltas:
            deltas[k].append(float(np.mean(scores['probit'])-np.mean(scores[k])))
    assert all(len(a) >= 1000 for a in deltas.values())
    result = {'status': 'FIXED_PROBIT_DIAGNOSTIC_COMPLETE',
              'macro_auc': {k: float(m.mean()) for k, m in metrics.items()},
              'comparisons': {k: {'delta': float((metrics['probit']-metrics[k]).mean()),
                                 'paired_report_group_95_interval': np.quantile(deltas[k], [.025, .975]).tolist(),
                                 'valid_draws': len(deltas[k])} for k in deltas},
              'per_target': {t: {k: float(m[i]) for k, m in metrics.items()} for i, t in enumerate(labels)},
              'protocol_sha256': sha(pp), 'prediction_sha256': sha(OUT / 'diagnostic_predictions.csv'),
              'decision': 'HOLD_NO_SUBMISSION', 'private_author_method_reproduced': False,
              'gpu_launched': False, 'competition_submission_enabled': False,
              'limitations': protocol['limitations']}
    if metrics['probit'].mean() <= metrics['published_linear'].mean():
        result['decision'] = 'REJECT_THIS_PROBIT_INTERPRETATION'
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_target'}, indent=2))


if __name__ == '__main__':
    main()
