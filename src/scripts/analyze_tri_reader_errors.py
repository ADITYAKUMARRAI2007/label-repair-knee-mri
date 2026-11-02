"""Index repaired and regressed AUC pairs; do not fit or select blend weights."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/tri_reader_20261010'


def main():
    result = json.loads((OUT / 'result.json').read_text())
    candidate = OUT / 'diagnostic_predictions.csv'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == result['prediction_sha256']
    baseline = pd.read_csv(ROOT / 'runs/colab_n03/output/n02_diagnostic.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    linked = pd.read_csv(candidate, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    truth = pd.read_csv(ROOT / 'data/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    ids = json.loads((ROOT / 'runs/n03_paired_diagnostic/protocol.json').read_text())['diagnostic_ids']
    series = pd.read_csv(ROOT / 'data/train_series.csv', dtype={'StudyInstanceUID': str})
    targets, cases = {}, []
    for label in result['per_target']:
        pos = [u for u in ids if truth.loc[u, label] == 1]
        neg = [u for u in ids if truth.loc[u, label] == 0]

        def credit(p, n, frame):
            a, b = frame.loc[p, label], frame.loc[n, label]
            return float(a > b) + .5*float(a == b)

        repaired = regressed = 0.
        study_repaired = {u: 0. for u in ids}
        study_regressed = {u: 0. for u in ids}
        for p in pos:
            for n in neg:
                delta = credit(p, n, linked) - credit(p, n, baseline)
                if delta > 0:
                    repaired += delta
                    study_repaired[p] += delta
                    study_repaired[n] += delta
                elif delta < 0:
                    regressed -= delta
                    study_regressed[p] -= delta
                    study_regressed[n] -= delta
        auc_delta = (repaired-regressed)/(len(pos)*len(neg))
        assert abs(auc_delta-result['per_target'][label]['delta']) < 1e-12
        targets[label] = {'positives': len(pos), 'negatives': len(neg),
                          'positive_negative_pairs': len(pos)*len(neg),
                          'repaired_pair_credit': repaired, 'regressed_pair_credit': regressed,
                          'net_pair_credit': repaired-regressed, 'auc_delta': auc_delta,
                          'macro_contribution': auc_delta/12}
        for uid in sorted(ids, key=lambda u: study_regressed[u]-study_repaired[u], reverse=True)[:5]:
            meta = series[series.StudyInstanceUID == uid]
            cases.append({'StudyInstanceUID': uid, 'target': label, 'label': int(truth.loc[uid, label]),
                          'baseline_score': float(baseline.loc[uid, label]),
                          'tri_reader_score': float(linked.loc[uid, label]),
                          'repaired_pair_credit': study_repaired[uid],
                          'regressed_pair_credit': study_regressed[uid],
                          'available_series': len(meta),
                          'available_planes': ','.join(sorted(set(meta.Anatomical_Plane)))})
    assert abs(sum(x['macro_contribution'] for x in targets.values()) - result['delta']) < 1e-12
    pd.DataFrame(cases).to_csv(OUT / 'failure_case_index.csv', index=False)
    report = {'status': 'PAIRED_RANK_ERROR_ANALYSIS_COMPLETE',
              'selection': 'Per-target five studies with greatest regressed minus repaired pair credit; diagnostic only',
              'targets': targets, 'medical_cause_inferred': False, 'weight_search_performed': False,
              'limitations': 'Reused expert cohort. Ranking error analysis identifies cases for review, not an anatomical diagnosis or independent validation.'}
    (OUT / 'rank_error_analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in targets.items() if k in ['Lateral OA', 'Synovitis', 'Lateral Meniscus']}, indent=2))


if __name__ == '__main__':
    main()
