"""Associate metadata acquisition gaps with saved ranking errors, without fitting."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/window_integrity_20261010'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = RUN / 'anchor_published.py'
    prior = json.loads((RUN / 'ablation_protocol.json').read_text())
    assert sha(source) == prior['control_source_sha256']
    baseline_path = ROOT / 'runs/colab_n03/output/n02_diagnostic.csv'
    candidate_path = ROOT / 'runs/tri_reader_20261010/diagnostic_predictions.csv'
    metadata_path = ROOT / 'data/train_series.csv'
    labels_path = ROOT / 'data/train.csv'
    result = json.loads((ROOT / 'runs/tri_reader_20261010/result.json').read_text())
    assert sha(candidate_path) == result['prediction_sha256']
    blend_protocol = json.loads((ROOT / 'runs/e05_confidence_ablation/complete_blend_protocol.json').read_text())
    assert sha(baseline_path) == blend_protocol['baseline_sha256']
    protocol = {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'comparison': 'Saved N02 best recipe and exact released N04 recipe, same58 experts',
        'groups': ['both_metadata_complete', 'one_metadata_incomplete', 'both_metadata_incomplete'],
        'group_rule': 'Use original acquisition picker; incomplete means any unfilled slot',
        'metric': 'Wrong positive-negative ranking credit: inversion1, tie0.5, correct0',
        'targets': list(result['per_target']),
        'known_prior_results': 'Overall recipe scores, target rank errors and metadata gap counts are known',
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in
                    [source, baseline_path, candidate_path, metadata_path, labels_path]},
        'parameter_search': False, 'causal_effect_estimated': False,
        'gpu_launched': False, 'competition_submission_enabled': False,
    }
    protocol_path = RUN / 'coverage_error_protocol.json'
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text())
        assert saved['sources'] == protocol['sources'] and saved['groups'] == protocol['groups']
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    tree = ast.parse(source.read_text())
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == '_pick_series_for_slot':
            definitions.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'SLOTS' for t in node.targets):
            definitions.append(node)
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
    picker, slots = namespace['_pick_series_for_slot'], namespace['SLOTS']
    frozen = json.loads((ROOT / 'runs/E04_contract/e04_contract.json').read_text())
    ids = sorted(r['StudyInstanceUID'] for r in frozen['rows'] if r['expert'])
    assert len(ids) == 58
    meta = pd.read_csv(metadata_path, dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    old_audit = json.loads((RUN / 'metadata_window_audit.json').read_text())
    expected = {r['StudyInstanceUID']: r['metadata_slots_present'] for r in old_audit['studies']}
    assert set(expected) == set(ids)
    present, selected = {}, {}
    for uid in ids:
        rows = meta[meta.StudyInstanceUID == uid].to_dict('records')
        used, presence, series = set(), [], []
        for plane, fluid, _ in slots:
            record = picker(rows, plane, fluid, used)
            presence.append(record is not None)
            series.append(None if record is None else record['SeriesInstanceUID'])
            if record is not None:
                used.add(record['SeriesInstanceUID'])
        assert presence == expected[uid]
        present[uid], selected[uid] = presence, series
    missing = np.array([not all(present[u]) for u in ids])
    targets = protocol['targets']
    y = pd.read_csv(labels_path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, targets].to_numpy()
    assert np.isin(y, [0, 1]).all()
    predictions = {}
    for name, path in [('best', baseline_path), ('released_tri_reader', candidate_path)]:
        frame = pd.read_csv(path, dtype={'StudyInstanceUID': str})
        assert frame.StudyInstanceUID.is_unique
        predictions[name] = frame.set_index('StudyInstanceUID').loc[ids, targets].to_numpy()
    assert abs(roc_auc_score(y, predictions['released_tri_reader']) - result['linked_macro_auc']) < 1e-12
    assert abs(roc_auc_score(y, predictions['best']) - result['baseline_macro_auc']) < 1e-12
    targets_report, study_rows = {}, []
    for j, target in enumerate(targets):
        pos, neg = np.flatnonzero(y[:, j] == 1), np.flatnonzero(y[:, j] == 0)
        flags = missing[pos, None].astype(int) + missing[neg][None, :].astype(int)
        wrong = {}
        for name, pred in predictions.items():
            difference = pred[pos, j, None] - pred[neg, j][None, :]
            wrong[name] = (difference < 0).astype(float) + .5 * (difference == 0)
        strata = {}
        for count, group in enumerate(protocol['groups']):
            keep = flags == count
            pairs = int(keep.sum())
            entry = {'positive_negative_pairs': pairs}
            for name in predictions:
                error = float(wrong[name][keep].sum())
                entry[name] = {'wrong_pair_credit': error,
                               'concordance': 1 - error / pairs if pairs else None}
            strata[group] = entry
        assert sum(e['positive_negative_pairs'] for e in strata.values()) == len(pos) * len(neg)
        for name, pred in predictions.items():
            reconstructed = 1 - sum(e[name]['wrong_pair_credit'] for e in strata.values()) / (len(pos) * len(neg))
            assert abs(reconstructed - roc_auc_score(y[:, j], pred[:, j])) < 1e-12
            endpoint_errors = np.zeros(len(ids))
            endpoint_errors[pos] = wrong[name].sum(axis=1) / len(neg)
            endpoint_errors[neg] = wrong[name].sum(axis=0) / len(pos)
            for i, uid in enumerate(ids):
                study_rows.append({'StudyInstanceUID': uid, 'target': target, 'model': name,
                                   'label': int(y[i, j]), 'metadata_incomplete': bool(missing[i]),
                                   'missing_slot_indices': ','.join(str(s) for s, ok in enumerate(present[uid]) if not ok),
                                   'opposite_label_ranking_error_fraction': float(endpoint_errors[i])})
        targets_report[target] = {'positives': len(pos), 'negatives': len(neg), 'strata': strata}
    summary = {}
    for group in protocol['groups']:
        valid = [t for t in targets if targets_report[t]['strata'][group]['positive_negative_pairs']]
        summary[group] = {
            'supported_targets': len(valid),
            'mean_target_concordance': {name: float(np.mean([
                targets_report[t]['strata'][group][name]['concordance'] for t in valid]))
                for name in predictions},
        }
    output = {
        'status': 'METADATA_GAP_RANKING_ERROR_ASSOCIATION_ONLY',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'studies': 58, 'metadata_complete': int((~missing).sum()), 'metadata_incomplete': int(missing.sum()),
        'pair_strata_summary': summary, 'per_target': targets_report,
        'protocol_sha256': sha(protocol_path), 'source_sha256': sha(Path(__file__)),
        'causal_failure_claim': False, 'accuracy_gain_from_window_filter_measured': False,
        'competition_submission_enabled': False,
        'limitations': [
            'Metadata presence is replayed exactly; actual decoded masks are not measured here.',
            'Case mix, disease prevalence, protocol and acquisition selection can confound group comparisons.',
            'All complete studies also contain cross-acquisition triplets in the published input.',
            'Pair-stratum concordance is descriptive and differs from full-cohort AUC.',
            'Public model development and prior diagnostics reused these58 expert labels.',
            'This analysis must not be used to tune per-target or per-case weights.',
        ],
    }
    pd.DataFrame(study_rows).to_csv(RUN / 'coverage_error_cases.csv', index=False)
    (RUN / 'coverage_error_result.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k not in ['per_target', 'limitations']}))


if __name__ == '__main__':
    main()
