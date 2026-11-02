"""Frozen ViT/CoAtNet/ResNet-50 diagnostic using existing audited predictions."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/three_family_ensemble_20261010'
sys.path.insert(0, str(ROOT / 'inference'))
from three_family_ensemble import rank_ensemble


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    base = ROOT / 'runs/colab_n03/output'
    vit_dir = ROOT / 'runs/public0033_probe_outputs/public0033_probe'
    rad_dir = ROOT / 'research/B03_validation_audit/rsna-knee-e13-train/output'
    paths = {'vit': vit_dir / 'expert_predictions.csv',
             'coatnet': base / 'fixed_reader_blend/anchor.csv',
             'resnet50': rad_dir / 'rsna_rad_e11/v52_e11_oof.csv',
             'baseline': base / 'n02_diagnostic.csv'}
    vit_receipt = json.loads((vit_dir / 'manifest.json').read_text())
    assert vit_receipt['status'] == 'COMPLETE' and vit_receipt['fallback'] == 0
    assert vit_receipt['reference_parity']['status'] == 'PASS'
    assert sha(paths['vit']) == vit_receipt['artifacts']['expert_predictions.csv']['sha256']
    rad_receipt = json.loads((rad_dir / 'rad_e11_audit.json').read_text())
    assert rad_receipt['encoder'] == 'RadImageNet ResNet-50 official PyTorch release'
    assert sha(paths['resnet50']) == rad_receipt['oof_sha256']
    base_receipt = json.loads((base / 'diagnostic_receipt.json').read_text())
    assert sha(paths['baseline']) == base_receipt['files']['n02_diagnostic.csv']
    anchor_receipt = json.loads((base / 'fixed_reader_blend/receipt.json').read_text())
    assert sha(paths['coatnet']) == anchor_receipt['components']['anchor.csv']
    old_protocol = json.loads((ROOT / 'runs/n03_paired_diagnostic/protocol.json').read_text())
    ids = sorted(old_protocol['diagnostic_ids'])
    labels = vit_receipt['contract']['targets']
    linked_ids = vit_receipt['contract']['report_connected_expert_ids']
    assert len(ids) == len(set(ids)) == 58 and len(labels) == 12
    protocol = {
        'schema': 'three_family_fixed_rank_ensemble_v1',
        'component_weights': {'vit': 1/3, 'coatnet': 1/3, 'resnet50': 1/3},
        'anchored_weights': {'baseline': .90, 'ensemble': .10},
        'components': {'vit': 'Executed public0033 DINOv2 Base study classifier',
                       'coatnet': 'Nartaa384-trained CoAtNet;320 crop;anatomical mirror',
                       'resnet50': 'RadImageNet ResNet-50 frozen encoder plus E13 held-out fold heads'},
        'component_sha256': {name: sha(path) for name, path in paths.items()},
        'studies': ids, 'targets': labels, 'ranking': 'Average ties; identical58-study cohort',
        'bootstrap': 'Paired report-group resampling;2000 draws;seed20261010',
        'sensitivity': {'exclude_known_vit_report_connections': linked_ids,
                        'keep_ranks_from_full_cohort': True},
        'model_or_weight_search': False, 'gpu_launched': False,
        'competition_submission_enabled': False,
        'accuracy_selection_context': 'Individual component results were previously known. This fixed combination has not been scored before freezing this protocol.',
        'limitations': ['Gold58 reused in public and local model development; not fresh independent validation',
                       'ResNet probabilities are held-out fold predictions; other arms use full released models',
                       'Public0033 has one known expert/report connection; sensitivity excludes it',
                       'Source patient/site separation and all training memberships are not established',
                       'Ranks on58 studies can differ from full hidden-test fusion; no leaderboard prediction'],
    }
    pp = OUT / 'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol, 'Frozen protocol changed'
    else:
        pp.write_text(json.dumps(protocol, indent=2) + '\n')
    frames = {}
    for name, path in paths.items():
        f = pd.read_csv(path, dtype={'StudyInstanceUID': str})
        assert f.StudyInstanceUID.is_unique and set(ids) <= set(f.StudyInstanceUID)
        frames[name] = f.set_index('StudyInstanceUID').loc[ids, labels]
    rad_all = pd.read_csv(paths['resnet50'], dtype={'StudyInstanceUID': str})
    assert set(rad_all.loc[rad_all.is_gold == 1, 'StudyInstanceUID']) == set(ids)
    ensemble = rank_ensemble({name: frames[name] for name in protocol['component_weights']},
                             protocol['component_weights'], ids, labels)
    anchored = rank_ensemble({'baseline': frames['baseline'], 'ensemble': ensemble},
                            protocol['anchored_weights'], ids, labels)
    # Identity and alignment checks address mixing predictions from different readers.
    shuffled = rank_ensemble({name: frames[name].iloc[::-1, ::-1] for name in protocol['component_weights']},
                             protocol['component_weights'], ids, labels)
    np.testing.assert_array_equal(ensemble.to_numpy(), shuffled.to_numpy())
    for invalid in [frames['vit'].iloc[1:], pd.concat([frames['vit'], frames['vit'].iloc[:1]])]:
        try:
            rank_ensemble({'vit': invalid}, {'vit': 1.}, ids, labels)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing or duplicate study identity was accepted')
    ensemble.to_csv(OUT / 'ensemble_diagnostic.csv')
    anchored.to_csv(OUT / 'anchored_diagnostic.csv')
    official = pd.read_csv(ROOT / 'data/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    y = official.loc[ids, labels].to_numpy(dtype=int)
    assert np.isin(y, [0, 1]).all()
    arrays = {name: frame.to_numpy(dtype=float) for name, frame in frames.items()}
    arrays.update(ensemble=ensemble.to_numpy(), anchored=anchored.to_numpy())
    metrics = {name: roc_auc_score(y, value, average=None) for name, value in arrays.items()}
    np.testing.assert_allclose(metrics['baseline'].mean(), .9256624221336741, atol=1e-12)
    np.testing.assert_allclose(metrics['resnet50'].mean(), rad_receipt['gold_oof_auc'], atol=1e-12)
    # Reuse the independent tie-aware AUC calculation for a fast paired bootstrap.
    spec = importlib.util.spec_from_file_location('auc_helpers', ROOT / 'scripts/bootstrap_gold58.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    group_contract = json.loads((ROOT / 'runs/E04_contract/e04_contract.json').read_text())
    groups = {row['StudyInstanceUID']: row['report_group'] for row in group_contract['rows'] if row['expert']}
    clusters = [[i for i, uid in enumerate(ids) if groups[uid] == group] for group in sorted(set(groups[u] for u in ids))]
    candidates = ['ensemble', 'anchored']
    score_groups = {name: [helper.grouped_scores(arrays[name][:, t].tolist()) for t in range(12)]
                    for name in ['baseline', *candidates]}
    for name, target_groups in score_groups.items():
        aucs = [helper.auc(y[:, t].tolist(), target_groups[t], [1]*58) for t in range(12)]
        np.testing.assert_allclose(aucs, metrics[name], atol=1e-12)
    rng = np.random.default_rng(20261010)
    deltas = {name: [] for name in candidates}
    for _ in range(2000):
        counts = [0]*58
        for cluster_index in rng.integers(0, len(clusters), size=len(clusters)):
            for row_index in clusters[cluster_index]:
                counts[row_index] += 1
        scores = {name: [helper.auc(y[:, t].tolist(), score_groups[name][t], counts) for t in range(12)]
                  for name in score_groups}
        if any(value is None for value in scores['baseline']):
            continue
        for name in candidates:
            deltas[name].append(float(np.mean(scores[name]) - np.mean(scores['baseline'])))
    assert all(len(values) >= 1000 for values in deltas.values())
    keep = np.array([uid not in linked_ids for uid in ids])
    result = {
        'status': 'FIXED_ENSEMBLE_DIAGNOSTIC_COMPLETE',
        'macro_auc': {name: float(values.mean()) for name, values in metrics.items()},
        'comparison_to_best_recipe': {name: {
            'delta': float((metrics[name] - metrics['baseline']).mean()),
            'paired_report_group_95_interval': np.quantile(deltas[name], [.025, .975]).tolist(),
            'valid_bootstrap_draws': len(deltas[name])} for name in candidates},
        'sensitivity_without_known_vit_report_connection': {
            'studies': int(keep.sum()),
            'macro_auc': {name: float(roc_auc_score(y[keep], a[keep])) for name, a in arrays.items()}},
        'per_target': {label: {name: float(values[t]) for name, values in metrics.items()} for t, label in enumerate(labels)},
        'rank_correlations': {a + '__' + b: float(frames[a].corrwith(frames[b], method='spearman').mean())
                              for a, b in [('vit', 'coatnet'), ('vit', 'resnet50'), ('coatnet', 'resnet50')]},
        'report_groups': len(clusters), 'protocol_sha256': sha(pp),
        'ensemble_prediction_sha256': sha(OUT / 'ensemble_diagnostic.csv'),
        'anchored_prediction_sha256': sha(OUT / 'anchored_diagnostic.csv'),
        'software_checks': 'Row and column order invariance;missing and duplicate IDs rejected;AUC independently replayed',
        'decision': 'HOLD_NO_SUBMISSION',
        'limitations': protocol['limitations'],
        'gpu_launched': False, 'competition_submission_enabled': False, 'expected_public_score': None,
    }
    if all(result['comparison_to_best_recipe'][name]['delta'] <= 0 for name in candidates):
        result['decision'] = 'DO_NOT_PROMOTE_THIS_FIXED_ENSEMBLE'
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_target'}, indent=2))


if __name__ == '__main__':
    main()
