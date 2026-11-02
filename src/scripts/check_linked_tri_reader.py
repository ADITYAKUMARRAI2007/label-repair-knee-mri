"""Replay one linked public recipe on verified saved predictions; no GPU or submission."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'research/tri_reader_20261010'
OUT = ROOT / 'runs/tri_reader_20261010'
PRED = ROOT / 'runs/colab_n03/output'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def modules():
    tree = ast.parse((ROOT / 'public_kernels/fixed_reader_blend_v3/rsna-knee-n02-fixed-reader-blend-v3.py').read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'MODULES' for t in n.targets))


def main():
    OUT.mkdir(exist_ok=True)
    notebook_path = next(RESEARCH.glob('*.ipynb'))
    notebook = json.loads(notebook_path.read_text())
    receipt = json.loads((RESEARCH / 'source_receipt.json').read_text())
    assert sha(notebook_path) == receipt['source_sha256']
    source = lambda i: ''.join(notebook['cells'][i]['source'])
    baseline_modules = modules()
    # Exact source comparison, allowing only output filenames and an extra CSV copy.
    anchor = source(8).replace('    sub.to_csv("/kaggle/working/_sota_0949.csv", index=False)\n', '').replace(
        '    print("Saved SOTA 0.949 backup to /kaggle/working/_sota_0949.csv", flush=True)\n', '')
    assert anchor == baseline_modules['anchor/published_inference.py']
    original_crop = json.loads((ROOT / 'research/new_public/nartaa_0945_20261009/rsna-knee-0945-efficient-224crop.ipynb').read_text())
    crop = source(5).split('\n', 1)[1].replace('    out = "/kaggle/working/_c224.csv"', '    out = "/kaggle/working/submission.csv"')
    assert crop == ''.join(original_crop['cells'][4]['source'])
    for index, name in [(11, 'reader/preprocess.py'), (12, 'reader/knee.py'), (13, 'reader/infer.py')]:
        assert source(index).split('\n', 1)[1] == baseline_modules[name], name
    # Extract literals without executing public notebook code.
    assignments = {}
    for node in ast.parse(source(15)).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'TARGET_WEIGHTS':
                    assignments['reader_weights'] = ast.literal_eval(node.value)
                if isinstance(target, ast.Tuple) and [t.id for t in target.elts] == ['W_C224', 'W_OWN']:
                    assignments['crop_weight'], assignments['own_weight'] = ast.literal_eval(node.value)
    assert assignments['own_weight'] == 0
    assert assignments['crop_weight'] == .30
    assert set(assignments['reader_weights']) == set(json.loads(baseline_modules['contract.json'])['weights'])
    assert all(0 <= w <= 1 for w in assignments['reader_weights'].values())
    assert source(15)[source(15).index('legs = {'):].strip() == '''legs = {"c384": leg("_sota_0949"), "cnxt": leg("_own")}
coat, reader = legs["c384"], legs["cnxt"]
if W_C224 > 0:
    legs["c224"] = leg("_c224")
    coat = ((1 - W_C224) * coat + W_C224 * legs["c224"]).rank(method="average", pct=True)
if W_OWN > 0:
    legs["own"] = leg("_own2")
    reader = ((1 - W_OWN) * reader + W_OWN * legs["own"]).rank(method="average", pct=True)
print("mean spearman between legs over the 12 findings")
print(pd.DataFrame({a: {b: legs[a].corrwith(legs[b], method="spearman").mean() for b in legs} for a in legs}).round(3).to_string(), flush=True)

final = sota.copy()
for c in labels:
    final[c] = (1 - TARGET_WEIGHTS[c]) * coat[c] + TARGET_WEIGHTS[c] * reader[c]
final[labels] = final[labels].rank(method="average", pct=True)
assert final[labels].notna().all().all()
final.to_csv(WORK + "/submission.csv", index=False)
print(f"fusion written: legs {list(legs)}, c224 weight {W_C224}, own weight in reader {W_OWN}", flush=True)
print(final.head().to_string())'''
    protocol = {'recipe': assignments, 'source_sha256': sha(notebook_path),
                'prediction_cohort': 78, 'expert_studies': 58,
                'recipe_selection': 'Single user-linked published recipe; no fitting or parameter search',
                'bootstrap_seed': 20261010, 'bootstrap_draws': 2000,
                'comparison': 'Frozen 0.950 submission recipe on the same saved predictions',
                'gpu_launched': False, 'competition_submission_enabled': False,
                'limitation': 'Gold58 is reused and informed public model development. Descriptive screen, not independent validation or a leaderboard prediction.'}
    pp = OUT / 'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol, 'Frozen protocol changed'
    else:
        pp.write_text(json.dumps(protocol, indent=2) + '\n')
    # Independently verify saved predictions, N03 checkpoint receipt and serving ranks.
    spec = importlib.util.spec_from_file_location('n03_audit', ROOT / 'scripts/audit_n03_paired_results.py')
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    previous_audit = auditor.audit(PRED, bootstrap_draws=1000)
    np.testing.assert_allclose(previous_audit['baseline_macro'], .9256624221, atol=1e-9)
    old_protocol = json.loads((ROOT / 'runs/n03_paired_diagnostic/protocol.json').read_text())
    labels = list(assignments['reader_weights'])
    def read(name):
        f = pd.read_csv(PRED / name, dtype={'StudyInstanceUID': str})
        assert f.StudyInstanceUID.tolist() == old_protocol['cohort']
        values = f.set_index('StudyInstanceUID')[labels]
        assert np.isfinite(values.to_numpy()).all()
        return values
    anchor = read('fixed_reader_blend/anchor.csv').rank(method='average', pct=True)
    reader = read('fixed_reader_blend/reader.csv').rank(method='average', pct=True)
    crop = read('n03_diagnostic.csv').rank(method='average', pct=True)
    coat = ((1 - assignments['crop_weight']) * anchor + assignments['crop_weight'] * crop).rank(method='average', pct=True)
    weights = pd.Series(assignments['reader_weights'])
    linked = ((1 - weights) * coat + weights * reader).rank(method='average', pct=True)
    linked.to_csv(OUT / 'diagnostic_predictions.csv')
    baseline = read('n02_diagnostic.csv')
    official = pd.read_csv(ROOT / 'data/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    ids = old_protocol['diagnostic_ids']
    truth = official.loc[ids, labels].to_numpy(dtype=int)
    b, c = baseline.loc[ids].to_numpy(), linked.loc[ids].to_numpy()
    assert truth.shape == (58, 12) and np.isin(truth, [0, 1]).all()
    ba, ca = roc_auc_score(truth, b, average=None), roc_auc_score(truth, c, average=None)
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    deltas = []
    for _ in range(protocol['bootstrap_draws']):
        sample = rng.integers(0, len(ids), size=len(ids))
        y = truth[sample]
        if ((y.sum(0) == 0) | (y.sum(0) == len(ids))).any():
            continue
        deltas.append(float(roc_auc_score(y, c[sample]) - roc_auc_score(y, b[sample])))
    assert len(deltas) >= 1000
    result = {'status': 'EXACT_COMPONENT_REPLAY_COMPLETE_FULL_NOTEBOOK_NOT_EXECUTED',
              'baseline_macro_auc': float(ba.mean()), 'linked_macro_auc': float(ca.mean()),
              'delta': float((ca - ba).mean()),
              'per_target': {name: {'baseline_auc': float(ba[i]), 'linked_auc': float(ca[i]),
                                   'delta': float(ca[i] - ba[i])} for i, name in enumerate(labels)},
              'paired_study_bootstrap': {'valid': len(deltas), 'attempted': 2000,
                                         'delta_percentile_95_interval': np.quantile(deltas, [.025, .975]).tolist()},
              'decision': 'HOLD_NO_SUBMISSION',
              'component_source_match': 'Exact, except CSV filenames and an extra CSV copy',
              'execution_outputs_in_downloaded_notebook': sum(len(cell.get('outputs', [])) for cell in notebook['cells']),
              'official_0954_score_verified': False,
              'source_sha256': sha(notebook_path), 'protocol_sha256': sha(pp),
              'prediction_sha256': sha(OUT / 'diagnostic_predictions.csv'),
              'saved_prediction_receipt_sha256': sha(PRED / 'diagnostic_receipt.json'),
              'limitations': [protocol['limitation'],
                             'Ranks use the saved 78-study cohort; hidden-test cohort may change combined rankings.',
                             'Bootstrap resamples studies and is not a patient/report-group independence guarantee.',
                             'Existing checkpoint and decoder execution was reused; the complete linked notebook was not run on a GPU.'],
              'gpu_launched': False, 'competition_submission_enabled': False}
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_target'}, indent=2))


if __name__ == '__main__':
    main()
