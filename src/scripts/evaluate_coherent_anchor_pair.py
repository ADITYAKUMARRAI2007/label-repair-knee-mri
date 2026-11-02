"""Audit paired execution, then score three frozen recipes without fitting."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/window_integrity_20261010'
sys.path.insert(0, str(ROOT / 'inference'))
from coherent_anchor_pair import read_input_audit
from coherent_triplets import coherent_centers
from nartaa_reproduction_guard import reported_failure, validate_submission
from bootstrap_gold58 import grouped_scores, auc


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packaged_modules():
    path = ROOT / 'public_kernels/coherent_anchor_pair/pair.py'
    build = json.loads((RUN / 'pair_build.json').read_text())
    assert sha(path) == build['code_sha256']
    modules = ast.literal_eval(ast.parse(path.read_text()).body[0].value)
    contract = json.loads(modules['pair_contract.json'])
    assert hashlib.sha256(modules['pair_contract.json'].encode()).hexdigest() == build['contract_sha256']
    return modules, contract


def audit_execution(output):
    modules, contract = packaged_modules()
    sources, folder = output / 'coherent_pair_source', output / 'coherent_anchor_pair'
    assert not (folder / 'failure.json').exists()
    assert not (folder / 'rejected_submission.csv').exists()
    assert not (output / 'submission.csv').exists(), 'Only diagnostic outputs are eligible'
    for name, source in modules.items():
        assert (sources / name).read_text() == source, name
    receipt = json.loads((folder / 'receipt.json').read_text())
    assert receipt['status'] == 'PAIRED_ANCHOR_PREDICTIONS_READY_NOT_SCORED'
    assert receipt['contract'] == contract
    assert receipt['accuracy_measured'] is False and receipt['competition_submission_enabled'] is False
    assert 0 < receipt['elapsed_seconds'] <= contract['runtime_cap_seconds']
    assert receipt['decoder']['reported_failure_count'] == 0
    assert 'DECODER_ABI_PASS cp313' in (folder / 'decoder.log').read_text()
    inputs_path = folder / 'decoded_inputs.json'
    assert sha(inputs_path) == receipt['decoded_inputs_sha256']
    inputs = json.loads(inputs_path.read_text())
    assert set(inputs) == {'control', 'candidate'}
    cohort = json.loads((ROOT / 'runs/n03_paired_diagnostic/protocol.json').read_text())
    assert contract['studies'] == cohort['cohort']
    assert contract['diagnostic_ids'] == cohort['diagnostic_ids']
    for arm in ['control', 'candidate']:
        log = folder / f'{arm}.log'
        text = log.read_text()
        assert not any(reported_failure(line) for line in text.splitlines())
        assert '[weights] sha256 ' + contract['checkpoint_sha256'] in text
        assert read_input_audit(log, contract['studies']) == inputs[arm]
        record = receipt['arms'][arm]
        assert record['source_sha256'] == sha(sources / f'{arm}.py')
        assert record['execution']['reported_failure_count'] == 0
        expected = record['output']
        frame = pd.read_csv(folder / f'{arm}.csv', dtype={'StudyInstanceUID': str})
        assert frame['StudyInstanceUID'].tolist() == contract['studies']
        assert frame.columns.tolist() == pd.read_csv(ROOT / 'data/sample_submission.csv', nrows=0).columns.tolist()
        values = frame.iloc[:, 1:].to_numpy(dtype=float)
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        assert expected == {'rows': 78, 'columns': 13, 'sha256': sha(folder / f'{arm}.csv')}
    # Replay selection independently from the original function, using logged masks.
    tree = ast.parse(modules['control.py'])
    selector = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_eval_centers')
    scope = {'np': np}
    exec(compile(ast.Module(body=[selector], type_ignores=[]), '<selector-audit>', 'exec'), scope)
    removed = {}
    for uid in contract['studies']:
        control, candidate = inputs['control'][uid], inputs['candidate'][uid]
        assert control['study'] == candidate['study'] == uid
        for key in ['mask', 'shape', 'volume_sha256']:
            assert control[key] == candidate[key]
        assert control['shape'] == [96, 384, 384]
        digest = control['volume_sha256']
        assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        mask = np.asarray(control['mask'])
        original = scope['_eval_centers'](mask, 96, 94)
        kept = coherent_centers(original, mask, [26, 22, 18, 12, 18])
        assert control['windows'] == len(original)
        assert control['coherent_windows'] == candidate['windows'] == candidate['coherent_windows'] == len(kept)
        removed[uid] = len(original) - len(kept)
    return {'status': 'PASS_PAIRED_EXECUTION_AND_INPUT_IDENTITY',
            'receipt_sha256': sha(folder / 'receipt.json'), 'removed_windows': removed,
            'actual_checkpoint_bytes_independently_downloaded': False}


def fixed_recipes(anchor, reader, crop, protocol):
    labels = anchor.columns.tolist()
    assert anchor.index.equals(reader.index) and anchor.index.equals(crop.index)
    assert reader.columns.tolist() == crop.columns.tolist() == labels
    rank = lambda frame: frame.rank(method='average', pct=True)
    a, r, c = rank(anchor), rank(reader), rank(crop)
    weights = pd.Series(protocol['baseline_reader_weights']).loc[labels]
    n02 = rank(a * (1 - weights) + r * weights)
    recipe = protocol['tri_reader_recipe']
    assert recipe['own_weight'] == 0
    spatial = rank(a * (1 - recipe['crop_weight']) + c * recipe['crop_weight'])
    weights = pd.Series(recipe['reader_weights']).loc[labels]
    n04 = rank(spatial * (1 - weights) + r * weights)
    return {'anchor': a, 'n02': n02, 'n04': n04}


def paired_metrics(truth, controls, candidates, clusters):
    names = ['anchor', 'n02', 'n04']
    base_auc = {name: roc_auc_score(truth, controls[name], average=None) for name in names}
    candidate_auc = {name: roc_auc_score(truth, candidates[name], average=None) for name in names}
    grouped = {arm: {name: [grouped_scores(frame[:, t].tolist()) for t in range(12)]
                      for name, frame in frames.items()}
               for arm, frames in [('control', controls), ('candidate', candidates)]}
    for arm, expected in [('control', base_auc), ('candidate', candidate_auc)]:
        for name in names:
            replay = [auc(truth[:, t].tolist(), grouped[arm][name][t], [1] * len(truth)) for t in range(12)]
            np.testing.assert_allclose(replay, expected[name], atol=1e-12, rtol=0)
    draws = {name: [] for name in names}
    rng = np.random.default_rng(20261010)
    for _ in range(2000):
        counts = [0] * len(truth)
        for index in rng.integers(0, len(clusters), size=len(clusters)):
            for row in clusters[index]:
                counts[row] += 1
        if any(sum(counts[i] * int(truth[i, t]) for i in range(len(truth))) in [0, sum(counts)] for t in range(12)):
            continue
        for name in names:
            scores = {arm: [auc(truth[:, t].tolist(), grouped[arm][name][t], counts) for t in range(12)]
                      for arm in ['control', 'candidate']}
            draws[name].append(float(np.mean(scores['candidate']) - np.mean(scores['control'])))
    assert all(len(values) >= 1000 for values in draws.values())
    return {name: {'control_macro_auc': float(base_auc[name].mean()),
                   'candidate_macro_auc': float(candidate_auc[name].mean()),
                   'delta': float((candidate_auc[name] - base_auc[name]).mean()),
                   'paired_report_group_95_interval': np.quantile(draws[name], [.025, .975]).tolist(),
                   'valid_bootstrap_draws': len(draws[name]),
                   'per_target_control_auc': base_auc[name].tolist(),
                   'per_target_candidate_auc': candidate_auc[name].tolist()}
            for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN / 'pair_output')
    args = parser.parse_args()
    if not (args.output / 'coherent_anchor_pair/receipt.json').exists():
        print(json.dumps({'status': 'WAIT_FOR_PAIRED_GPU_OUTPUT', 'accuracy_available': False}))
        return
    audit = audit_execution(args.output)
    protocol = json.loads((RUN / 'pair_analysis_protocol.json').read_text())
    modules, contract = packaged_modules()
    assert protocol['pair_contract_sha256'] == hashlib.sha256(modules['pair_contract.json'].encode()).hexdigest()
    ids, expert_ids = contract['studies'], contract['diagnostic_ids']
    archive = ROOT / 'runs/colab_n03/output'
    paths = {'reader': archive / 'fixed_reader_blend/reader.csv',
             'crop224': archive / 'n03_diagnostic.csv',
             'old_anchor': archive / 'fixed_reader_blend/anchor.csv'}
    def read(path):
        frame = pd.read_csv(path, dtype={'StudyInstanceUID': str})
        assert frame['StudyInstanceUID'].tolist() == ids
        assert frame.columns.tolist() == pd.read_csv(ROOT / 'data/sample_submission.csv', nrows=0).columns.tolist()
        result = frame.set_index('StudyInstanceUID')
        values = result.to_numpy(dtype=float)
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        return result
    for name, path in paths.items():
        assert sha(path) == protocol['fixed_reference_sha256'][name]
    refs = {name: read(path) for name, path in paths.items()}
    arms = {arm: read(args.output / 'coherent_anchor_pair' / f'{arm}.csv') for arm in ['control', 'candidate']}
    labels = arms['control'].columns.tolist()
    predictions = {arm: fixed_recipes(frame, refs['reader'], refs['crop224'], protocol) for arm, frame in arms.items()}
    gold = pd.read_csv(ROOT / 'data/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    truth = gold.loc[expert_ids, labels].to_numpy(dtype=int)
    assert truth.shape == (58, 12) and np.isin(truth, [0, 1]).all()
    manifest = json.loads((ROOT / 'runs/E04_contract/e04_contract.json').read_text())
    groups = {row['StudyInstanceUID']: row['report_group'] for row in manifest['rows'] if row['expert']}
    clusters = [[i for i, uid in enumerate(expert_ids) if groups[uid] == group]
                for group in sorted(set(groups[uid] for uid in expert_ids))]
    selected = {arm: {name: frame.loc[expert_ids].to_numpy() for name, frame in frames.items()}
                for arm, frames in predictions.items()}
    scores = paired_metrics(truth, selected['control'], selected['candidate'], clusters)
    report = {'status': 'PAIRED_INPUT_ACCURACY_DIAGNOSTIC_COMPLETE', 'execution_audit': audit,
              'comparisons': scores, 'targets': labels,
              'archived_anchor_max_rank_drift': float((arms['control'].rank(pct=True) - refs['old_anchor'].rank(pct=True)).abs().to_numpy().max()),
              'analysis_protocol_sha256': sha(RUN / 'pair_analysis_protocol.json'),
              'ground_truth_sha256': sha(ROOT / 'data/train.csv'),
              'expert_studies': 58, 'report_groups': len(clusters),
              'decision': 'HOLD_FOR_REVIEW_NO_AUTOMATIC_SUBMISSION',
              'expected_public_score': None, 'competition_submission_enabled': False,
              'limitations': [*protocol['limitations'],
                             'Archived crop and reader branches are reused on exactly the same cohort.',
                             'All three comparisons are descriptive; no multiple-comparison correction or independent validation.',
                             'Report groups do not establish patient/site independence.']}
    for arm, frames in predictions.items():
        for name, frame in frames.items():
            frame.to_csv(RUN / f'{arm}_{name}_paired.csv')
    (RUN / 'pair_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['status', 'archived_anchor_max_rank_drift', 'decision']}))
    print(json.dumps({name: {key: value for key, value in score.items() if not key.startswith('per_target')}
                      for name, score in scores.items()}))


if __name__ == '__main__':
    main()
