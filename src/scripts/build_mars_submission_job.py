"""Package a fixed candidate only after complete training and validation review."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/foundation_refresh_20261010/MARS'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literal_modules(code):
    assignments = [node for node in ast.parse(code).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'MODULES' for target in node.targets)]
    assert len(assignments) == 1
    return ast.literal_eval(assignments[0].value)


def main():
    approval_path = BASE / 'serving_resource_approval.json'
    if not approval_path.exists():
        print(json.dumps({'status': 'WAIT_FOR_TRAINED_VALIDATED_CANDIDATE_AND_SERVING_RESOURCE_REVIEW',
                          'new_gpu_jobs': 0, 'competition_submission_enabled': False})); return
    assert not (BASE / 'serving_launch.json').exists(), 'Preserve an accepted or uncertain run'
    approval = json.loads(approval_path.read_text())
    assert approval['decision'] == 'APPROVE_FIXED_MARS_VISIBLE_QUALIFICATION'
    evaluation_path = BASE / 'classifier_evaluation.json'
    evaluation = json.loads(evaluation_path.read_text())
    assert sha(evaluation_path) == approval['candidate_evaluation_sha256']
    assert evaluation['decision'] == 'REVIEW_SERVING_AND_EXECUTION_PARITY_BEFORE_SUBMISSION'
    assert evaluation['fixed_blend_delta'] > .001 and evaluation['paired_report_group_95_interval'][0] > 0
    audit = json.loads((BASE / 'classifier_execution_audit.json').read_text())
    assert audit['status'] == 'PASS_NATIVE_FIXED_CLASSIFIER_EXECUTION_CHAIN'
    assert audit['heldout_head_predictions_replayed'] == 4407
    trained = ROOT / audit['training_output']
    assert sha(trained / 'training_receipt.json') == approval['training_receipt_sha256'] == evaluation['training_receipt_sha256']
    training_receipt = json.loads((trained / 'training_receipt.json').read_text())
    assert training_receipt['status'] == 'FIVE_FROZEN_MARS_HEADS_AND_COMPLETE_OOF_TRAINED'
    assert training_receipt['studies'] == 4407
    assert training_receipt['extraction_contract_sha256'] == approval['extraction_contract_sha256']
    assert sha(trained / 'heldout_replay.json') == approval['heldout_replay_sha256'] == audit['heldout_replay_sha256']
    assert sha(trained.parent / 'execution_receipt.json') == approval['classifier_execution_receipt_sha256']
    protocol_path, serving_path = BASE / 'classifier_protocol.json', BASE / 'serving_protocol.json'
    protocol, serving = [json.loads(p.read_text()) for p in [protocol_path, serving_path]]
    assert sha(protocol_path) == serving['classifier_protocol_sha256'] == evaluation['protocol_sha256']
    assert training_receipt['protocol_sha256'] == sha(protocol_path)
    assert protocol['fixed_rank_blend'] == serving['fixed_rank_blend'] == {
        'baseline': .9, 'candidate': .1, 'target_weights_searched': False}
    assert sha(ROOT / 'inference/mars_head_inference.py') == serving['head_source_sha256']
    assert sha(ROOT / 'inference/fixed_reader_blend.py') == serving['fusion_source_sha256']
    assert approval['head_checkpoint_sha256'] == {
        f'mars_head_fold{fold}.pt': sha(trained / f'mars_head_fold{fold}.pt') for fold in range(5)}
    baseline_root = ROOT / 'public_kernels/fixed_reader_blend_v3'
    baseline_path = baseline_root / 'rsna-knee-n02-fixed-reader-blend-v3.py'
    baseline_build = json.loads((ROOT / 'runs/fixed_reader_blend_v3/build.json').read_text())
    assert sha(baseline_path) == baseline_build['sources'][str(baseline_path.relative_to(ROOT))]
    baseline_submission = json.loads((ROOT / 'runs/fixed_reader_blend_v3/submission.json').read_text())
    assert baseline_submission['submission_id'] == 56967269
    assert baseline_submission['official_status'] == 'COMPLETE' and baseline_submission['our_public_score'] == .950
    baseline_modules = literal_modules(baseline_path.read_text())
    baseline_contract = json.loads(baseline_modules['contract.json'])
    modules = {'baseline.py': baseline_path.read_text(),
        'inference/mars_blend_driver.py': (ROOT / 'inference/mars_blend_driver.py').read_text(),
        'fusion/fixed_reader_blend.py': (ROOT / 'inference/fixed_reader_blend.py').read_text(),
        'fusion/nartaa_reproduction_guard.py': (ROOT / 'inference/nartaa_reproduction_guard.py').read_text(),
        'mars/classifier_protocol.json': protocol_path.read_text(),
        'mars/serving_protocol.json': serving_path.read_text()}
    corpus_notebook = json.loads((ROOT / 'public_kernels/mars_full_corpus/pilot.ipynb').read_text())
    accepted_corpus = literal_modules(corpus_notebook['cells'][0]['source'])
    corpus_contract = json.loads(accepted_corpus['corpus_contract.json'])
    for name in ['mars_encoder.py', 'mars_volume.py', 'mars_study.py', 'mars_feature_cache.py',
                 'mars_legacy_merging.py', 'orthodiffusion_volume.py', 'check_decoders.py']:
        source = accepted_corpus[name]
        assert hashlib.sha256(source.encode()).hexdigest() == corpus_contract['module_sha256'][name]
        modules['mars/' + name] = source
    for name, relative in [('mars_head_inference.py', 'inference/mars_head_inference.py'),
                           ('mars_test_inference.py', 'inference/mars_test_inference.py'),
                           ('mars_spatial_head.py', 'training/mars_spatial_head.py')]:
        modules['mars/' + name] = (ROOT / relative).read_text()
    assert hashlib.sha256(modules['mars/mars_spatial_head.py'].encode()).hexdigest() == protocol['sources_sha256']['training/mars_spatial_head.py']
    test_contract = {'torch_version': corpus_contract['torch_version'], 'monai_version': '1.5.1',
        'training_receipt_sha256': approval['training_receipt_sha256'],
        'extraction_contract_sha256': approval['extraction_contract_sha256'],
        'module_sha256': {name.removeprefix('mars/'): hashlib.sha256(source.encode()).hexdigest()
            for name, source in modules.items() if name.startswith('mars/')}}
    modules['mars/test_contract.json'] = json.dumps(test_contract, indent=2) + '\n'
    contract = {key: approval[key] for key in ['training_receipt_sha256', 'heldout_replay_sha256',
        'classifier_execution_receipt_sha256', 'head_checkpoint_sha256', 'extraction_contract_sha256',
        'visible_seconds_cap', 'hidden_seconds_cap']}
    assert contract['visible_seconds_cap'] > baseline_contract['visible_seconds_cap'] + 120
    assert baseline_contract['worker_seconds_cap'] + 120 < contract['hidden_seconds_cap']
    assert contract['visible_seconds_cap'] < contract['hidden_seconds_cap']
    assert contract['hidden_seconds_cap'] + 120 < approval['platform_timeout_seconds']
    contract.update(schema='fixed_n02_mars_serving_execution_v1',
        torch_version=corpus_contract['torch_version'],
        baseline_visible_seconds_cap=baseline_contract['visible_seconds_cap'],
        baseline_hidden_seconds_cap=baseline_contract['worker_seconds_cap'],
        asset_sha256=corpus_contract['asset_sha256'], monai_wheel=corpus_contract['monai_wheel'],
        decoder_wheels=corpus_contract['decoder_wheels'],
        resource_approval_sha256=sha(approval_path), candidate_evaluation_sha256=sha(evaluation_path),
        module_sha256={name: hashlib.sha256(source.encode()).hexdigest() for name, source in modules.items()})
    modules['blend_execution_contract.json'] = json.dumps(contract, indent=2) + '\n'
    for name, source in modules.items():
        if name.endswith('.py'): compile(source, name, 'exec')
    code = 'MODULES=' + repr(modules) + '\n' + (
        "exec(compile(MODULES['inference/mars_blend_driver.py'], 'mars_blend_driver.py', 'exec'))\n")
    ast.parse(code)
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': corpus_notebook['metadata'],
        'cells': [{'cell_type': 'code', 'id': 'fixed-blend-driver', 'metadata': {},
                   'execution_count': None, 'outputs': [], 'source': code}]}
    metadata = json.loads((baseline_root / 'kernel-metadata.json').read_text()); metadata.pop('id_no', None)
    metadata.update(id='adityakumarrai0401/rsna-knee-n02-mars-fixed-blend',
        title='RSNA Knee N02 MARS Fixed Blend', code_file='pilot.ipynb', kernel_type='notebook',
        kernel_sources=['adityakumarrai0401/rsna-knee-mars-frozen-spatial-classifier'],
        dataset_sources=sorted(set(metadata['dataset_sources']) | {
            'adityakumarrai0401/rsna-knee-mars-model-assets-v2'}))
    package = ROOT / 'public_kernels/mars_fixed_blend'; package.mkdir(exist_ok=True)
    (package / 'pilot.ipynb').write_text(json.dumps(notebook, indent=2) + '\n')
    (package / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    result = {'status': 'QUALIFIED_FIXED_BLEND_PACKAGED_NOT_LAUNCHED',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'code_cell_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'notebook_sha256': sha(package / 'pilot.ipynb'), 'metadata_sha256': sha(package / 'kernel-metadata.json'),
        'resource_approval_sha256': sha(approval_path), 'candidate_evaluation_sha256': sha(evaluation_path),
        'local_image_feature_or_label_arrays_embedded': False, 'new_gpu_jobs': 0,
        'competition_submission_enabled': False, 'expected_public_score': None}
    (BASE / 'serving_build.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
