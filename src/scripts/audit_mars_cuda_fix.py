"""Audit an executed resource pilot; its features are not accuracy evidence."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS/cuda_fix_run'
PACKAGE = ROOT / 'public_kernels/mars_resource_cuda_fix'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def modules():
    build = read(RUN.parent / 'cuda_fix_build.json')
    notebook = read(PACKAGE / 'pilot.ipynb')
    code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert len(code_cells) == 1
    code = code_cells[0]['source']
    code = ''.join(code) if isinstance(code, list) else code
    assert hashlib.sha256(code.encode()).hexdigest() == build['code_cell_sha256']
    source = ast.literal_eval(ast.parse(code).body[0].value)
    return source, json.loads(source['pilot_contract.json']), json.loads(source['resource_protocol.json'])


def audit(output):
    source, contract, protocol = modules()
    folder = output / 'mars_resource_pilot'
    assert not (output / 'submission.csv').exists()
    assert not (output / 'mars_resource_bootstrap_failure.json').exists()
    assert not (folder / 'failure.json').exists()
    for name, value in source.items():
        assert (output / 'mars_pilot_source' / name).read_text() == value, name
    execution = read(output / 'mars_resource_execution.json')
    assert execution['status'] == 'BOUNDED_PILOT_EXECUTION_COMPLETE'
    assert execution['contract_sha256'] == sha(output / 'mars_pilot_source/pilot_contract.json')
    assert execution['pilot_receipt_sha256'] == sha(folder / 'receipt.json')
    assert not execution['classification_fitting_enabled'] and not execution['competition_submission_enabled']
    assert 0 < execution['elapsed_seconds'] <= contract['bootstrap_and_worker_seconds_cap']
    assert 'DECODER_ABI_PASS cp313' in (output / 'mars_resource_bootstrap.log').read_text()
    result = read(folder / 'receipt.json')
    assert result['status'] == 'REAL_DICOM_CUDA_RESOURCE_PILOT_COMPLETE_NO_ACCURACY'
    assert result['protocol_sha256'] == sha(output / 'mars_pilot_source/resource_protocol.json')
    assert result['torch_version'] == contract['torch_version']
    assert result['monai_version'] == protocol['monai_version']
    assert result['allocated_device_count'] == 2
    assert isinstance(result['gpu_peak_bytes'], int) and 0 < result['gpu_peak_bytes'] <= 16 * (1 << 30)
    assert result['completed_studies'] == 24
    assert 0 < result['elapsed_seconds'] < protocol['complete_worker_seconds_cap']
    for name in ['label_values_used', 'accuracy_measured', 'classification_fitting_enabled',
                 'competition_submission_enabled', 'full_corpus_continuation_enabled']:
        assert result[name] is False, name
    parity = read(folder / 'fixture_parity.json')
    assert result['fixture_parity'] == parity
    assert parity['study_id'] == protocol['studies'][0]
    assert parity['reference_created_inside_competition_mount'] is True
    assert parity['finite'] is True
    assert parity['threshold'] == protocol['cuda_fixture_max_abs_error']
    assert math.isfinite(parity['cpu_cuda_max_abs_error'])
    assert 0 <= parity['cpu_cuda_max_abs_error'] <= parity['threshold']
    cases = [json.loads(line) for line in (folder / 'study_measurements.jsonl').read_text().splitlines()]
    assert [r['study_id'] for r in cases] == protocol['studies']
    assert {p.name for p in (folder / 'features').iterdir()} == {uid + '.npz' for uid in protocol['studies']}
    planes = ['Axial', 'Coronal', 'Sagittal']
    feature_contract = parity['extraction_contract']
    assert feature_contract['checkpoint_sha256'] == protocol['checkpoint_sha256']
    assert feature_contract['monai_version'] == protocol['monai_version']
    assert feature_contract['torch_version'] == contract['torch_version']
    assert feature_contract['device_type'] == 'cuda' and 't4' in feature_contract['gpu_name'].lower()
    assert feature_contract['planes'] == planes and feature_contract['features_shape'] == [3, 768, 3, 3, 3]
    assert feature_contract['dtype'] == 'float32' and feature_contract['autocast'] is False
    assert feature_contract['labels_used'] is False and feature_contract['global_pooling_applied'] is False
    for name, expected in feature_contract['source_sha256'].items():
        assert hashlib.sha256(source[name].encode()).hexdigest() == expected, name
    for case in cases:
        uid = case['study_id']
        with np.load(folder / 'features' / (uid + '.npz'), allow_pickle=False) as archive:
            assert set(archive.files) == {'features', 'present', 'record'}
            feature, present = archive['features'], archive['present']
            record = json.loads(str(archive['record'].item()))
        assert feature.shape == (3, 768, 3, 3, 3) and feature.dtype == np.float32
        assert present.shape == (3,) and present.dtype == np.bool_ and present.any()
        assert np.isfinite(feature).all() and np.all(feature[~present] == 0)
        assert record['study_id'] == uid and record['extraction_contract'] == feature_contract
        assert hashlib.sha256(json.dumps(feature_contract, sort_keys=True).encode()).hexdigest() == record['contract_sha256']
        assert digest(feature) == record['feature_sha256'] == case['feature_sha256']
        assert digest(present) == record['present_sha256']
        identity = record['input_identity']
        assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == record['input_sha256'] == case['input_sha256']
        assert identity['present_sha256'] == digest(present)
        receipt = identity['input_receipt']
        assert receipt['study_id'] == uid
        assert receipt['missing_mask'] == case['missing_mask'] == (~present).tolist()
        assert not receipt['label_values_used'] and not receipt['fallback_predictions_used']
        assert [p['plane'] for p in receipt['planes']] == planes
        assert case['present_planes'] == int(present.sum())
        slices = 0
        for index, plane in enumerate(receipt['planes']):
            if not present[index]:
                assert plane['state'] == 'missing_acquisition'
                continue
            assert plane['state'] == 'decoded'
            volume = plane['volume_receipt']
            assert volume['study_id'] == uid and volume['series_id'] == plane['series_id']
            assert volume['decoded_slices'] == volume['source_geometry']['native_slices'] == len(volume['selected_files'])
            assert volume['decoded_slices'] >= 2 and volume['full_physical_bounds_covered'] is True
            assert volume['fixed_physical_crop'] is False and volume['output_shape'] == [96, 96, 96]
            slices += volume['decoded_slices']
        assert slices == case['decoded_native_slices']
        assert case['decode_seconds'] > 0 and case['encoder_and_persistence_seconds'] > 0
        assert math.isclose(case['complete_study_seconds'], case['decode_seconds'] + case['encoder_and_persistence_seconds'], abs_tol=1e-9)
    latency = np.array([r['complete_study_seconds'] for r in cases])
    assert math.isclose(float(latency.mean()), result['mean_study_seconds'], abs_tol=1e-9)
    assert math.isclose(float(np.median(latency)), result['median_study_seconds'], abs_tol=1e-9)
    assert math.isclose(float(np.quantile(latency, .95)), result['p95_study_seconds'], abs_tol=1e-9)
    assert math.isclose(float(latency.mean()*4407*1.2/3600), result['serial_4407_studies_projection_hours_mean_20pct_margin'], abs_tol=1e-9)
    return {
        'status': 'PASS_REAL_DICOM_RESOURCE_OUTPUT_IDENTITY_AND_FEATURE_ARRAYS',
        'pilot_receipt_sha256': sha(folder / 'receipt.json'), 'studies': 24,
        'cpu_cuda_max_abs_error': parity['cpu_cuda_max_abs_error'],
        'serial_projection_mean_hours_with_margin': result['serial_4407_studies_projection_hours_mean_20pct_margin'],
        'allocated_gpu_hours_projection_not_established': True,
        'clinical_accuracy_measured': False, 'competition_submission_enabled': False,
        'limits': ['Raw DICOM/decoded volumes were not independently downloaded or replayed locally.',
                  'This is numerical/resource verification, not clinical validation or model promotion.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN / 'resource_output')
    args = parser.parse_args()
    if not (args.output / 'mars_resource_pilot/receipt.json').exists():
        print(json.dumps({'status': 'WAIT_FOR_EXECUTED_RESOURCE_PILOT', 'gpu_execution_verified': False})); return
    record = audit(args.output)
    (RUN / 'resource_execution_audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
