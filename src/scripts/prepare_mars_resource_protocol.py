"""Select a fixed, label-free real-data resource pilot; never launch a job."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = RUN / 'resource_protocol.json'
    assert not output.exists(), 'Preserve the frozen pilot cohort'
    audit_path = RUN / 'runtime15/legacy_compatibility_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'PASS_REAL_STUDY_MONAI15_REFERENCE_FEATURE_PARITY_CPU'
    assert audit['encoder_source_sha256'] == sha(ROOT / 'inference/mars_encoder.py')
    split_path = ROOT / 'runs/E03_contract/split.json'
    split = json.loads(split_path.read_text())
    eligible = set(split['train']) | set(split['weak_validation'])
    excluded = set(split['gold']) | set(split['excluded_gold_group_weak'])
    assert len(eligible) == 4345 and not eligible & excluded
    receipt_path = ROOT / 'runs/orthodiffusion_trial_v2/outputs/diffusion_probe_v2/volume_receipts.jsonl'
    stats = {}
    with receipt_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            uid = row['StudyInstanceUID']
            if uid not in eligible:
                continue
            planes = [p for p in row['planes'] if p['state'] == 'decoded']
            if not planes:
                continue
            extents = [max(p['native_shape'][0] * p['spacing_mm'][0],
                           p['native_shape'][1] * p['spacing_mm'][1],
                           p['native_slices'] * p['median_slice_spacing_mm']) for p in planes]
            stats[uid] = {
                'maximum_native_extent_mm': max(extents),
                'maximum_obliquity': max(1 - max(map(abs, p['slice_normal'])) for p in planes),
                'maximum_native_slice_count': max(p['native_slices'] for p in planes),
                'minimum_native_slice_count': min(p['native_slices'] for p in planes),
                'maximum_native_pixels': max(math.prod(p['native_shape']) for p in planes),
                'missing_prior_reader_planes': 3 - len(planes),
                'non_fluid_sensitive_planes': sum(p.get('Fluid_Sensitive') != 1 for p in planes),
                'non_fat_suppressed_planes': sum(p.get('Fat_Suppression') != 1 for p in planes),
            }
    assert set(stats) == eligible
    ordering = sorted(stats, key=lambda uid: hashlib.sha256(('20261010:MARS:' + uid).encode()).hexdigest())
    selected = {}
    def add(uid, reason):
        selected.setdefault(uid, []).append(reason)
    for uid in ordering[:8]:
        add(uid, 'fixed_hash_sample')
    extrema = [('maximum_native_extent_mm', True), ('maximum_native_extent_mm', False),
               ('maximum_obliquity', True), ('maximum_native_slice_count', True),
               ('minimum_native_slice_count', False), ('maximum_native_pixels', True),
               ('missing_prior_reader_planes', True), ('non_fluid_sensitive_planes', True),
               ('non_fat_suppressed_planes', True)]
    for field, descending in extrema:
        uid = sorted(stats, key=lambda u: ((-1 if descending else 1) * stats[u][field], u))[0]
        add(uid, ('maximum_' if descending else 'minimum_') + field)
    for uid in ordering:
        if len(selected) >= 24:
            break
        if uid not in selected:
            add(uid, 'fixed_hash_fill')
    assert len(selected) == 24 and not set(selected) & excluded
    meta_path = ROOT / 'data/train_series.csv'
    with meta_path.open() as handle:
        metadata_ids = {r['StudyInstanceUID'] for r in csv.DictReader(handle)}
    assert set(selected) <= metadata_ids
    result = {
        'schema': 'mars_rsna_resource_pilot_v1', 'status': 'FROZEN_NOT_LAUNCHED',
        'purpose': 'Measure actual full-DICOM decoding, CUDA parity and resource cost before any classifier training',
        'studies': list(selected),
        'selection': {uid: {'reasons': reasons, 'prior_geometry': stats[uid]}
                      for uid, reasons in selected.items()},
        'sources_sha256': {str(p.relative_to(ROOT)): sha(p)
                           for p in [split_path, receipt_path, meta_path, audit_path]},
        'eligibility': '4345weak studies; exclude all58official experts and4report-group neighbors',
        'checkpoint_sha256': audit['checkpoint_sha256'],
        'monai_version': '1.5.1', 'patch_merging': 'Preserve MONAI1.4 checkpoint offsets',
        'monai_wheel_sha256': audit['wheel_sha256'],
        'fixture_sha256': audit['fixture_sha256'],
        'cuda_fixture_max_abs_error': 0.003,
        'dtype': 'float32', 'autocast': False, 'tf32': False,
        'complete_worker_seconds_cap': 900,
        'projections': 'Report measured study latency including decoding; no automatic full-corpus continuation',
        'classification_fitting_enabled': False, 'label_values_used': False,
        'competition_submission_enabled': False,
        'launch_dependency': 'Evaluate the accepted coherent-window experiment before allocating this pilot',
        'limitations': ['Geometry extrema refer to the prior reader acquisition selection.',
                       'Small purposive sample cannot establish complete corpus coverage or clinical accuracy.',
                       'Exact upstream knee preprocessing remains unavailable.'],
    }
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'studies': len(selected),
                      'expert_overlap': len(set(selected) & excluded),
                      'new_gpu_jobs': 0, 'new_competition_submissions': 0}))


if __name__ == '__main__':
    main()
