"""Recompute saved synthetic precision checks and runtime projection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT/'runs/orthodiffusion_resource_v1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args();out=args.output
    result=json.loads((out/'result.json').read_text())
    contract=json.loads((LOCAL/'contract.json').read_text())
    assert result['status']=='PASS_SYNTHETIC_COMPATIBILITY_ONLY'
    assert not result['accuracy_measured'] and not result['competition_submission_enabled']
    assert result['contract_sha256']==sha(LOCAL/'contract.json')==sha(out/'contract.json')
    build=json.loads((LOCAL/'build.json').read_text())
    launch=json.loads((LOCAL/'launch.json').read_text())
    assert launch['status']=='PUSH_ACCEPTED' and launch['build_sha256']==sha(LOCAL/'build.json')
    for name, expected in build['sources'].items():assert sha(ROOT/name)==expected
    for name, expected in result['artifacts'].items():assert sha(out/name)==expected
    assert set(result['planes'])=={'axial','coronal','sagittal'}
    cpu=np.load(LOCAL/'cpu_fixture.npz');features=np.load(out/'features.npz')
    inputs=np.load(out/'full_inputs.npz')
    assert inputs['volume'].shape==tuple(contract['full_input_shape'])==inputs['noise'].shape
    assert np.isfinite(inputs['volume']).all() and np.isfinite(inputs['noise']).all()
    error=float(np.max(np.abs(features['axial_reduced_fp32']-cpu['feature'])))
    assert error<=contract['cpu_gpu_max_abs_error']
    assert abs(error-result['planes']['axial']['cpu_gpu_fp32_max_abs_error'])<1e-9
    verified={}
    for plane,row in result['planes'].items():
        assert row['strict_load'] and row['tensors']==257
        assert row['checkpoint_sha256']==contract['checkpoints'][plane]['sha256']
        fp32=features[plane+'_fp32'].astype(np.float64)
        amp=features[plane+'_amp'].astype(np.float64)
        assert fp32.shape==amp.shape==tuple(contract['expected_feature_shape'])
        assert np.isfinite(fp32).all() and np.isfinite(amp).all()
        difference=np.abs(fp32-amp)
        max_error=float(difference.max())
        relative=float(np.sqrt(np.mean(difference**2)/np.mean(fp32**2)))
        assert max_error<=contract['amp_max_abs_error']
        assert relative<=contract['amp_normalized_rmse']
        assert abs(max_error-row['amp_fp32_max_abs_error'])<1e-7
        assert abs(relative-row['amp_fp32_normalized_rmse'])<1e-7
        durations=row['amp_seconds']
        assert len(durations)==contract['timed_amp_forwards']
        assert np.isfinite(durations).all() and min(durations)>0
        assert row['amp_p95_seconds']==float(np.percentile(durations,95))
        assert row['replay_max_error']==0
        assert max(row['precision_peak_allocated_bytes'],row['amp_peak_allocated_bytes'])<contract['gpu_memory_cap_bytes']
        verified[plane]=dict(amp_max_abs_error=max_error,amp_normalized_rmse=relative,
                            amp_p95_seconds=row['amp_p95_seconds'],
                            precision_peak_gib=row['precision_peak_allocated_bytes']/2**30,
                            amp_peak_gib=row['amp_peak_allocated_bytes']/2**30)
    projection=4407*1.2*sum(r['amp_p95_seconds'] for r in result['planes'].values())
    assert abs(projection-result['single_gpu_encoder_only_4407_studies_3_planes_seconds_p95_margin20'])<1e-8
    assert result['worker_seconds']<contract['worker_seconds_cap']
    audit=dict(status='PASS_SAVED_NUMERICS_AND_PROJECTION_AUDIT_ONLY',
               result_sha256=sha(out/'result.json'),cpu_gpu_max_abs_error=error,planes=verified,
               encoder_only_4407_studies_3_planes_hours_p95_margin20=projection/3600,
               worker_seconds=result['worker_seconds'],clinical_accuracy_measured=False,
               competition_submission_enabled=False,official_best_unchanged=.944,
               limitations=result['exclusions']+[
                   'Deterministic replay and memory telemetry checked in pinned GPU worker; not independently rerun on GPU',
                   'Does not measure full DICOM pipeline or clinical generalization'])
    (LOCAL/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
