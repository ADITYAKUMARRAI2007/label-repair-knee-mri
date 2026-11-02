"""Synthetic-only full-resolution resource pilot. Never trains or submits."""
import gc
import json
import platform
import signal
import time
from pathlib import Path

import numpy as np
import torch
from orthodiffusion_encoder import load_encoder, feature_with_noise, synthetic_volume, sha

BASE = Path(__file__).resolve().parent
OUT = Path('/kaggle/working/orthodiffusion_resource')
OUT.mkdir(exist_ok=True)
CONTRACT = json.loads((BASE/'contract.json').read_text())
STARTED = time.monotonic()

def save(name, value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')

def timeout(*_):
    raise TimeoutError('600-second worker deadline')

def timed(model, x, noise, amp):
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        feature = feature_with_noise(model, x, noise, amp=amp)
    torch.cuda.synchronize()
    seconds = time.monotonic()-start
    assert torch.isfinite(feature).all() and feature.std() > 0
    return feature, seconds

def main():
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(CONTRACT['worker_seconds_cap'])
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    for name, expected in CONTRACT['source_files'].items():
        assert sha(BASE/'author'/name) == expected
    assert sha(BASE/'cpu_fixture.npz') == CONTRACT['cpu_fixture_sha256']
    assert sha(BASE/'orthodiffusion_encoder.py') == CONTRACT['adapter_sha256']
    cpu = np.load(BASE/'cpu_fixture.npz')
    result = dict(status='RUNNING', contract_sha256=sha(BASE/'contract.json'),
                  torch_version=torch.__version__, cuda=torch.version.cuda,
                  python=platform.python_version(), gpu=torch.cuda.get_device_name(0),
                  total_gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  planes={}, accuracy_measured=False, competition_submission_enabled=False)
    save('contract.json', CONTRACT)
    full_cpu = synthetic_volume(256)
    generator = torch.Generator(device='cpu').manual_seed(CONTRACT['seed'])
    full_noise_cpu = torch.randn(full_cpu.shape, generator=generator)
    np.savez_compressed(OUT/'full_inputs.npz', volume=full_cpu.numpy(), noise=full_noise_cpu.numpy())
    full = full_cpu.cuda(); noise = full_noise_cpu.cuda()
    reduced = torch.from_numpy(cpu['volume']).cuda()
    reduced_noise = torch.from_numpy(cpu['noise']).cuda()
    tensors = {}
    for plane, identity in CONTRACT['checkpoints'].items():
        matches = list(Path('/kaggle/input').rglob(identity['file']))
        assert len(matches) == 1, f'ambiguous/missing {plane} checkpoint'
        begin = time.monotonic()
        model = load_encoder(BASE/'author', matches[0], identity['sha256']).cuda()
        row = dict(checkpoint_sha256=identity['sha256'], strict_load=True,
                   tensors=257, load_seconds=time.monotonic()-begin)
        if plane == 'axial':
            small, duration = timed(model, reduced, reduced_noise, False)
            tensors['axial_reduced_fp32'] = small.cpu().numpy()
            error = float(np.max(np.abs(tensors['axial_reduced_fp32']-cpu['feature'])))
            row.update(cpu_gpu_fp32_max_abs_error=error, reduced_seconds=duration)
            assert error <= CONTRACT['cpu_gpu_max_abs_error'], 'CPU/GPU parity gate'
            del small
        torch.cuda.reset_peak_memory_stats()
        fp32, fp32_seconds = timed(model, full, noise, False)
        assert list(fp32.shape) == CONTRACT['expected_feature_shape']
        amp, amp_seconds = timed(model, full, noise, True)
        diff = (amp.float()-fp32).abs()
        max_error = float(diff.max())
        nrmse = float(diff.square().mean().sqrt()/fp32.square().mean().sqrt())
        fp32_norm = float(fp32.square().mean().sqrt())
        row.update(full_fp32_seconds=fp32_seconds, first_amp_seconds=amp_seconds,
                   full_feature_shape=list(fp32.shape), amp_fp32_max_abs_error=max_error,
                   amp_fp32_normalized_rmse=nrmse, feature_rms=fp32_norm,
                   precision_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        tensors[plane+'_fp32'] = fp32.cpu().numpy()
        tensors[plane+'_amp'] = amp.cpu().numpy()
        assert max_error <= CONTRACT['amp_max_abs_error'], 'AMP absolute-error gate'
        assert nrmse <= CONTRACT['amp_normalized_rmse'], 'AMP relative-error gate'
        del fp32, diff
        replay, _ = timed(model, full, noise, True)
        replay_error = float((replay-amp).abs().max())
        assert replay_error == 0, 'GPU deterministic replay gate'
        del replay, amp
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(CONTRACT['timed_amp_forwards']):
            feature, seconds = timed(model, full, noise, True)
            durations.append(seconds)
            del feature
        row.update(replay_max_error=replay_error, amp_seconds=durations,
                   amp_p50_seconds=float(np.median(durations)),
                   amp_p95_seconds=float(np.percentile(durations,95)),
                   amp_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
        assert row['precision_peak_allocated_bytes'] < CONTRACT['gpu_memory_cap_bytes']
        assert row['amp_peak_allocated_bytes'] < CONTRACT['gpu_memory_cap_bytes']
        result['planes'][plane] = row
        save('progress.json', result)
        print(json.dumps({'plane':plane,**row}),flush=True)
        del model
        gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(OUT/'features.npz', **tensors)
    result.update(status='PASS_SYNTHETIC_COMPATIBILITY_ONLY',
                  worker_seconds=time.monotonic()-STARTED,
                  single_gpu_encoder_only_4407_studies_3_planes_seconds_p95_margin20=
                    4407*1.2*sum(r['amp_p95_seconds'] for r in result['planes'].values()),
                  exclusions=['No DICOM decode or image preprocessing timing',
                              'No diagnostic heads, training, clinical accuracy or ensemble validation',
                              'Synthetic throughput does not establish total extraction/training cost'],
                  artifacts={p.name:sha(p) for p in OUT.glob('*.npz')})
    save('result.json',result)
    signal.alarm(0)
    print(json.dumps(result,indent=2),flush=True)

if __name__ == '__main__':
    try: main()
    except BaseException as error:
        save('failure.json', dict(error=type(error).__name__, message=str(error),
                                 seconds=time.monotonic()-STARTED))
        raise
