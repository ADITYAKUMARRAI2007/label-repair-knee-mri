"""Execute two frozen readers and a fixed published rank blend; never submit."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from nartaa_reproduction_guard import sha, read_csv, run_source, validate_submission
from reader_recovery import observed_load_series


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = ((i + 1 + j) / 2) / len(order)
        for index in order[i:j]:
            out[index] = rank
        i = j
    return out


def blend(anchor, reader, output, weights, test, sample):
    validate_submission(anchor, test, sample)
    validate_submission(reader, test, sample)
    fields, aa = read_csv(anchor)
    _, bb = read_csv(reader)
    assert set(weights) == set(fields[1:])
    rows = [{'StudyInstanceUID': r['StudyInstanceUID']} for r in aa]
    for label in fields[1:]:
        w = weights[label]
        assert 0 <= w <= 1
        a = average_ranks([float(r[label]) for r in aa])
        b = average_ranks([float(r[label]) for r in bb])
        fused = average_ranks([(1-w)*x + w*y for x, y in zip(a, b)])
        for row, value in zip(rows, fused):
            row[label] = value
    with Path(output).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return validate_submission(output, test, sample)


def checked_view(result):
    assert bool(result[1].any()), 'reader study has no decoded acquisition'
    return result


def reader_main(root):
    """Preserve published acquisition recovery, log it, and reject unusable studies."""
    sys.path.insert(0, str(root / 'reader'))
    import infer
    contract = json.loads((root / 'contract.json').read_text())
    runtime = json.loads((root / 'reader_runtime.json').read_text())
    original_load = infer.load_series
    original_item = infer.DicomStudyDataset.__getitem__
    original_models = infer.load_models

    def checked_load(path):
        return observed_load_series(original_load, path)

    def checked_item(self, i):
        return checked_view(original_item(self, i))

    infer.load_series = checked_load
    infer.DicomStudyDataset.__getitem__ = checked_item
    def checked_models(paths, device='cuda'):
        import torch
        models = original_models(paths, device)
        # Visible data has only three studies; exercise the hidden batch size of four
        # with all six slots populated before accepting the operational qualification.
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            x = torch.zeros((4,6,16,3,384,384),dtype=torch.uint8,device=device)
            mask = torch.ones((4,6),dtype=torch.bool,device=device)
            pos = torch.zeros((4,6,16),device=device)
            for model,res in models:
                prediction = model(x,mask,pos,res=res)
                assert prediction.shape == (4,12) and torch.isfinite(prediction).all()
            del x,mask,pos,prediction
        torch.cuda.synchronize()
        print('READER_BATCH4_ALL6VIEWS_PASS peak_allocated_bytes=' +
              str(torch.cuda.max_memory_allocated()),flush=True)
        torch.cuda.empty_cache()
        if runtime['qualification']:
            from reader_pipeline_preflight import qualify
            qualify(infer, models, runtime['data'], contract['qualification_studies'],
                    str(Path(runtime['output']).parent / 'pipeline_preflight.json'))
        return models
    infer.load_models = checked_models
    # The published source already uses strict model.load_state_dict.
    result = infer.run(runtime['data'], 'test', runtime['checkpoints'], k=16, workers=4, bs=4)
    result.to_csv(runtime['output'])
    print('READER_COMPLETE rows=' + str(len(result)), flush=True)


def main():
    root = Path(__file__).resolve().parent
    if '--reader' in sys.argv:
        return reader_main(root)
    start = time.monotonic()
    contract = json.loads((root / 'contract.json').read_text())
    work = Path('/kaggle/working')
    out = work / 'fixed_reader_blend'
    out.mkdir(exist_ok=False)
    submission = work / 'submission.csv'
    assert not submission.exists(), 'stale submission exists'
    try:
        for name, expected in contract['module_sha256'].items():
            assert sha(root / name) == expected, name
        import torch
        import timm
        assert torch.cuda.device_count() == 2
        assert str(torch.__version__) == contract['torch_version']
        assert timm.__version__ == contract['timm_version']
        data = next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
                     Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
                    if (p / 'test.csv').is_file())
        assets = next(p for p in [Path('/kaggle/input/rsna-knee-2-5d-convnext-reader'),
                     Path('/kaggle/input/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader')]
                      if (p / 'cnxt_v0_fold0.pt').is_file())
        checkpoints = [str(assets / name) for name in sorted(contract['reader_checkpoints'])]
        assert len(checkpoints) == 3
        for name, record in contract['reader_checkpoints'].items():
            assert sha(assets / name) == record['sha256'], name
        _, tests = read_csv(data / 'test.csv')
        cap = contract['visible_seconds_cap'] if len(tests) <= 3 else contract['worker_seconds_cap']
        def remaining():
            seconds = cap - (time.monotonic() - start)
            if seconds <= 0:
                raise TimeoutError('combined readers exceeded runtime cap')
            return seconds
        # Install only published decoder wheels, with a separate subprocess environment.
        target = work / 'reader_decoder_env'
        abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
        assert abi == 'cp313', f'Unexpected runtime ABI {abi}'
        decoder_assets = next(p for p in [Path('/kaggle/input/rsna-knee-decoder-cp313'),
             Path('/kaggle/input/datasets/adityakumarrai0401/rsna-knee-decoder-cp313')]
             if (p / 'decoder_identity.json').is_file())
        wheel_names = contract['decoder_wheels'][abi]
        wheels = []
        for name, expected in wheel_names.items():
            assert sha(decoder_assets / name) == expected, name
            wheels.append(str(decoder_assets / name))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--target', str(target)] + wheels, check=True, timeout=min(180, remaining()))
        probe = root / 'check_decoders.py'
        probe.write_text('TARGET=' + repr(str(target)) + '\n' + "import sys,json,importlib.metadata\nsys.path.insert(0, TARGET)\nimport pylibjpeg,libjpeg,openjpeg\nfrom pydicom.pixels import get_decoder\nfrom pydicom.uid import JPEGBaseline8Bit,JPEGLosslessSV1,JPEGLSLossless,JPEG2000Lossless\nversions={name:importlib.metadata.version(name) for name in ['pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg']}\nassert versions=={'pylibjpeg':'2.1.0','pylibjpeg-libjpeg':'2.4.0','pylibjpeg-openjpeg':'2.6.0'},versions\nfor uid in [JPEGBaseline8Bit,JPEGLosslessSV1,JPEGLSLossless,JPEG2000Lossless]:\n    assert get_decoder(uid).is_available, str(uid)\nprint('DECODER_ABI_PASS cp313 '+json.dumps(versions),flush=True)\n")
        decoder_check = run_source(probe, out / 'decoder.log', min(60,remaining()), work)
        anchor_contract = json.loads((root / 'anchor/contract.json').read_text())
        anchor_source = root / 'anchor/published_inference.py'
        assert sha(anchor_source) == anchor_contract['source_sha256']
        anchor_model = next(Path('/kaggle/input') / suffix / anchor_contract['checkpoint_filename']
            for suffix in ['rsna-knee-publication-swa-weights-20261007',
                           'datasets/nartaa/rsna-knee-publication-swa-weights-20261007']
            if (Path('/kaggle/input') / suffix / anchor_contract['checkpoint_filename']).is_file())
        assert sha(anchor_model) == anchor_contract['checkpoint_sha256']
        # One subprocess layer lets the shared timeout stop the whole inference group.
        n01 = run_source(anchor_source, out / 'anchor.log', remaining(), work)
        anchor_output = validate_submission(submission, data / 'test.csv', data / 'sample_submission.csv')
        n01_receipt = dict(status='PASS_EXECUTION_ONLY',contract=anchor_contract,
                          source_sha256=sha(anchor_source),checkpoint_sha256=sha(anchor_model),
                          submission=anchor_output,**n01)
        shutil.move(submission, out / 'anchor.csv')
        runtime = dict(data=str(data), checkpoints=checkpoints, output=str(out / 'reader.csv'),
                       qualification=len(tests) <= 3)
        (root / 'reader_runtime.json').write_text(json.dumps(runtime))
        runner = root / 'run_reader.py'
        runner.write_text('import os,runpy,sys\n'
            + f"sys.path.insert(0,{str(target)!r})\n"
            + "sys.argv=['fixed_reader_blend.py','--reader']\n"
            + f"runpy.run_path({str(Path(__file__).resolve())!r},run_name='__main__')\n")
        reader = run_source(runner, out / 'reader.log', remaining(), work)
        pending = out / 'blend.pending.csv'
        details = blend(out / 'anchor.csv', out / 'reader.csv', pending,
                        contract['weights'], data / 'test.csv', data / 'sample_submission.csv')
        assert n01_receipt['status'] == 'PASS_EXECUTION_ONLY'
        assert remaining() > 0
        pending.replace(submission)
        receipt = dict(status='PASS_EXECUTION_ONLY', contract=contract, anchor=n01, reader=reader,
                       anchor_receipt=n01_receipt, decoder_check=decoder_check, output=details,
                       components={name: sha(out / name) for name in ['anchor.csv','reader.csv']},
                       elapsed_seconds=time.monotonic()-start, runtime_cap=cap,
                       accuracy_measured=False, competition_submission_enabled=False)
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print('FIXED_BLEND_COMPLETE ' + json.dumps(details), flush=True)
    except BaseException as error:
        if submission.exists():
            submission.replace(out / 'rejected_submission.csv')
        (out / 'failure.json').write_text(json.dumps(dict(status='REJECTED',
            error_type=type(error).__name__, message=str(error)), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
