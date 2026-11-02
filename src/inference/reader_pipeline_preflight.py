"""Exercise the actual multi-worker reader on a fixed, label-free training cohort."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def qualify(infer, models, data, studies, output):
    import numpy as np
    import pandas as pd
    import torch

    started = time.monotonic()
    assert len(studies) == len(set(studies)) == 20
    series = pd.read_csv(Path(data) / 'train_series.csv')
    series = series[series.StudyInstanceUID.isin(set(studies))]
    root = Path(data) / 'train_series'
    n_files = {sr: min(len(os.listdir(root / st / sr)), infer.MAX_SLICES)
               for st, sr in zip(series.StudyInstanceUID, series.SeriesInstanceUID)}
    table = infer.build_study_table(series, n_files)
    dataset = infer.DicomStudyDataset(studies, table, str(root), 16)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)
    records = []
    predictions = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch, (x, mask, pos) in enumerate(loader):
            assert x.shape == (4, 6, 16, 3, 384, 384)
            assert mask.any(dim=1).all(), 'preflight study has no usable views'
            x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                p = torch.stack([m(x, mask, pos, res=res).float().sigmoid()
                                 for m, res in models]).mean(0)
            assert p.shape == (4, 12) and torch.isfinite(p).all()
            predictions.append(p.cpu().numpy())
            records.append({'batch': batch, 'rows': 4, 'usable_slots': mask.sum(1).tolist()})
            print('READER_PIPELINE_BATCH ' + json.dumps(records[-1]), flush=True)
        del x, mask, pos, p
    assert len(records) == 5
    # Same recovered pixel array after an injected unreadable file, on this exact
    # Linux decoder environment. No modification to competition inputs.
    st = studies[0]
    sr = next(c[0] for c in table[st] if c)
    clean, _ = infer.load_series(str(root / st / sr))
    with tempfile.TemporaryDirectory(prefix='reader_recovery_') as temp:
        copied = Path(temp) / 'series'
        shutil.copytree(root / st / sr, copied)
        (copied / 'injected_unreadable.dcm').write_bytes(b'injected unreadable slice')
        recovered, info = infer.load_series(str(copied))
        assert info['n_bad'] >= 1 and np.array_equal(clean, recovered)
    pred = np.concatenate(predictions)
    receipt = dict(status='PASS_PIPELINE_AND_RECOVERY', studies=studies, batches=records,
                   rows=20, batch_size=4, workers=4, prediction_shape=list(pred.shape),
                   predictions_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),
                   gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                   elapsed_seconds=time.monotonic() - started,
                   recovery_pixel_array_exact=True, labels_used=False, accuracy_measured=False)
    assert receipt['gpu_peak_bytes'] < 14 * 1024**3
    Path(output).write_text(json.dumps(receipt, indent=2) + '\n')
    print('READER_PIPELINE_PASS ' + json.dumps(receipt), flush=True)
    torch.cuda.empty_cache()
