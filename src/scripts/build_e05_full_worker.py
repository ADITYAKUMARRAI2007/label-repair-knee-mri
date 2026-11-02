"""Stage full paired fold training from the frozen E04 trainer; never launch."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def replace_once(source, old, new):
    assert source.count(old)==1,old
    return source.replace(old,new)


def main():
    original=ROOT/'public_kernels/e04_convnext_folds_01/e04_full_fold_worker_preview.py'
    source=original.read_text()
    assert hashlib.sha256(source.encode()).hexdigest()=='082cd85baed7bf293983e86b572e7f42347a008f350756264c34f69346212a80'
    protocol=ROOT/'runs/e05_confidence_ablation/protocol.json'
    protocol_sha=hashlib.sha256(protocol.read_bytes()).hexdigest()
    source=replace_once(source,'FOLD = int(os.environ["E04_TRAIN_FOLD"])',
        'FOLD = int(os.environ["E04_TRAIN_FOLD"])\nARM = os.environ["E05_ARM"]\nassert ARM in {"control", "uniform"}')
    source=replace_once(source,'SCHEMA = "rsna_knee_e04_convnext_expert_fold_v1"',
        'SCHEMA = "rsna_knee_e05_confidence_fold_v1"')
    source=replace_once(source,'"schema": SCHEMA, "fold": FOLD,',
        f'"schema": SCHEMA, "arm": ARM, "protocol_sha256": {protocol_sha!r}, "fold": FOLD,')
    source=replace_once(source,'"weak_confidence": "clip_one_minus_source_range_0.25_to_1",',
        '"weak_confidence": "uniform_training_fold_mean" if ARM == "uniform" else "clip_one_minus_source_range_0.25_to_1",')
    source=replace_once(source,'def main():\n',
        (ROOT/'training/e05_confidence.py').read_text()+'\n\ndef main():\n')
    source=replace_once(source,'    prefix = f"e04_fold{FOLD}"',
        '    prefix = f"e05_{ARM}_fold{FOLD}"')
    # The full trainer reads verified cached images, never raw train_series.
    source=replace_once(source,'    root = find_competition_root()',
        '    root = find_unique("train.csv").parent')
    source=replace_once(source,'        checkpoint_overflows = overflows\n\n    if not state_path.exists():',
        '        checkpoint_overflows = overflows\n'
        '        if os.environ.get("E05_UPLOAD_LEASES") and completed_epoch in {1, 5, 9}:\n'
        '            from e05_checkpoint_export import publish_snapshot\n'
        '            publish_snapshot(out, prefix, f"e{completed_epoch:02d}", completed_epoch)\n\n'
        '    if not state_path.exists():')
    source=replace_once(source,'    train_set = SeriesWindows(training_rows, targets, masks, TRAIN_PER_SLOT, True)',
        '    if ARM == "uniform":\n        weights, _ = uniform_weak_weights(weights, gold, training_rows)\n'
        '    (out / f"{prefix}_effective_weights.json").write_text(json.dumps({\n'
        '        "arm": ARM, "fold": FOLD, "original_weight_file_sha256": E04_WEIGHT_SHA256,\n'
        '        "effective_weight_array_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),\n'
        '        "row_order": "owned_cache_ids", "heldout_rows_used_to_set_weights": False\n'
        '    }, indent=2) + "\\n")\n'
        '    train_set = SeriesWindows(training_rows, targets, masks, TRAIN_PER_SLOT, True)')
    source=replace_once(source,'resumes = bounded_files(f"{prefix}_last.pt")',
        'resumes = bounded_files(f"{prefix}_last.pt")')
    # Environment storage is selected at packaging time; use Colab here.
    source=source.replace('/kaggle/','/content/kaggle/')
    final_marker='    print(json.dumps(manifest, indent=2), flush=True)'
    if final_marker not in source:
        # Use the manifest file write, which is the definitive completion point.
        matches=[line for line in source.splitlines() if 'manifest.json' in line and 'write_text' in line]
        assert len(matches)==1,matches
        final_marker=matches[0]
    source=replace_once(source,final_marker,final_marker+'\n'
        '    if manifest["completed_epoch"] > 0 and os.environ.get("E05_UPLOAD_LEASES"):\n'
        '        from e05_checkpoint_export import publish_snapshot\n'
        '        publish_snapshot(out, prefix, "final", manifest["completed_epoch"])')
    compile(source,'e05_full_worker.py','exec')
    out=ROOT/'runs/e05_confidence_ablation';out.mkdir(exist_ok=True)
    dest=out/'full_worker.py';dest.write_text(source)
    record={'status':'FULL_WORKER_STAGED_NOT_LAUNCHED','code_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'protocol_sha256':protocol_sha,'original_e04_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
            'arms':['control','uniform'],'folds':[0,1,2,3,4],'epochs':14,
            'checkpoint_rule':'final_epoch_14_ema','resumable_full_optimizer_state':True,
            'colab_snapshot_epochs':[1,5,9,14],
            'export_helper_sha256':hashlib.sha256((ROOT/'training/e05_checkpoint_export.py').read_bytes()).hexdigest(),
            'remaining':['Resource pilot audit','Fresh available compute budget',
                         'Durable checkpoint transfer and full-job supervisor','Full CUDA execution'],
            'accuracy_measured':False,'competition_submission_enabled':False}
    (out/'full_worker_build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
