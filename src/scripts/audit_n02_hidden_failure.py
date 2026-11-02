"""Reproduce N02's stricter-than-published slice policy on a local real series.

This diagnoses a wrapper defect, not the inaccessible hidden-test traceback.
No model inference, training, network requests, or submission is performed.
"""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runs/orthodiffusion_volume_v1/test_dependencies'),
               str(ROOT / 'inference')]
import numpy as np
from nartaa_reproduction_guard import reported_failure
from reader_recovery import observed_load_series
from fixed_reader_blend_v2 import checked_view


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'research/new_public/after_n01_20261008/fixed_v3_components/preprocess.py'
    wrapper = ROOT / 'inference/fixed_reader_blend_v2.py'
    tree = ast.parse(source.read_text())
    # pandas is used only by the unused cache-building main(), not load_series.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(a.name == 'pandas' for a in node.names))]
    env = {'__name__': 'published_preprocess_audit'}
    exec(compile(tree, str(source), 'exec'), env)
    load = env['load_series']
    node = next(n for n in ast.walk(ast.parse(wrapper.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == 'checked_load')
    checked_env = {'original_load': load}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(wrapper), 'exec'), checked_env)
    checked = checked_env['checked_load']
    series = sorted({p.parent for p in
        (ROOT / 'runs/orthodiffusion_volume_v2/raw_failed_study').rglob('*.dcm')})[0]
    clean, clean_info = load(str(series))
    assert clean_info['n_bad'] == 0 and len(clean) >= 3
    clean_checked, _ = checked(str(series))
    assert np.array_equal(clean, clean_checked)
    with tempfile.TemporaryDirectory() as temp:
        case = Path(temp) / 'partially_readable_series'
        shutil.copytree(series, case)
        (case / 'unreadable.dcm').write_bytes(b'not a readable DICOM slice')
        recovered, recovered_info = load(str(case))
        assert recovered_info['n_bad'] == 1
        assert np.array_equal(clean, recovered)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            try:
                checked(str(case))
            except AssertionError as error:
                rejection = str(error)
            else:
                raise AssertionError('Expected submitted wrapper to reject recoverable series')
        assert reported_failure(log.getvalue())
        repaired_log = io.StringIO()
        with contextlib.redirect_stdout(repaired_log):
            repaired, repaired_info = observed_load_series(load, str(case))
        assert np.array_equal(repaired, recovered) and repaired_info == recovered_info
        assert 'READER_ACQUISITION_RECOVERY' in repaired_log.getvalue()
        assert not reported_failure(repaired_log.getvalue())
    # A short acquisition remains skippable by the original dataset; an entirely
    # unusable study must still be rejected rather than receiving fake predictions.
    short = np.zeros((2, 3, 3), dtype=np.uint8)
    with contextlib.redirect_stdout(io.StringIO()):
        short_result = observed_load_series(lambda _: (short, {'n_bad': 0}), 'short')
    assert short_result[0] is short
    try:
        checked_view((None, np.zeros(6, dtype=bool), None))
    except AssertionError:
        pass
    else:
        raise AssertionError('No-view study must remain rejected')
    checked_view((None, np.array([True, False]), None))
    result = dict(
        status='CONFIRMED_RECOVERABLE_INPUT_REJECTED_BY_WRAPPER',
        hidden_failure_cause_confirmed=False,
        official_submission_id=56964335,
        original_preprocess_sha256=digest(source), submitted_wrapper_sha256=digest(wrapper),
        audit_sha256=digest(Path(__file__)),
        clean_series=str(series.relative_to(ROOT)), clean_info=clean_info,
        injected_case_info=recovered_info,
        published_recovery_array_exact=True,
        submitted_wrapper_rejection=rejection,
        parent_guard_would_reject_even_if_child_exits_zero=True,
        proposed_recovery_sha256=digest(ROOT / 'inference/reader_recovery.py'),
        repair_checks=['exact_published_recovered_array_and_metadata',
                       'warning_visible_without_fatal_marker',
                       'short_series_return_preserved_for_original_skip',
                       'no_usable_views_still_rejected'],
        limitations=['Unreadable file was injected; its presence in hidden data is unknown',
                     'Does not rule out a separate memory, decoder, or other runtime failure'],
        next_action='Restore published per-series recovery with explicit warnings; retain rejection of studies with no usable views; qualify before any submission',
    )
    out = ROOT / 'runs/fixed_reader_blend_v2/hidden_failure_local_audit.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
