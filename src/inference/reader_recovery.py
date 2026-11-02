"""Log the published reader's input recovery without changing its returned data."""
import json


def observed_load_series(original_load, path):
    try:
        volume, info = original_load(path)
    except Exception as error:
        # The published dataset skips this acquisition and tries its other slots.
        print('READER_ACQUISITION_SKIPPED ' + json.dumps({
            'path': str(path), 'error_type': type(error).__name__,
            'message': str(error)[:300],
        }), flush=True)
        raise
    if info.get('n_bad', 0) or len(volume) < 3:
        print('READER_ACQUISITION_RECOVERY ' + json.dumps({
            'path': str(path), 'unreadable_files': int(info.get('n_bad', 0)),
            'usable_slices': len(volume),
            'action': 'use_decoded_slices' if len(volume) >= 3 else 'skip_short_acquisition',
        }), flush=True)
    return volume, info
