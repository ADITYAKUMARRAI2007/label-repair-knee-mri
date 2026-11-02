"""Fixed three-plane acquisition selection for a prospective MARS diagnostic."""
from pathlib import Path

import numpy as np

from mars_volume import read_series
from orthodiffusion_volume import PLANES, flag, inspect_series, ranked_series


def build_study(study_id, records, series_root):
    """Prefer fluid-sensitive/fat-suppressed acquisitions, then stable series UID.

    Missing planes have a false mask. Short single-slice acquisitions are
    recorded and skipped; geometry and pixel failures are never hidden.
    """
    rows = list(records)
    if any(str(r['StudyInstanceUID']) != str(study_id) for r in rows):
        raise ValueError('Records contain another study')
    ids = [r['SeriesInstanceUID'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate series metadata')
    volumes = np.zeros((3, 1, 96, 96, 96), dtype=np.float32)
    present = np.zeros(3, dtype=bool)
    receipts = []
    for index, plane in enumerate(PLANES):
        skipped = []
        chosen = None
        for row in ranked_series(rows, plane):
            series_id = row['SeriesInstanceUID']
            directory = Path(series_root) / study_id / series_id
            # Header inspection avoids trying a 3D adapter on a single slice.
            files, _ = inspect_series(directory, study_id, series_id)
            if len(files) < 2:
                skipped.append({'series_id': series_id, 'reason': 'fewer_than_two_native_slices'})
                continue
            volume, receipt = read_series(directory, study_id, series_id)
            volumes[index, 0] = volume
            present[index] = True
            chosen = {'plane': plane, 'series_id': series_id, 'state': 'decoded',
                      'fluid_sensitive': flag(row.get('Fluid_Sensitive')),
                      'fat_suppression': flag(row.get('Fat_Suppression')),
                      'volume_receipt': receipt, 'skipped_single_slice_series': skipped}
            break
        receipts.append(chosen or {'plane': plane, 'state': 'missing_acquisition',
                                   'skipped_single_slice_series': skipped})
    if not present.any():
        raise ValueError('Study has no eligible 3D acquisition')
    return volumes, present, {'study_id': study_id, 'planes': receipts,
                             'selection': 'one per plane; prefer fluid-sensitive then fat-suppressed then series UID',
                             'missing_mask': (~present).tolist(), 'label_values_used': False,
                             'fallback_predictions_used': False}
