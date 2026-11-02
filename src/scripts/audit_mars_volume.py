"""CPU geometry/photometric checks and one existing real-series decode."""
import json
from pathlib import Path
import signal
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/foundation_refresh_20261010/MARS'
sys.path[:0] = [str(RUN / 'deps'), str(ROOT / 'runs/orthodiffusion_volume_v1/test_dependencies'),
               str(ROOT / 'inference')]

import numpy as np
import torch
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from mars_volume import canonical_cube, read_series


def fail(*_):
    raise TimeoutError('CPU geometry audit time limit')


def main():
    signal.signal(signal.SIGALRM, fail); signal.alarm(120)
    torch.set_num_threads(2)
    z, y, x = np.indices((8, 12, 16))
    volume = (2*z + 3*y + 5*x + 10).astype(np.float32)
    origin = np.array([7., -12., 30.])
    affine = np.array([[0., 0., .8], [0., .6, 0.], [2.5, 0., 0.]])
    base, receipt = canonical_cube(volume, origin, affine)
    alternate_affine = affine.copy(); alternate_affine[:, 2] *= -1
    flipped, _ = canonical_cube(volume[..., ::-1], origin + affine[:, 2] * 15, alternate_affine)
    swapped, _ = canonical_cube(volume.transpose(2, 0, 1), origin, affine[:, [2, 0, 1]])
    reversed_photo, _ = canonical_cube(2000-volume, origin, affine, 'MONOCHROME1')
    errors = {name: float(np.max(np.abs(base - other))) for name, other in
              [('stored_column_flip', flipped), ('stored_axis_permutation', swapped),
               ('photometric_equivalence', reversed_photo)]}
    assert all(error < 2e-5 for error in errors.values())
    bounds = np.array(receipt['native_bounds_lps_mm'])
    center = np.array(receipt['output_center_lps_mm']); half = receipt['isotropic_spacing_mm']*48
    assert np.all(bounds[0] >= center-half-1e-9) and np.all(bounds[1] <= center+half+1e-9)
    # Verify world-axis mapping against a continuous physical ramp away from borders.
    pitch = receipt['isotropic_spacing_mm']
    di, hi, wi = 48, 48, 48
    point = center + np.array([47.5-wi, 47.5-hi, di-47.5])*pitch
    source = np.linalg.solve(affine, point-origin)
    expected_display = 2*source[0] + 3*source[1] + 5*source[2]
    expected = (expected_display-receipt['nonzero_mean'])/receipt['nonzero_std']
    assert abs(float(base[di, hi, wi])-expected) < 2e-5

    with tempfile.TemporaryDirectory(dir='/private/tmp', prefix='mars_geometry_') as temp:
        folder = Path(temp); study, series = generate_uid(), generate_uid()
        for index in range(8):
            path = folder / f'{7-index:02}.dcm'
            sop = generate_uid(); meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = MRImageStorage; meta.MediaStorageSOPInstanceUID = sop
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
            ds.SOPClassUID = MRImageStorage; ds.SOPInstanceUID = sop
            ds.StudyInstanceUID = study; ds.SeriesInstanceUID = series; ds.Modality = 'MR'
            ds.Rows, ds.Columns = (12, 16); ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'; ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15; ds.PixelRepresentation = 0
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = (origin + affine[:, 0]*index).tolist()
            ds.PixelSpacing = [.6, .8]; ds.InstanceNumber = 800-index
            ds.RescaleSlope = 1.; ds.RescaleIntercept = 0.
            ds.PixelData = volume[index].astype('<u2').tobytes()
            ds.save_as(path, enforce_file_format=True)
        decoded, decoded_receipt = read_series(folder, study, series)
        assert np.max(np.abs(decoded-base)) < 2e-5
        assert decoded_receipt['selected_files'] == [f'{7-i:02}.dcm' for i in range(8)]
        ds = pydicom.dcmread(folder/'00.dcm'); ds.ImagePositionPatient[0] += 2.
        ds.save_as(folder/'00.dcm', enforce_file_format=True)
        try:
            read_series(folder, study, series)
        except ValueError as error:
            assert 'Nonregular slice origins' in str(error)
        else:
            raise AssertionError('Unmodeled slice shear accepted')

    raw_root = ROOT / 'runs/orthodiffusion_volume_v2/raw_failed_study/train_series'
    real_series = sorted({p.parent for p in raw_root.rglob('*.dcm')})[0]
    real, real_receipt = read_series(real_series, real_series.parent.name, real_series.name)
    assert real.shape == (96, 96, 96) and np.isfinite(real).all()
    (RUN / 'real_series_input_receipt.json').write_text(json.dumps(real_receipt, indent=2)+'\n')
    np.savez_compressed(RUN / 'real_series_cpu_fixture.npz', volume=real)
    result = {'status': 'PASS_RSNA_PHYSICAL_VOLUME_ADAPTER_CPU',
              'equivalent_encoding_max_abs_errors': errors,
              'analytic_world_coordinate_ramp_passed': True, 'full_bounds_covered': True,
              'filename_and_instance_order_ignored': True, 'unmodeled_slice_shear_rejected': True,
              'real_series_decoded_slices': real_receipt['decoded_slices'],
              'real_series_isotropic_spacing_mm': real_receipt['isotropic_spacing_mm'],
              'clinical_accuracy_measured': False, 'gpu_used': False,
              'author_knee_preprocessing_parity': False,
              'limitations': ['Only one real acquisition checked; broad decode coverage remains untested.',
                             '96-cube resampling loses spatial detail; clinical transfer requires evaluation.',
                             'Regular BIPED single-frame acquisitions only; no geometry fallback.']}
    (RUN / 'volume_adapter_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result)); signal.alarm(0)


if __name__ == '__main__':
    main()
