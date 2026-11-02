"""RSNA-specific MARS volume adaptation; not author knee-preprocessing parity.

Decode the full acquisition, use physical DICOM geometry, and resample into a
96-cube with isotropic spacing covering the complete physical bounding box.
Output tensor axes are superior, anterior, right (D,H,W). No label is used.
"""
import hashlib
import itertools

import numpy as np
import torch
import torch.nn.functional as F

from orthodiffusion_volume import inspect_series

SIZE = 96


def canonical_cube(volume, origin_lps, affine_dhw_lps, photometric='MONOCHROME2'):
    """Resample a regularly spaced DHW acquisition into canonical patient axes."""
    volume = np.asarray(volume, dtype=np.float32)
    origin = np.asarray(origin_lps, dtype=np.float64)
    affine = np.asarray(affine_dhw_lps, dtype=np.float64)
    if (volume.ndim != 3 or min(volume.shape) < 2 or not np.isfinite(volume).all()
            or origin.shape != (3,) or affine.shape != (3, 3)
            or not np.isfinite(origin).all() or not np.isfinite(affine).all()):
        raise ValueError('Invalid volume or physical geometry')
    if abs(np.linalg.det(affine)) < 1e-8:
        raise ValueError('Singular acquisition geometry')
    if photometric not in {'MONOCHROME1', 'MONOCHROME2'}:
        raise ValueError('Unsupported photometric interpretation')
    low, high = float(volume.min()), float(volume.max())
    if high <= low:
        raise ValueError('Constant MRI acquisition')
    display = high - volume if photometric == 'MONOCHROME1' else volume - low
    nonzero = display > 0
    mean, std = float(display[nonzero].mean()), float(display[nonzero].std())
    if not np.isfinite(std) or std <= 1e-8:
        raise ValueError('Degenerate nonzero intensity distribution')
    normalized = np.zeros_like(display)
    normalized[nonzero] = (display[nonzero] - mean) / std

    # Bounds of voxel edges, not only centers: retain the complete acquired FOV.
    corners = np.array(list(itertools.product(*[(-.5, n-.5) for n in volume.shape])))
    physical = corners @ affine.T + origin
    lower, upper = physical.min(0), physical.max(0)
    center = (lower + upper) / 2
    pitch = float((upper - lower).max() / SIZE)
    index = torch.arange(SIZE, dtype=torch.float64) - (SIZE-1)/2
    zz, yy, xx = torch.meshgrid(index, index, index, indexing='ij')
    # DICOM LPS: x increases left, y posterior, z superior.
    target = torch.stack((-xx, -yy, zz), dim=-1) * pitch + torch.from_numpy(center)
    source = (target - torch.from_numpy(origin)) @ torch.from_numpy(np.linalg.inv(affine)).T
    shape = torch.tensor(volume.shape, dtype=torch.float64)
    normalized_grid = 2 * (source + .5) / shape - 1
    grid = normalized_grid[..., [2, 1, 0]].float().unsqueeze(0)
    data = torch.from_numpy(np.ascontiguousarray(normalized))[None, None]
    result = F.grid_sample(data, grid, mode='bilinear', padding_mode='zeros',
                           align_corners=False)[0, 0].numpy()
    if result.shape != (SIZE, SIZE, SIZE) or not np.isfinite(result).all():
        raise ValueError('Invalid canonical volume')
    receipt = {'schema': 'rsna_mars_full_volume_v1', 'native_shape': list(volume.shape),
               'output_shape': list(result.shape), 'origin_lps_mm': origin.tolist(),
               'affine_dhw_to_lps_mm': affine.tolist(),
               'native_bounds_lps_mm': [lower.tolist(), upper.tolist()],
               'output_center_lps_mm': center.tolist(), 'isotropic_spacing_mm': pitch,
               'output_axes_dhw': ['superior', 'anterior', 'right'],
               'normalization': 'photometric reversal if needed; global min shift; nonzero population zscore before resampling',
               'nonzero_mean': mean, 'nonzero_std': std, 'photometric': photometric,
               'interpolation': 'trilinear; align_corners=False; zero outside acquisition',
               'full_physical_bounds_covered': True, 'fixed_physical_crop': False,
               'author_knee_preprocessing_parity': False,
               'volume_sha256': hashlib.sha256(result.tobytes()).hexdigest()}
    return np.ascontiguousarray(result), receipt


def read_series(directory, expected_study=None, expected_series=None):
    import pydicom
    from pydicom.pixels import apply_modality_lut
    files, geometry = inspect_series(directory, expected_study, expected_series)
    if len(files) < 2:
        raise ValueError('At least two native slices required for 3D geometry')
    first = pydicom.dcmread(files[0], stop_before_pixels=True)
    if str(getattr(first, 'AnatomicalOrientationType', 'BIPED')) != 'BIPED':
        raise ValueError('Only BIPED patient coordinates are supported')
    origin = np.asarray(first.ImagePositionPatient, dtype=float)
    iop = np.asarray(geometry['orientation'])
    normal = np.asarray(geometry['slice_normal'])
    dz = geometry['median_slice_spacing_mm']
    row_spacing, column_spacing = geometry['spacing_mm']
    affine = np.stack((normal * dz, iop[3:] * row_spacing, iop[:3] * column_spacing), axis=1)
    arrays = []
    for index, path in enumerate(files):
        ds = pydicom.dcmread(path)
        position = np.asarray(ds.ImagePositionPatient, dtype=float)
        # Reject sheared/misregistered stacks rather than silently losing geometry.
        expected = origin + affine[:, 0] * index
        if not np.allclose(position, expected, rtol=0, atol=.1):
            raise ValueError('Nonregular slice origins require another physical-space adapter')
        pixels = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
        if pixels.shape != tuple(geometry['native_shape']) or not np.isfinite(pixels).all():
            raise ValueError('Invalid decoded MRI pixels')
        arrays.append(pixels)
    result, receipt = canonical_cube(np.stack(arrays), origin, affine, geometry['photometric'])
    receipt.update(study_id=str(first.StudyInstanceUID), series_id=str(first.SeriesInstanceUID),
                   selected_files=[p.name for p in files], decoded_slices=len(files),
                   geometry_source='DICOM ImagePositionPatient/ImageOrientationPatient/PixelSpacing',
                   source_geometry=geometry)
    return result, receipt
