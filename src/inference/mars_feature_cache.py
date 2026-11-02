"""Reusable frozen spatial features; no labels, fitted head, or score claim."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import torch
import monai

from mars_encoder import CHECKPOINT_SHA256, encode_study_sequences, load_encoder
from orthodiffusion_volume import PLANES


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class MarsFeatureCache:
    def __init__(self, directory, checkpoint, device='cpu'):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        if self.device.type not in {'cpu', 'cuda'}:
            raise ValueError('Unverified feature-extraction device')
        self.encoder = load_encoder(checkpoint).to(self.device)
        here = Path(__file__).resolve().parent
        self.contract = {
            'schema': 'rsna_mars_spatial_feature_cache_v1',
            'checkpoint_sha256': CHECKPOINT_SHA256,
            'source_sha256': {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                              for name in ['mars_encoder.py', 'mars_volume.py',
                                           'mars_study.py', 'mars_feature_cache.py',
                                           'mars_legacy_merging.py']},
            'planes': list(PLANES), 'features_shape': [3, 768, 3, 3, 3],
            'dtype': 'float32', 'autocast': False, 'chunk_size': 1,
            'torch_version': torch.__version__, 'monai_version': monai.__version__,
            'patch_merging': 'MONAI1.4 checkpoint offsets', 'device_type': self.device.type,
            'gpu_name': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None,
            'labels_used': False, 'global_pooling_applied': False,
            'author_knee_preprocessing_parity': False,
        }
        self.contract_sha = hashlib.sha256(json.dumps(
            self.contract, sort_keys=True).encode()).hexdigest()

    def path(self, study_id):
        if not isinstance(study_id, str) or len(study_id) > 64 or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', study_id):
            raise ValueError('Invalid DICOM study UID')
        return self.directory / (study_id + '.npz')

    def read(self, study_id, expected_input_sha=None):
        with np.load(self.path(study_id), allow_pickle=False) as archive:
            if set(archive.files) != {'features', 'present', 'record'}:
                raise ValueError('Unexpected feature-cache fields')
            features, present = archive['features'], archive['present']
            record = json.loads(str(archive['record'].item()))
        if record['study_id'] != study_id or record['contract_sha256'] != self.contract_sha:
            raise ValueError('Feature-cache study or extraction contract differs')
        if (features.shape != (3, 768, 3, 3, 3) or features.dtype != np.float32
                or present.shape != (3,) or present.dtype != np.bool_
                or not present.any() or not np.isfinite(features).all()
                or np.any(features[~present] != 0)
                or digest(features) != record['feature_sha256']
                or digest(present) != record['present_sha256']):
            raise ValueError('Feature-cache arrays failed integrity checks')
        if expected_input_sha is not None and record['input_sha256'] != expected_input_sha:
            raise ValueError('Feature-cache input identity differs')
        return features, present, record

    def extract(self, study_id, volumes, present, input_receipt):
        volumes, present = np.asarray(volumes), np.asarray(present)
        if (volumes.shape != (3, 1, 96, 96, 96) or volumes.dtype != np.float32
                or present.shape != (3,) or present.dtype != np.bool_
                or not present.any() or not np.isfinite(volumes[present]).all()
                or np.any(volumes[~present] != 0)):
            raise ValueError('Invalid canonical study tensors or missing planes')
        if (input_receipt['study_id'] != study_id or input_receipt['label_values_used']
                or input_receipt['fallback_predictions_used']
                or input_receipt['missing_mask'] != (~present).tolist()
                or [p['plane'] for p in input_receipt['planes']] != list(PLANES)):
            raise ValueError('Input receipt differs from the fixed acquisition contract')
        for i, plane in enumerate(input_receipt['planes']):
            if present[i] and plane['volume_receipt']['volume_sha256'] != digest(volumes[i, 0]):
                raise ValueError('Canonical volume differs from its decode receipt')
        identity = {'volumes_sha256': digest(volumes), 'present_sha256': digest(present),
                    'input_receipt': input_receipt}
        input_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.path(study_id)
        if path.exists():
            return (*self.read(study_id, input_sha), True)
        x = torch.from_numpy(volumes[None]).to(self.device)
        mask = torch.from_numpy(present[None]).to(self.device)
        with torch.autocast(self.device.type, enabled=False):
            features = encode_study_sequences(self.encoder, x, mask, chunk_size=1)[0].float().cpu().numpy()
        record = {'study_id': study_id, 'contract_sha256': self.contract_sha,
                  'extraction_contract': self.contract, 'input_sha256': input_sha,
                  'input_identity': identity, 'feature_sha256': digest(features),
                  'present_sha256': digest(present)}
        # Hard-link installation is exclusive and atomic. An interrupted write
        # cannot replace an existing verified study record.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix='.npz', delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(handle, features=features, present=present,
                         record=np.array(json.dumps(record, sort_keys=True)))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                return (*self.read(study_id, input_sha), True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return (*self.read(study_id, input_sha), False)
