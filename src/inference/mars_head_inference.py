"""Load verified final heads; keep held-out and test-cohort inference explicit."""
import hashlib
import json
from pathlib import Path

import torch

import mars_spatial_head
from mars_spatial_head import MarsSpatialHead


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MarsHeadInference:
    def __init__(self, directory, protocol_path, extraction_contract_sha256, device):
        directory, protocol_path = Path(directory), Path(protocol_path)
        protocol = json.loads(protocol_path.read_text())
        protocol_sha = sha(protocol_path)
        assert protocol['schema'] == 'mars_frozen_spatial_classifier_v1'
        assert protocol['folds'] == list(range(5)) and protocol['epochs'] == 30
        assert sha(mars_spatial_head.__file__) == protocol['sources_sha256']['training/mars_spatial_head.py']
        receipt = json.loads((directory / 'training_receipt.json').read_text())
        assert receipt['status'] == 'FIVE_FROZEN_MARS_HEADS_AND_COMPLETE_OOF_TRAINED'
        assert receipt['studies'] == 4407 and receipt['protocol_sha256'] == protocol_sha
        assert receipt['extraction_contract_sha256'] == extraction_contract_sha256
        folds = json.loads((directory / 'fold_receipts.json').read_text())
        assert [f['fold'] for f in folds] == list(range(5))
        self.device = torch.device(device)
        assert self.device.type in {'cpu', 'cuda'}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.heads, self.validation_ids = [], []
        self.training_ids = set()
        self.target_order = protocol['target_order']
        for record in folds:
            fold = record['fold']; path = directory / f'mars_head_fold{fold}.pt'
            assert record['completed_epoch'] == 30
            assert sha(path) == record['checkpoint_sha256']
            state = torch.load(path, map_location='cpu', weights_only=True)
            assert state['fold'] == fold and state['completed_epoch'] == 30
            assert state['protocol_sha256'] == protocol_sha
            assert state['extraction_contract_sha256'] == extraction_contract_sha256
            assert all(torch.isfinite(value).all() for value in state['model'].values())
            model = MarsSpatialHead()
            model.load_state_dict(state['model'], strict=True)
            model.to(self.device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.heads.append(model)
            self.validation_ids.append(set(record['validation_ids']))
            self.training_ids.update(record['training_ids'])
        assert len(self.training_ids) == 4407

    def inputs(self, features, present, study_ids):
        assert features.dtype == torch.float32 and present.dtype == torch.bool
        assert features.shape == (len(study_ids), 3, 768, 3, 3, 3)
        assert present.shape == (len(study_ids), 3)
        assert len(study_ids) > 0 and len(study_ids) == len(set(study_ids))
        assert torch.isfinite(features).all() and present.any(dim=1).all()
        assert (features[~present] == 0).all()
        return features.to(self.device), present.to(self.device)

    @torch.inference_mode()
    def predict_heldout(self, features, present, study_ids, fold):
        """Replay only each study's excluded head; never average on training truth."""
        assert fold in range(5)
        assert set(study_ids).issubset(self.validation_ids[fold])
        features, present = self.inputs(features, present, study_ids)
        result = self.heads[fold](features, present).sigmoid()
        assert torch.isfinite(result).all()
        return result.cpu().numpy()

    @torch.inference_mode()
    def predict_test(self, features, present, study_ids):
        """Use the fixed arithmetic mean of five probabilities on unseen studies."""
        assert not (set(study_ids) & self.training_ids), 'Training studies require held-out inference'
        features, present = self.inputs(features, present, study_ids)
        result = torch.stack([head(features, present).sigmoid() for head in self.heads]).mean(dim=0)
        assert torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
        return result.cpu().numpy()
