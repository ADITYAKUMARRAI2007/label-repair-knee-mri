"""Prediction helpers assembled with the exact executed E04 model definitions.

Final-fold receipts must be pinned by the builder after complete OOF auditing.
This module neither chooses checkpoints nor submits to the competition.
"""
from concurrent.futures import ThreadPoolExecutor


def load_verified_fold(receipt, device):
    path = find_unique(receipt["filename"], receipt["sha256"])
    if path.stat().st_size != receipt["bytes"]:
        raise RuntimeError("final EMA checkpoint size mismatch")
    # Only our explicitly pinned, checksum-verified final training outputs enter
    # this loader. A pilot or an unverified external pickle is not accepted.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("schema") != "rsna_knee_e04_convnext_expert_fold_v1"
            or saved.get("completed_epoch") != 14 or list(saved.get("targets", [])) != TARGETS
            or saved.get("contract") != receipt["contract"]
            or saved.get("contract_sha256") != receipt["contract_sha256"]):
        raise RuntimeError("final fold checkpoint recipe/identity mismatch")
    contract = saved["contract"]
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (contract_sha != receipt["contract_sha256"] or contract["fold"] != receipt["fold"]
            or contract["checkpoint_rule"] != "final_epoch_14_ema_only"
            or contract["expert_checkpoint_selection"] or contract["input_span"] != [.06, .94]
            or contract["physical_crop_mm"] != 140.0 or contract["network_size"] != 288
            or contract["eval_windows_per_slot"] != 8):
        raise RuntimeError("inference input or final-checkpoint contract changed")
    runtime = saved["runtime"]
    if (str(runtime["torch"]) != str(torch.__version__) or runtime["timm"] != timm.__version__
            or runtime["cuda"] != torch.version.cuda
            or runtime["capability"] != list(torch.cuda.get_device_capability(device))):
        raise RuntimeError("checkpoint runtime changed; explicit replay is required before using it")
    generic = find_unique("model.safetensors", PRETRAINED_SHA256)
    model = SeriesMIL(generic)
    model.load_state_dict(saved["state"], strict=True)
    model.eval().requires_grad_(False).to(device)
    del saved
    return model


@torch.no_grad()
def predict_fold_on_device(model, x, mask, pos, device):
    if (tuple(x.shape[1:]) != (5, 8, 3, 336, 336) or x.dtype != torch.uint8
            or mask.shape != (len(x), 5) or mask.dtype != torch.bool
            or pos.shape != (len(x), 5, 8) or pos.dtype != torch.float32
            or len(x) not in (1, 2)):
        raise RuntimeError("inference batch differs from the frozen evaluation contract")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(x.to(device), mask.to(device), pos.to(device), training=False)
    probabilities = logits.float().sigmoid().cpu().numpy()
    if probabilities.shape != (len(x), len(TARGETS)) or not np.isfinite(probabilities).all():
        raise RuntimeError("nonfinite or malformed final-fold prediction")
    return probabilities


class EqualFoldPredictor:
    """Two device workers, fixed equal probability averaging of all five folds."""
    def __init__(self, receipts):
        if sorted(r["fold"] for r in receipts) != list(range(5)):
            raise RuntimeError("hidden inference requires all five independently trained final models")
        if not torch.cuda.is_available():
            raise RuntimeError("E04 hidden inference requires a supported GPU")
        self.groups = {}
        count = min(torch.cuda.device_count(), 2)
        for receipt in sorted(receipts, key=lambda r: r["fold"]):
            device = torch.device(f"cuda:{receipt['fold'] % count}")
            model = load_verified_fold(receipt, device)
            self.groups.setdefault(device, []).append((receipt["fold"], model))
        self.pool = ThreadPoolExecutor(max_workers=len(self.groups))

    @torch.no_grad()
    def _predict_device(self, device, models, x, mask, pos):
        # One worker owns each device's model sequence. No DataParallel or
        # concurrent calls to the same model are used.
        with torch.cuda.device(device):
            gx, gm, gp = x.to(device), mask.to(device), pos.to(device)
            results = [(fold, predict_fold_on_device(model, gx, gm, gp, device)) for fold, model in models]
            torch.cuda.synchronize(device)
            return results

    def predict(self, x, mask, pos):
        futures = [self.pool.submit(self._predict_device, device, models, x, mask, pos)
                   for device, models in self.groups.items()]
        by_fold = {fold: values for future in futures for fold, values in future.result()}
        if sorted(by_fold) != list(range(5)):
            raise RuntimeError("one or more fold predictions missing")
        raw = np.stack([by_fold[f] for f in range(5)])
        result = raw.mean(axis=0, dtype=np.float64)
        if not np.isfinite(result).all():
            raise RuntimeError("invalid equal-fold ensemble")
        return result, raw

    def close(self):
        self.pool.shutdown(wait=True)
        self.groups.clear()
        torch.cuda.empty_cache()
