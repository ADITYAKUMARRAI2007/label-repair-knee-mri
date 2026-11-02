#!/usr/bin/env python3
"""E03 full trainer template; builder embeds the pilot-tested core and split."""
from __future__ import annotations
# __CORE__
# __SPLIT__
from sklearn.metrics import roc_auc_score

EXPECTED_SPLIT_SHA256 = "__EXPECTED_SPLIT_SHA256__"
EPOCHS = 14
SCHEMA = "rsna_knee_e03_resnet34_series_transformer_v1"
TRAINING_CONTRACT = {
    "architecture": "resnet34_series_transformer", "seed": SEED,
    "epochs": EPOCHS, "checkpoint_rule": "final_epoch_ema_only",
    "ema_decay": 0.998, "network_size": IMAGE_SIZE,
    "train_windows_per_slot": TRAIN_PER_SLOT, "eval_windows_per_slot": EVAL_PER_SLOT,
    "batch_studies": BATCH_STUDIES, "slot_ranges": SLOT_RANGES,
    "backbone_lr": 1e-4, "head_lr": 5e-4, "weight_decay": 0.025,
    "schedule": "one_epoch_linear_warmup_then_cosine_to_1_percent",
    "teacher_logit_weight": 0.75, "jev_logit_weight": 0.25,
    "pretrained_and_label_sha256": EXPECTED,
    "split_sha256": EXPECTED_SPLIT_SHA256,
    "gold_rows_used_for_training": 0, "gold_checkpoint_selection": False,
}
CONTRACT_SHA256 = hashlib.sha256(json.dumps(TRAINING_CONTRACT, sort_keys=True).encode()).hexdigest()


def auc_metrics(truth, pred):
    if truth.shape != pred.shape or not np.isfinite(pred).all():
        raise RuntimeError("invalid validation prediction matrix")
    if not ((pred >= 0) & (pred <= 1)).all():
        raise RuntimeError("predictions outside probability range")
    each = {}
    for j, target in enumerate(TARGETS):
        if len(np.unique(truth[:, j])) != 2:
            raise RuntimeError(f"validation target {target} has only one class")
        each[target] = float(roc_auc_score(truth[:, j], pred[:, j]))
    return float(np.mean(list(each.values()))), each


@torch.no_grad()
def predict(model, loader):
    model.eval()
    predictions, rows = [], []
    for x, mask, pos, _, row in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x.cuda(), mask.cuda(), pos.cuda())
        p = logits.float().sigmoid().cpu().numpy()
        if not np.isfinite(p).all():
            raise RuntimeError("nonfinite validation prediction")
        predictions.append(p)
        rows.extend(row.tolist())
    loader.dataset.release_pages()
    return np.concatenate(predictions), np.asarray(rows)


def write_predictions(path, ids, rows, pred):
    frame = pd.DataFrame(pred, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", [ids[r] for r in rows])
    if frame.StudyInstanceUID.duplicated().any():
        raise RuntimeError("duplicate validation study")
    frame.to_csv(path, index=False)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("E03 requires a GPU")
    seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    working = Path("/kaggle/working")
    root = find_competition_root()
    # dtype=str preserves the exact report and expert-label interpretation used
    # by the independently frozen local split contract.
    raw = pd.read_csv(root / "train.csv", dtype=str, keep_default_na=False)
    split = freeze_split(raw.to_dict("records"), TARGETS)
    if split_sha256(split) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("frozen split drift")
    (working / "e03_split.json").write_text(json.dumps(split, indent=2) + "\n")
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    corpus, masks, ids = open_corpus()
    corpus.release_pages()
    del corpus
    if set(ids) != set(train.StudyInstanceUID.astype(str)):
        raise RuntimeError("corpus/train identity mismatch")
    for name in ("teacher_soft_labels.csv", "HYBRID_labels_scores.csv"):
        find_unique(name, EXPECTED[name])
    targets, labels = build_targets(train, ids)
    row_by_uid = {uid: i for i, uid in enumerate(ids)}
    row_sets = {name: np.asarray([row_by_uid[uid] for uid in split[name]], np.int64)
                for name in ("train", "weak_validation", "gold")}
    truth = np.zeros((len(ids), len(TARGETS)), np.float32)
    by_uid = train.set_index("StudyInstanceUID")
    for uid in split["gold"]:
        truth[row_by_uid[uid]] = by_uid.loc[uid, TARGETS].to_numpy(np.float32)
    train_set = SeriesWindows(row_sets["train"], targets, masks, TRAIN_PER_SLOT, True)
    train_loader = DataLoader(train_set, batch_size=BATCH_STUDIES, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    eval_loaders = {name: DataLoader(SeriesWindows(rows, targets, masks, EVAL_PER_SLOT, False),
                                    batch_size=BATCH_STUDIES, shuffle=False,
                                    num_workers=0, pin_memory=False)
                    for name, rows in row_sets.items() if name != "train"}
    model = SeriesMIL(find_unique("resnet34.pth", EXPECTED["resnet34.pth"])).cuda()
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": TRAINING_CONTRACT["backbone_lr"]},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
         "lr": TRAINING_CONTRACT["head_lr"]}], weight_decay=TRAINING_CONTRACT["weight_decay"])
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader)

    def schedule(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    state_path = working / "e03_last.pt"
    start_epoch, step, history, overflows = 1, 0, [], 0
    # Future resumes attach this trainer's completed-epoch state as an input.
    # A pilot checkpoint or a changed recipe fails this contract.
    resume_hits = bounded_files("e03_last.pt")
    if resume_hits:
        if len(resume_hits) != 1:
            raise RuntimeError("ambiguous E03 resume state")
        saved = torch.load(resume_hits[0], map_location="cuda", weights_only=False)
        if saved.get("contract_sha256") != CONTRACT_SHA256 or saved.get("pilot_only", False):
            raise RuntimeError("resume contract mismatch")
        model.load_state_dict(saved["model"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch, step = saved["completed_epoch"] + 1, saved["step"]
        history, overflows = saved["history"], saved["amp_skipped_updates"]
        del saved
        # Preserve the state artifact even when resuming a fully finished run.
        import shutil
        shutil.copyfile(resume_hits[0], state_path)
        log(f"resuming after epoch {start_epoch - 1}")
    log(json.dumps({"contract": TRAINING_CONTRACT, "counts": {k: len(v) for k, v in row_sets.items()}}, sort_keys=True))
    for epoch in range(start_epoch, EPOCHS + 1):
        # Per-epoch reseeding makes a completed-epoch resume repeatable.
        seed_all(SEED + epoch)
        model.train()
        epoch_start, loss_sum, count = time.time(), 0.0, 0
        for batch, (x, mask, pos, y, _) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x.cuda(), mask.cuda(), pos.cuda(), training=True)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y.cuda())
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite training forward")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            overflow = not bool(torch.isfinite(norm))
            overflows += int(overflow)
            if overflows > 10:
                raise RuntimeError("persistent gradient overflow")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if not overflow:
                with torch.no_grad():
                    decay = min(0.998, (1 + step) / (10 + step))
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - decay)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            loss_sum += float(loss.detach())
            count += 1
            if batch % 100 == 0:
                rss = current_rss_gib()
                gpu = torch.cuda.max_memory_allocated() / (1024 ** 3)
                log(json.dumps({"epoch": epoch, "batch": batch, "batches": len(train_loader),
                                "loss": loss_sum / count, "rss_gib": rss,
                                "gpu_peak_gib": gpu, "epoch_seconds": time.time() - epoch_start}))
                if rss > 20 or gpu > 13:
                    raise RuntimeError("training resource gate exceeded")
        train_set.release_pages()
        weak_pred, weak_rows = predict(ema, eval_loaders["weak_validation"])
        weak_auc, weak_each = auc_metrics((targets[weak_rows] >= 0.5).astype(np.uint8), weak_pred)
        record = {"epoch": epoch, "loss": loss_sum / count, "weak_validation_auc": weak_auc,
                  "weak_validation_per_target_auc": weak_each, "seconds": time.time() - epoch_start,
                  "rss_gib": current_rss_gib()}
        history.append(record)
        atomic_save({"schema": SCHEMA, "contract_sha256": CONTRACT_SHA256,
                     "model": model.state_dict(), "ema": ema.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "completed_epoch": epoch, "step": step,
                     "history": history, "amp_skipped_updates": overflows,
                     "training_contract": TRAINING_CONTRACT, "pilot_only": False}, state_path)
        (working / "e03_history.json").write_text(json.dumps(history, indent=2) + "\n")
        log(json.dumps(record, sort_keys=True))
    # Gold is measured only on the fixed final EMA, never for checkpoint choice.
    gold_pred, gold_rows = predict(ema, eval_loaders["gold"])
    gold_auc, gold_each = auc_metrics(truth[gold_rows], gold_pred)
    weak_pred, weak_rows = predict(ema, eval_loaders["weak_validation"])
    weak_auc, weak_each = auc_metrics((targets[weak_rows] >= 0.5).astype(np.uint8), weak_pred)
    checkpoint = working / "e03_resnet34_series_ema.pt"
    atomic_save({"schema": SCHEMA, "state": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
                 "targets": TARGETS, "contract": TRAINING_CONTRACT,
                 "contract_sha256": CONTRACT_SHA256, "source_shape": [44, 336, 336]}, checkpoint)
    gold_path, weak_path = working / "e03_gold_predictions.csv", working / "e03_weak_predictions.csv"
    write_predictions(gold_path, ids, gold_rows, gold_pred)
    write_predictions(weak_path, ids, weak_rows, weak_pred)
    artifacts = [checkpoint, gold_path, weak_path, working / "e03_split.json", state_path]
    manifest = {"schema": SCHEMA, "status": "COMPLETE", "contract": TRAINING_CONTRACT,
                "contract_sha256": CONTRACT_SHA256, "strict_gold_holdout": True,
                "gold_report_groups_excluded_from_training": True,
                "split_counts": {n: len(split[n]) for n in ("train", "weak_validation", "gold", "excluded_gold_group_weak")},
                "validation": {"gold_auc": gold_auc, "gold_per_target_auc": gold_each,
                               "weak_auc_against_thresholded_soft_labels": weak_auc,
                               "weak_per_target_auc": weak_each},
                "history": history, "label_audit": labels, "amp_skipped_updates": overflows,
                "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in artifacts},
                "elapsed_seconds_this_run": time.time() - T0,
                "limitations": "report grouping does not prove patient/scanner independence; no official score yet"}
    (working / "e03_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(json.dumps({"status": "COMPLETE", "gold_auc": gold_auc, "weak_auc": weak_auc}))


if __name__ == "__main__":
    main()
