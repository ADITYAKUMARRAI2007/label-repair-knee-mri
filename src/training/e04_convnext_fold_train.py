"""Full fold trainer. Frozen final epoch-14 EMA; completed-epoch resumes only."""
from sklearn.metrics import roc_auc_score

FOLD = int(os.environ["E04_TRAIN_FOLD"])
EPOCHS = 14
SCHEMA = "rsna_knee_e04_convnext_expert_fold_v1"
TRAINING_CONTRACT = {
    "schema": SCHEMA, "fold": FOLD, "architecture": "convnext_small_series_transformer",
    "worker_component_sha256": WORKER_COMPONENT_SHA256,
    "seed": SEED + 1000 * (FOLD + 1), "epochs": EPOCHS,
    "checkpoint_rule": "final_epoch_14_ema_only", "expert_checkpoint_selection": False,
    "pretrained_sha256": PRETRAINED_SHA256, "fold_sha256": E04_FOLD_SHA256,
    "target_sha256": E04_TARGET_SHA256, "weight_sha256": E04_WEIGHT_SHA256,
    "wide_cache": EXPECTED_WIDE_CACHE, "input_span": [.06, .94], "physical_crop_mm": 140.0,
    "network_size": IMAGE_SIZE, "train_windows_per_slot": TRAIN_PER_SLOT,
    "eval_windows_per_slot": EVAL_PER_SLOT, "batch_studies": BATCH_STUDIES,
    "slot_ranges": SLOT_RANGES, "backbone_lr": 3e-5, "head_lr": 3e-4,
    "weight_decay": .025, "ema_decay": .998, "gradient_checkpointing": True,
    "schedule": "one_epoch_linear_warmup_then_cosine_to_one_percent",
    "expert_weight": 2.0, "weak_fusion": "fixed_four_report_family_median",
    "weak_confidence": "clip_one_minus_source_range_0.25_to_1",
    "heldout_report_groups_excluded": True, "independent_backbone_per_fold": True,
    "validation_rule": "one_final_prediction_pass_after_all_14_epochs",
}
CONTRACT_SHA256 = hashlib.sha256(json.dumps(TRAINING_CONTRACT, sort_keys=True).encode()).hexdigest()


def runtime_versions():
    return {"torch": torch.__version__, "timm": timm.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "capability": list(torch.cuda.get_device_capability())}


@torch.no_grad()
def predict_final(model, loader):
    model.eval()
    predictions, rows = [], []
    for x, mask, pos, _, row in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x.cuda(), mask.cuda(), pos.cuda())
        values = logits.float().sigmoid().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("nonfinite final OOF probabilities")
        predictions.append(values)
        rows.extend(row.tolist())
    loader.dataset.release_pages()
    return np.concatenate(predictions), np.asarray(rows, np.int64)


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or FOLD not in range(5):
        raise RuntimeError("a full fold requires exactly one assigned GPU")
    torch.set_num_threads(2)
    seed_all(TRAINING_CONTRACT["seed"])
    torch.backends.cudnn.benchmark = True
    out = Path(os.environ["E04_TRAIN_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ["E04_PARENT_DEADLINE"])
    prefix = f"e04_fold{FOLD}"
    root = find_competition_root()
    corpus, masks, ids = open_corpus()
    corpus.release_pages()
    del corpus
    targets, weights, gold, folds, groups, split = reconstruct_frozen_inputs(root / "train.csv", ids)
    training_rows, validation_rows, expert_rows = fold_row_sets(ids, gold, folds, groups, FOLD)
    train_set = SeriesWindows(training_rows, targets, masks, TRAIN_PER_SLOT, True)
    train_loader = DataLoader(train_set, batch_size=BATCH_STUDIES, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=False)
    model = SeriesMIL(find_unique("model.safetensors", PRETRAINED_SHA256)).cuda()
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": TRAINING_CONTRACT["backbone_lr"]},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
         "lr": TRAINING_CONTRACT["head_lr"]}], weight_decay=TRAINING_CONTRACT["weight_decay"])
    total_steps, warmup_steps = len(train_loader) * EPOCHS, len(train_loader)

    def schedule(step):
        if step < warmup_steps:
            return .1 + .9 * step / max(1, warmup_steps)
        progress = min(1., (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    state_path = out / f"{prefix}_last.pt"
    versions = runtime_versions()
    start_epoch, step, overflows, history = 1, 0, 0, []
    checkpoint_overflows = 0
    resumes = bounded_files(f"{prefix}_last.pt")
    if resumes:
        if len(resumes) != 1:
            raise RuntimeError("ambiguous completed-epoch resume checkpoint")
        saved = torch.load(resumes[0], map_location="cuda", weights_only=False)
        if (saved.get("schema") != SCHEMA or saved.get("contract_sha256") != CONTRACT_SHA256
                or saved.get("pilot_only", True) or saved.get("runtime") != versions):
            raise RuntimeError("resume recipe or runtime contract changed")
        if saved["completed_epoch"] not in range(0, EPOCHS + 1):
            raise RuntimeError("invalid completed-epoch resume point")
        model.load_state_dict(saved["model"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch, step = saved["completed_epoch"] + 1, saved["step"]
        history, overflows = saved["history"], saved["amp_skipped_updates"]
        checkpoint_overflows = overflows
        if [h["epoch"] for h in history] != list(range(1, start_epoch)) or step != (start_epoch - 1) * len(train_loader):
            raise RuntimeError("resume history/step coverage drift")
        del saved
        import shutil
        shutil.copyfile(resumes[0], state_path)

    def save_completed_epoch(completed_epoch):
        nonlocal checkpoint_overflows
        atomic_save({"schema": SCHEMA, "contract_sha256": CONTRACT_SHA256, "contract": TRAINING_CONTRACT,
                     "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "completed_epoch": completed_epoch, "step": step, "history": history,
                     "amp_skipped_updates": overflows, "runtime": versions, "pilot_only": False}, state_path)
        checkpoint_overflows = overflows

    if not state_path.exists():
        # Preserve a valid fresh state even if the first epoch reaches its deadline.
        save_completed_epoch(0)

    split_receipt = {"heldout_fold": FOLD, "fold_audit": split,
                     "training_ids": ids[training_rows].tolist(), "validation_ids": ids[validation_rows].tolist(),
                     "expert_validation_ids": ids[expert_rows].tolist(),
                     "expert_training_ids": ids[training_rows[gold[training_rows]]].tolist(),
                     "heldout_report_groups_excluded": True}
    split_path = out / f"{prefix}_split.json"
    split_path.write_text(json.dumps(split_receipt, indent=2) + "\n")
    log(json.dumps({"event": "full_fold_ready", "fold": FOLD, "contract_sha256": CONTRACT_SHA256,
                    "training_rows": len(training_rows), "expert_training_rows": int(gold[training_rows].sum()),
                    "validation_rows": len(validation_rows), "expert_validation_rows": len(expert_rows),
                    "runtime": versions, "next_epoch": start_epoch}))
    budget_stopped = False
    for epoch in range(start_epoch, EPOCHS + 1):
        # Per-epoch reseeding preserves the same sampling/augmentation rule on resume.
        seed_all(TRAINING_CONTRACT["seed"] + epoch)
        recent = [h["epoch_wall_seconds"] for h in history[-2:]]
        needed = (max(recent) * 1.08 if recent else 1800) + (900 if epoch == EPOCHS else 90)
        if time.time() + needed > deadline:
            if not state_path.exists():
                save_completed_epoch(epoch - 1)
            budget_stopped = True
            break
        model.train()
        epoch_started, loss_sum, batches = time.time(), 0.0, 0
        for batch, (x, mask, pos, y, row) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x.cuda(), mask.cuda(), pos.cuda(), training=True)
                confidence = torch.from_numpy(weights[row.numpy()]).cuda()
                loss = (F.binary_cross_entropy_with_logits(logits.float(), y.cuda(), reduction="none") * confidence).mean()
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite full-fold training forward")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 3.)
            overflow = not bool(torch.isfinite(norm))
            overflows += int(overflow)
            if overflows > 10:
                raise RuntimeError("persistent full-fold gradient overflow")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if not overflow:
                with torch.no_grad():
                    decay = min(.998, (1 + step) / (10 + step))
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - decay)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            loss_sum += float(loss.detach())
            batches += 1
            if batch % 100 == 0:
                item = {"fold": FOLD, "epoch": epoch, "batch": batch, "batches": len(train_loader),
                        "loss": loss_sum / batches, "rss_gib": current_rss_gib(),
                        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "epoch_seconds": time.time() - epoch_started}
                log(json.dumps(item))
                if item["rss_gib"] > 18 or item["gpu_peak_gib"] > 13:
                    raise RuntimeError("full-fold resident/GPU memory limit exceeded")
            if time.time() > deadline - 90:
                # Keep the previous completed epoch, never a partially updated state.
                budget_stopped = True
                break
        train_set.release_pages()
        if budget_stopped:
            if not state_path.exists():
                raise RuntimeError("budget expired during the first epoch without a completed checkpoint")
            break
        record = {"epoch": epoch, "loss": loss_sum / batches,
                  "epoch_wall_seconds": time.time() - epoch_started,
                  "rss_gib": current_rss_gib(), "batches": batches, "step": step}
        history.append(record)
        save_started = time.time()
        save_completed_epoch(epoch)
        # This optional wall-time note is not needed to reconstruct optimization.
        log(json.dumps({"event": "epoch_complete", "fold": FOLD, **record,
                        "checkpoint_seconds": time.time() - save_started,
                        "remaining_budget_seconds": deadline - time.time()}))
        (out / f"{prefix}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    complete = not budget_stopped and len(history) == EPOCHS
    artifacts = [state_path, split_path]
    validation = None
    if complete:
        validation_loader = DataLoader(SeriesWindows(validation_rows, targets, masks, EVAL_PER_SLOT, False),
                                        batch_size=BATCH_STUDIES, shuffle=False, num_workers=0, pin_memory=False)
        pred, rows = predict_final(ema, validation_loader)
        if not np.array_equal(rows, validation_rows):
            raise RuntimeError("final prediction identity/order changed")
        frame = pd.DataFrame(pred, columns=TARGETS)
        frame.insert(0, "StudyInstanceUID", ids[rows])
        oof_path = out / f"{prefix}_oof_predictions.csv"
        frame.to_csv(oof_path, index=False)
        expert_mask = gold[rows]
        expert_prediction = pred[expert_mask]
        expert_truth = targets[rows[expert_mask]]
        aucs = {target: float(roc_auc_score(expert_truth[:, j], expert_prediction[:, j]))
                for j, target in enumerate(TARGETS)}
        validation = {"expert_rows": int(expert_mask.sum()), "expert_macro_auc": float(np.mean(list(aucs.values()))),
                      "expert_per_target_auc": aucs, "selection_used": False,
                      "scope": "one held-out fold; complete five-fold OOF is still required"}
        checkpoint = out / f"{prefix}_ema.pt"
        atomic_save({"schema": SCHEMA, "contract_sha256": CONTRACT_SHA256, "contract": TRAINING_CONTRACT,
                     "state": {k: v.detach().cpu() for k, v in ema.state_dict().items()}, "targets": TARGETS,
                     "completed_epoch": EPOCHS, "runtime": versions}, checkpoint)
        # Verify this final saved state before treating it as an inference artifact.
        sample = next(iter(validation_loader))
        ex, em, ep = (v.cuda() for v in sample[:3])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            reference = ema(ex, em, ep).float()
        saved_ema = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ema.load_state_dict(saved_ema["state"], strict=True)
        del saved_ema
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            replay = ema(ex, em, ep).float()
        replay_error = float((reference - replay).abs().max())
        if not torch.isfinite(replay).all() or replay_error > 1e-6:
            raise RuntimeError("final checkpoint prediction replay failed")
        validation["final_checkpoint_replay_max_abs_error"] = replay_error
        validation_loader.dataset.release_pages()
        artifacts.extend([checkpoint, oof_path])
    manifest = {"schema": SCHEMA, "status": "COMPLETE" if complete else "INCOMPLETE_RESUMABLE",
                "fold": FOLD, "completed_epoch": history[-1]["epoch"] if history else 0,
                "contract": TRAINING_CONTRACT, "contract_sha256": CONTRACT_SHA256,
                "history": history, "validation": validation, "runtime": versions,
                "amp_skipped_updates": checkpoint_overflows,
                "amp_skipped_updates_including_discarded_partial_epoch": overflows,
                "checkpoint_step": len(history) * len(train_loader),
                "attempted_step_including_discarded_partial_epoch": step,
                "budget_stopped": budget_stopped,
                "training_expert_rows": int(gold[training_rows].sum()),
                "validation_expert_rows": len(expert_rows), "competition_submission_enabled": False,
                "checkpoint_selection_on_expert_labels": False,
                "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in artifacts},
                "elapsed_seconds_this_run": time.time() - T0}
    (out / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(json.dumps({"event": "fold_run_finished", "fold": FOLD, "status": manifest["status"],
                    "completed_epoch": manifest["completed_epoch"], "validation": validation}))
