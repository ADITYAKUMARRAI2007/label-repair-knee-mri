"""Reconstruct E04's frozen labels/split from public Kaggle inputs at runtime.

No competition records or fold map are uploaded with the pretrained weights.
The builder embeds only the expected hashes and unchanged split algorithm.
"""
import io

# __FROZEN_HASHES__
# __SPLIT_ALGORITHM__

E04_REPORT_FAMILIES = ("report_labels_v2.csv", "labels_llm_gpt56sol.csv",
                       "llm_labels_v4_blend.csv", "labels_llm_soft.csv")


def numpy_file_sha256(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def reconstruct_frozen_inputs(train_path, corpus_ids):
    if sha256_file(train_path) != E04_TRAIN_SHA256:
        raise RuntimeError("official training table changed")
    train = pd.read_csv(train_path, dtype={"StudyInstanceUID": str})
    uids = train.StudyInstanceUID.astype(str).tolist()
    if len(uids) != 4407 or set(uids) != set(corpus_ids):
        raise RuntimeError("training/corpus identity drift")
    gold = train[TARGETS].notna().all(1).to_numpy()
    if int(gold.sum()) != 58:
        raise RuntimeError("expert coverage drift")
    values = {}
    for name in PUBLIC_LABEL_SHA256:
        # The historical confidence table is unnecessary to reconstruct folds.
        if name == "train_targets_weights.csv":
            continue
        path = find_unique(name, PUBLIC_LABEL_SHA256[name])
        frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
        if frame.StudyInstanceUID.duplicated().any():
            raise RuntimeError(f"duplicate report source UID: {name}")
        values[name] = frame.set_index("StudyInstanceUID").reindex(uids)[TARGETS].to_numpy(np.float32)
    historical = np.clip(.60 * values["train_targets_soft.csv"] + .40 * np.nanmedian(
        np.stack([values[n] for n in (*E04_REPORT_FAMILIES, "report_labels_v5.csv")]), axis=0),
        .01, .99).astype(np.float32)
    historical[gold] = train.loc[gold, TARGETS].to_numpy(np.float32)
    folds, audit = make_folds(train, historical, gold.astype(np.uint8))
    if audit["fold_sha256"] != E04_FOLD_SHA256:
        raise RuntimeError("frozen fold map failed reproduction")
    groups = report_groups(train.Report.fillna(""))
    families = np.stack([values[n] for n in E04_REPORT_FAMILIES])
    weak = ~gold
    if np.any(np.isfinite(families[:, weak]).sum(0) < 2):
        raise RuntimeError("inadequate weak-label source coverage")
    y = np.full((len(uids), len(TARGETS)), np.nan, np.float32)
    y[gold] = train.loc[gold, TARGETS].to_numpy(np.float32)
    y[weak] = np.clip(np.nanmedian(families[:, weak], axis=0), .01, .99)
    weights = np.full_like(y, 2.0)
    spread = np.nanmax(families[:, weak], axis=0) - np.nanmin(families[:, weak], axis=0)
    weights[weak] = np.clip(1 - spread, .25, 1.0)
    if (numpy_file_sha256(y) != E04_TARGET_SHA256
            or numpy_file_sha256(weights) != E04_WEIGHT_SHA256):
        raise RuntimeError("frozen target/weight reproduction failed")
    by_uid = {uid: i for i, uid in enumerate(uids)}
    order = np.asarray([by_uid[uid] for uid in corpus_ids])
    return y[order], weights[order], gold[order], folds[order], groups[order], audit


def fold_row_sets(ids, gold, folds, groups, heldout_fold):
    training = np.flatnonzero(folds != heldout_fold)
    validation = np.flatnonzero(folds == heldout_fold)
    if set(groups[training]) & set(groups[validation]):
        raise RuntimeError("report group crossed the held-out boundary")
    if len(set(ids[training]) & set(ids[validation])):
        raise RuntimeError("study crossed the held-out boundary")
    if np.any(folds[training] == heldout_fold):
        raise RuntimeError("held-out study in training updates")
    expert_validation = validation[gold[validation]]
    if len(expert_validation) not in (11, 12):
        raise RuntimeError("expert fold coverage drift")
    return training, validation, expert_validation
