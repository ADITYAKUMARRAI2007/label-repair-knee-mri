import hashlib,json,re,unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
"""Reconstruct E04's frozen labels/split from public Kaggle inputs at runtime.

No competition records or fold map are uploaded with the pretrained weights.
The builder embeds only the expected hashes and unchanged split algorithm.
"""
import io

E04_TRAIN_SHA256 = '8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33'
E04_FOLD_SHA256 = '62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357'
E04_TARGET_SHA256 = '4bfa52f911afd34d771c67a75aa2532d013d2f71085c57944ead99a241103a03'
E04_WEIGHT_SHA256 = '2631d8d53f7ebe4f6f45d9bc60caf9108b6edff6e88405f95bce5dbe5b355524'
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

PUBLIC_LABEL_SHA256 = {
    "train_targets_soft.csv": "6dac0e2d9ba79065ec794fee743c0286978c2323fcb9625bb74ef2a73d4d1475",
    "train_targets_weights.csv": "3fedf0a0f792d5644f0e9fd537a3add2e0f81aa25e9540db74ec523bdd3a3c18",
    "report_labels_v2.csv": "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920",
    "labels_llm_gpt56sol.csv": "a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae",
    "llm_labels_v4_blend.csv": "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4",
    "labels_llm_soft.csv": "8cec477fb555e88bbb4196197fd3f806cbafed8cc2f6b2d48d26a8d49fbd86a1",
    "report_labels_v5.csv": "16cf44a89e027c80ff9b1a69e1ac6713d1f6da1765b3748c01f6ae91827ef167",
}

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _normalise_report(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\b\d+(?:[.,]\d+)?\b", " <n> ", text)
    text = re.sub(r"[^a-z<>]+", " ", text)
    return " ".join(text.split())

def _simhash(text: str) -> int:
    words = text.split()
    grams = [" ".join(words[i:i + 3]) for i in range(max(len(words) - 2, 1))]
    if len(words) < 3:
        grams = [text]
    votes = np.zeros(64, np.int32)
    for gram in grams:
        value = int.from_bytes(hashlib.sha1(gram.encode()).digest()[:8], "big")
        for bit in range(64):
            votes[bit] += 1 if value & (1 << bit) else -1
    out = 0
    for bit in range(64):
        if votes[bit] >= 0:
            out |= 1 << bit
    return out

def report_groups(reports) -> np.ndarray:
    """Join exact and extremely close report templates without stochastic NLP."""
    normal = [_normalise_report(v) for v in reports]
    n = len(normal)
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    exact: dict[str, int] = {}
    for i, text in enumerate(normal):
        if text in exact:
            union(i, exact[text])
        else:
            exact[text] = i

    hashes = [_simhash(text) for text in normal]
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, value in enumerate(hashes):
        for band in range(4):
            buckets.setdefault((band, (value >> (16 * band)) & 0xFFFF), []).append(i)
    checked = set()
    for members in buckets.values():
        if len(members) > 80:  # generic language buckets are not evidence of duplication
            continue
        for p, a in enumerate(members):
            for b in members[p + 1:]:
                pair = (min(a, b), max(a, b))
                if pair in checked:
                    continue
                checked.add(pair)
                la, lb = len(normal[a]), len(normal[b])
                if min(la, lb) < 40 or min(la, lb) / max(la, lb, 1) < 0.90:
                    continue
                if (hashes[a] ^ hashes[b]).bit_count() <= 3:
                    union(a, b)

    roots = [find(i) for i in range(n)]
    labels = {root: j for j, root in enumerate(sorted(set(roots)))}
    return np.asarray([labels[root] for root in roots], np.int32)

def make_folds(train: pd.DataFrame, targets: np.ndarray, gold: np.ndarray,
               seed: int = 20260930) -> tuple[np.ndarray, dict]:
    """Five grouped, multilabel-balanced folds with Gold balanced independently."""
    groups = report_groups(train["Report"].fillna(""))
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    gold_groups = [g for g in unique if gold[members[g]].any()]
    if any(int(gold[members[g]].sum()) != 1 for g in gold_groups):
        raise RuntimeError("near-duplicate grouping joined expert-labelled studies")
    rng = np.random.default_rng(seed)
    gold_idx = np.flatnonzero(gold > 0)
    gold_y = targets[gold_idx].astype(np.int8)
    capacities = np.asarray([12, 12, 12, 11, 11], np.int64)
    total_gold_pos = gold_y.sum(0)
    best = None
    for _ in range(50000):
        permutation = rng.permutation(len(gold_idx))
        gold_folds = np.empty(len(gold_idx), np.int8)
        start = 0
        for fold, count in enumerate(capacities):
            gold_folds[permutation[start:start + count]] = fold
            start += count
        fold_pos = np.stack([gold_y[gold_folds == fold].sum(0) for fold in range(5)])
        if np.any(fold_pos == 0) or np.any(fold_pos == capacities[:, None]):
            continue
        expected = capacities[:, None] * total_gold_pos[None, :] / len(gold_idx)
        denom = np.maximum(expected * (1 - total_gold_pos[None, :] / len(gold_idx)), 0.5)
        objective = float(np.sum((fold_pos - expected) ** 2 / denom))
        if best is None or objective < best[0]:
            assignment = {int(groups[idx]): int(fold)
                          for idx, fold in zip(gold_idx, gold_folds)}
            best = (objective, assignment, capacities.astype(float), fold_pos.astype(float))
    if best is None:
        raise RuntimeError("could not construct valid grouped Gold folds")

    assignment = best[1]
    hard = (targets >= 0.5).astype(np.float64)
    desired_n = len(train) / 5
    desired_pos = hard.sum(0) / 5
    fold_n = np.zeros(5, np.float64)
    fold_pos = np.zeros((5, len(TARGETS)), np.float64)
    for g, fold in assignment.items():
        idx = members[g]
        fold_n[fold] += len(idx)
        fold_pos[fold] += hard[idx].sum(0)

    remaining = [int(g) for g in unique if int(g) not in assignment]
    rarity = 1 / np.maximum(hard.sum(0), 1)
    remaining.sort(key=lambda g: (
        -float((hard[members[g]].sum(0) * rarity).sum()),
        -len(members[g]), g,
    ))
    for g in remaining:
        idx = members[g]
        n = len(idx)
        pos = hard[idx].sum(0)
        costs = []
        for fold in range(5):
            size_before = ((fold_n[fold] - desired_n) / desired_n) ** 2
            size_after = ((fold_n[fold] + n - desired_n) / desired_n) ** 2
            scale = np.maximum(desired_pos, 1)
            pos_before = np.mean(((fold_pos[fold] - desired_pos) / scale) ** 2)
            pos_after = np.mean(((fold_pos[fold] + pos - desired_pos) / scale) ** 2)
            costs.append((size_after - size_before) + 0.35 * (pos_after - pos_before))
        chosen = min(range(5), key=lambda f: (costs[f], fold_n[f], f))
        assignment[g] = chosen
        fold_n[chosen] += n
        fold_pos[chosen] += pos

    folds = np.asarray([assignment[int(g)] for g in groups], np.int8)
    gold_counts = np.bincount(folds[gold > 0], minlength=5)
    gold_pos = np.stack([targets[(folds == f) & (gold > 0)].sum(0) for f in range(5)])
    if np.any(gold_counts < 9) or np.any(gold_counts > 14):
        raise RuntimeError(f"Gold fold counts invalid: {gold_counts.tolist()}")
    if np.any(gold_pos <= 0) or np.any(gold_pos >= gold_counts[:, None]):
        raise RuntimeError("a Gold fold lacks a class for at least one target")
    if any(len(np.unique(folds[idx])) != 1 for idx in members.values()):
        raise RuntimeError("report group crossed fold boundary")

    mapping = "".join(f"{uid},{int(fold)}\n" for uid, fold in
                      sorted(zip(train["StudyInstanceUID"].astype(str), folds)))
    audit = {
        "schema": "t03_grouped_folds_v1",
        "seed": seed,
        "fold_sha256": hashlib.sha256(mapping.encode()).hexdigest(),
        "fold_counts": np.bincount(folds, minlength=5).tolist(),
        "gold_fold_counts": gold_counts.tolist(),
        "report_groups": int(len(unique)),
        "duplicate_groups": int(sum(len(v) > 1 for v in members.values())),
        "largest_group": int(max(map(len, members.values()))),
    }
    return folds, audit

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

METADATA={'schema': 'e04_frozen_report_contract_v1', 'status': 'FROZEN', 'target_order': ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'], 'official_train_sha256': '8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33', 'split_source_sha256': 'e30c65a3260fcbb99460ac953e7df7cbfc6a55b32489d506608c35ec2a6ff6b0', 'split_audit': {'schema': 't03_grouped_folds_v1', 'seed': 20260930, 'fold_sha256': '62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357', 'fold_counts': [882, 882, 881, 881, 881], 'gold_fold_counts': [12, 12, 12, 11, 11], 'report_groups': 4073, 'duplicate_groups': 75, 'largest_group': 59}, 'fusion': {'families': ['report_labels_v2.csv', 'labels_llm_gpt56sol.csv', 'llm_labels_v4_blend.csv', 'labels_llm_soft.csv'], 'rule': 'cellwise median, clipped 0.01--0.99', 'weak_weight': 'clip(1 - source probability range, 0.25, 1.0)', 'expert_weight': 2.0, 'expert_labels': 'official binary truth in training folds only', 'minimum_weak_cell_sources': 4, 'source_receipts': {'report_labels_v2.csv': {'path': 'labels/pilkwang_rsna-knee-llm-labels/report_labels_v2.csv', 'sha256': '6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920'}, 'labels_llm_gpt56sol.csv': {'path': 'labels/lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', 'sha256': 'a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae'}, 'llm_labels_v4_blend.csv': {'path': 'labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', 'sha256': 'eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4'}, 'labels_llm_soft.csv': {'path': 'labels/dreaddevelopment_rsna-knee-labels/labels_llm_soft.csv', 'sha256': '8cec477fb555e88bbb4196197fd3f806cbafed8cc2f6b2d48d26a8d49fbd86a1'}}}, 'fold_reconstruction_only_sources': {'train_targets_soft.csv': {'path': 'labels/rayanbabur_rsna-knee-2026-calibrated-soft-targets/train_targets_soft.csv', 'sha256': '6dac0e2d9ba79065ec794fee743c0286978c2323fcb9625bb74ef2a73d4d1475'}, 'train_targets_weights.csv': {'path': 'labels/rayanbabur_rsna-knee-2026-calibrated-soft-targets/train_targets_weights.csv', 'sha256': '3fedf0a0f792d5644f0e9fd537a3add2e0f81aa25e9540db74ec523bdd3a3c18'}, 'report_labels_v2.csv': {'path': 'labels/pilkwang_rsna-knee-llm-labels/report_labels_v2.csv', 'sha256': '6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920'}, 'labels_llm_gpt56sol.csv': {'path': 'labels/lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv', 'sha256': 'a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae'}, 'llm_labels_v4_blend.csv': {'path': 'labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv', 'sha256': 'eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4'}, 'labels_llm_soft.csv': {'path': 'labels/dreaddevelopment_rsna-knee-labels/labels_llm_soft.csv', 'sha256': '8cec477fb555e88bbb4196197fd3f806cbafed8cc2f6b2d48d26a8d49fbd86a1'}, 'report_labels_v5.csv': {'path': 'labels/yunusgmsoy_rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv', 'sha256': '16cf44a89e027c80ff9b1a69e1ac6713d1f6da1765b3748c01f6ae91827ef167'}}, 'arrays': {'e04_targets.npy': '4bfa52f911afd34d771c67a75aa2532d013d2f71085c57944ead99a241103a03', 'e04_weights.npy': '2631d8d53f7ebe4f6f45d9bc60caf9108b6edff6e88405f95bce5dbe5b355524'}, 'validation_rules': ['five independently trained backbones; no shared expert-adapted backbone', 'exclude every held-out report group from model updates', 'fixed checkpoint rule before training; do not select on expert-fold scores', 'complete all 58 expert OOF predictions before the coarse validation gate'], 'limitations': ['report grouping does not establish patient/site separation', 'public report-label prompt tuning independence from Gold is unverified', 'previous experiments have already used Gold; OOF is a coarse rejection gate'], 'rows': None}
EXPECTED_METADATA_SHA256='88a7e801945c4cc5145cd3ece2dced031d662decd9588bcb0cf3a816e088272f'

def reconstruct(workspace, train_path, inputs):
    workspace, train_path, inputs = Path(workspace), Path(train_path), Path(inputs)
    def find_unique(name, expected):
        receipt = METADATA['fold_reconstruction_only_sources'][name]
        parent = Path(receipt['path']).parent.name
        owner, slug = parent.split('_', 1)
        candidates = [inputs / receipt['path'], inputs / slug / name,
                      inputs / 'datasets' / owner / slug / name]
        valid = [p for p in candidates if p.is_file() and sha256_file(p) == expected]
        if not valid: raise FileNotFoundError('Required public label CSV: ' + name)
        return valid[0]
    # The historical reconstruction uses a supplied find_unique helper.
    globals()['find_unique'] = find_unique
    train = pd.read_csv(train_path, dtype={'StudyInstanceUID': str})
    ids = np.asarray(sorted(train.StudyInstanceUID.astype(str)), dtype=str)
    y, weights, gold, folds, groups, audit = reconstruct_frozen_inputs(train_path, ids)
    assert audit == METADATA['split_audit']
    reconstructed = dict(METADATA)
    reconstructed['rows'] = [{'StudyInstanceUID': str(uid), 'fold': int(f),
                              'report_group': int(g), 'expert': bool(expert)}
                             for uid, f, g, expert in zip(ids, folds, groups, gold)]
    output = workspace / 'runs/E04_contract'; output.mkdir(parents=True, exist_ok=False)
    metadata = output / 'e04_contract.json'
    metadata.write_text(json.dumps(reconstructed, indent=2) + '\n')
    np.save(output / 'e04_targets.npy', y)
    np.save(output / 'e04_weights.npy', weights)
    assert sha256_file(metadata) == EXPECTED_METADATA_SHA256
    for name, expected in reconstructed['arrays'].items():
        assert sha256_file(output / name) == expected, name
    return {'status': 'EXACT_FROZEN_INPUTS_RECONSTRUCTED_FROM_NATIVE_PUBLIC_SOURCES',
            'studies': len(ids), 'experts': int(gold.sum()),
            'metadata_sha256': sha256_file(metadata),
            'local_image_feature_or_label_array_upload_needed': False}

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(reconstruct(args.workspace, args.train, args.inputs)))
