"""Deterministic report-group split; no model predictions or Gold target tuning."""
import hashlib
import json
import re
import unicodedata

SPLIT_SEED = 20261006


def normalized_report(value):
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\b\d+(?:[.,]\d+)?\b", " <n> ", text)
    return " ".join(re.sub(r"[^a-z<>]+", " ", text).split())


def report_simhash(text):
    words = text.split()
    grams = [" ".join(words[i:i + 3]) for i in range(max(len(words) - 2, 1))]
    if len(words) < 3:
        grams = [text]
    votes = [0] * 64
    for gram in grams:
        value = int.from_bytes(hashlib.sha1(gram.encode()).digest()[:8], "big")
        for bit in range(64):
            votes[bit] += 1 if value & (1 << bit) else -1
    return sum(1 << bit for bit, vote in enumerate(votes) if vote >= 0)


def freeze_split(records, targets):
    records = sorted(records, key=lambda r: r["StudyInstanceUID"])
    uids = [r["StudyInstanceUID"] for r in records]
    if len(uids) != 4407 or len(set(uids)) != len(uids):
        raise ValueError("expected 4,407 unique studies")
    gold = {r["StudyInstanceUID"] for r in records
            if all(str(r.get(t, "")) in {"0", "1", "0.0", "1.0"} for t in targets)}
    if len(gold) != 58:
        raise ValueError("expected Gold58")
    text = [normalized_report(r.get("Report", "")) for r in records]
    parent = list(range(len(uids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    exact = {}
    for i, value in enumerate(text):
        if value in exact:
            join(i, exact[value])
        else:
            exact[value] = i
    hashes = [report_simhash(t) for t in text]
    buckets = {}
    for i, value in enumerate(hashes):
        for band in range(4):
            buckets.setdefault((band, (value >> (band * 16)) & 65535), []).append(i)
    checked = set()
    for members in buckets.values():
        if len(members) > 80:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if (a, b) in checked:
                    continue
                checked.add((a, b))
                la, lb = len(text[a]), len(text[b])
                if min(la, lb) >= 40 and min(la, lb) / max(la, lb) >= 0.9:
                    if (hashes[a] ^ hashes[b]).bit_count() <= 3:
                        join(a, b)
    grouped = {}
    for i, uid in enumerate(uids):
        grouped.setdefault(uids[find(i)], []).append(uid)
    split = {"train": [], "weak_validation": [], "gold": sorted(gold),
             "excluded_gold_group_weak": [], "groups": {}}
    for representative, members in sorted(grouped.items()):
        if gold.intersection(members):
            name = "gold_group"
            split["excluded_gold_group_weak"].extend(u for u in members if u not in gold)
        else:
            value = hashlib.sha256(f"{SPLIT_SEED}:{representative}".encode()).digest()
            name = "weak_validation" if int.from_bytes(value[:8], "big") % 10 == 0 else "train"
            split[name].extend(members)
        split["groups"][representative] = {"members": members, "set": name}
    for name in ("train", "weak_validation", "gold", "excluded_gold_group_weak"):
        split[name].sort()
    partition = [set(split[n]) for n in ("train", "weak_validation", "gold", "excluded_gold_group_weak")]
    if sum(map(len, partition)) != len(set().union(*partition)) or set().union(*partition) != set(uids):
        raise ValueError("split is not an exact disjoint partition")
    split["schema"] = "e03_report_group_split_v1"
    split["seed"] = SPLIT_SEED
    return split


def split_sha256(split):
    return hashlib.sha256(json.dumps(split, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
