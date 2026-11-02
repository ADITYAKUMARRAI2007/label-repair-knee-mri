"""Build the standalone Kaggle EfficientNetV2 Raptor-corpus trainer.

The public Dread Development trainer contains the well-tested corpus reader and
study-level attention head.  This builder keeps that geometry while making the
experiment reproducible in this repository:

* use the audited six-family soft-label consensus;
* load EfficientNetV2-S ImageNet-21K weights from a mounted Kaggle Model;
* keep every expert-labelled study out of training;
* use fp16 on T4 GPUs and write all artifacts to /kaggle/working.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "public_kernels/dread_train/knee-mri-training-the-twelve-finding-model.py"
OUTPUT = ROOT / "public_kernels/effv2_raptor_train/rsna-knee-effv2-raptor-train.py"


LABEL_PRELUDE = r'''

from pathlib import Path

# ---------------------- audited weak-label consensus -------------------------
def _target_name(col):
    raw = str(col).strip()
    if raw in LAB:
        return raw
    aliases = {
        "acl":"ACL", "mcl":"MCL",
        "medial meniscus":"Medial Meniscus", "medial_meniscus":"Medial Meniscus",
        "lateral meniscus":"Lateral Meniscus", "lateral_meniscus":"Lateral Meniscus",
        "medial oa":"Medial OA", "medial_oa":"Medial OA",
        "lateral oa":"Lateral OA", "lateral_oa":"Lateral OA",
        "pf oa":"PF OA", "pf_oa":"PF OA", "patellofemoral oa":"PF OA",
        "effusion":"Effusion", "synovitis":"Synovitis",
        "baker's":"Baker's", "bakers":"Baker's", "baker cyst":"Baker's",
        "contusion":"Contusion", "fracture":"Fracture",
    }
    key = raw.lower().replace("-", " ").replace("__", "_")
    return aliases.get(key)


def _dataset_family(path):
    parts = Path(path).parts
    try:
        k = parts.index("datasets")
        return "/".join(parts[k:k + 2])
    except ValueError:
        return str(Path(path).parent)


def _source_priority(path):
    name = Path(path).name.lower()
    priorities = (
        ("train_targets_soft", 120),
        ("v4_blend", 110),
        ("report_labels_v5", 105),
        ("labels_llm_soft", 100),
        ("labels_llm_gpt56sol", 100),
        ("report_labels_v2", 95),
        ("llm_labels_v2", 90),
        ("llm_labels_full", 80),
    )
    return next((score for token, score in priorities if token in name), 50)


def _read_label_csv(path):
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    uid_col = next((c for c in frame.columns if str(c).strip().lower() in {
        "studyinstanceuid", "study_instance_uid", "studyuid", "study_id", "studyid"
    }), None)
    if uid_col is None:
        return None
    rename = {uid_col: "StudyInstanceUID"}
    for c in frame.columns:
        t = _target_name(c)
        if t:
            rename[c] = t
    frame = frame.rename(columns=rename)
    have = [t for t in LAB if t in frame.columns]
    if len(have) < 8:
        return None
    out = frame[["StudyInstanceUID", *have]].copy()
    out["StudyInstanceUID"] = out["StudyInstanceUID"].astype(str)
    for t in have:
        out[t] = pd.to_numeric(out[t], errors="coerce")
        out.loc[~out[t].between(0, 1), t] = np.nan
    return out


def build_consensus_labels():
    train_path = _find("train.csv")
    if not train_path:
        raise RuntimeError("competition train.csv is not mounted")
    train = pd.read_csv(train_path, dtype={"StudyInstanceUID": str})
    uids = train["StudyInstanceUID"].astype(str).tolist()
    idx = pd.Index(uids, name="StudyInstanceUID")
    family_candidates = {}
    discovered = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        for name in files:
            if not name.lower().endswith(".csv"):
                continue
            low = name.lower()
            if not any(k in low for k in ("label", "llm", "pseudo", "report", "target", "soft")):
                continue
            if any(k in low for k in (
                "target_weights", "targets_weights", "target_hard", "targets_hard",
                "gold_benchmark", "quality_audit",
            )):
                continue
            path = os.path.join(root, name)
            if os.path.samefile(path, train_path):
                continue
            frame = _read_label_csv(path)
            if frame is None or frame["StudyInstanceUID"].isin(uids).mean() < 0.50:
                continue
            family_candidates.setdefault(_dataset_family(path), []).append((path, frame))
            discovered.append(path)

    families = []
    selected = []
    for family, candidates in sorted(family_candidates.items()):
        best = max(_source_priority(path) for path, _ in candidates)
        chosen = [(path, frame) for path, frame in candidates if _source_priority(path) == best]
        selected.extend(path for path, _ in chosen)
        family_frame = pd.DataFrame(index=idx)
        for target in LAB:
            cols = []
            for _, frame in chosen:
                if target in frame.columns:
                    s = frame.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
                    cols.append(s.reindex(idx)[target].to_numpy(np.float32))
            if cols:
                family_frame[target] = np.nanmedian(np.stack(cols, axis=1), axis=1)
        families.append((family, family_frame))
        print("label family:", family, "selected=", [Path(p).name for p, _ in chosen],
              "candidates=", len(candidates), flush=True)

    if len(families) < 5:
        raise RuntimeError(f"expected at least five independent label families, got {len(families)}")
    consensus = pd.DataFrame({"StudyInstanceUID": uids})
    for target in LAB:
        cols = [frame[target].to_numpy(np.float32) for _, frame in families if target in frame]
        if not cols:
            raise RuntimeError(f"no weak labels for {target}")
        consensus[target] = np.nanmedian(np.stack(cols, axis=1), axis=1)
    coverage = consensus[LAB].notna().sum(axis=1)
    if int((coverage >= 8).sum()) < 4300:
        raise RuntimeError(f"weak-label coverage too low: {(coverage >= 8).sum()}/{len(consensus)}")
    if consensus[LAB].isna().any().any():
        med = consensus[LAB].median(axis=0)
        consensus[LAB] = consensus[LAB].fillna(med)

    # Match the ConvNeXt arm's gentle sharpening, while retaining soft uncertainty.
    values = np.clip(consensus[LAB].to_numpy(np.float32), 1e-4, 1 - 1e-4)
    logits = np.log(values / (1 - values))
    consensus[LAB] = 1 / (1 + np.exp(-logits / 0.85))
    out = Path("/kaggle/working/effv2_consensus_labels.parquet")
    consensus.to_parquet(out, index=False)
    audit = {
        "schema": "effv2_raptor_label_consensus_v1",
        "families": [family for family, _ in families],
        "selected_sources": selected,
        "discovered_sources": discovered,
        "studies": len(consensus),
        "complete_targets": int((coverage == len(LAB)).sum()),
    }
    Path("/kaggle/working/effv2_label_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2), flush=True)
    return str(out)


def _find_effv2_weights():
    hits = []
    for root, dirs, files in os.walk("/kaggle/input/models"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "model.safetensors" in files and "efficientnetv2" in root.lower():
            hits.append(os.path.join(root, "model.safetensors"))
    if len(hits) != 1:
        raise RuntimeError(f"expected one mounted EfficientNetV2 safetensors file, got {hits}")
    return hits[0]

'''


def build() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    text = text.replace(
        "This is the training half of the model behind the public 0.924 inference notebook.",
        "Derived from the public 0.924 Raptor trainer; this run tests an EfficientNetV2-S diversity arm.",
        1,
    )
    doc_start = text.index('"""')
    doc_end = text.index('"""', doc_start + 3) + 3
    honest_doc = '''"""Train the E01 EfficientNetV2-S Raptor-corpus diversity arm.

This program keeps the public 44-slice Raptor corpus geometry and its
target-specific attention head, but trains a distinct EfficientNetV2-S
ImageNet-21K backbone. The 4,349 report-labelled studies use an audited median
consensus of six independent soft-label families. All 58 expert-labelled
studies are held out and used only as a noisy rejection and checkpoint gate.

The run writes best-epoch and top-three weight-averaged checkpoints, Gold58
predictions, target-level metrics, and a label-source audit. It makes no public
leaderboard claim until the resulting checkpoint is submitted independently.
"""'''
    text = text[:doc_start] + honest_doc + text[doc_end:]
    text = text.replace(
        'HERE = os.path.dirname(os.path.abspath(__file__))\nRSNA = os.path.dirname(HERE)',
        'HERE = "/kaggle/working"\nRSNA = "/kaggle/input"',
        1,
    )
    old_find = '''def _find(*names, root="/kaggle/input"):
    """First path under root whose basename matches one of names."""
    for d, _, fs in os.walk(root):
        for n in names:
            if n in fs:
                return os.path.join(d, n)
    return None
'''
    new_find = '''def _find(*names, root="/kaggle/input"):
    """First matching input artifact, without walking the 820k DICOM files."""
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in
                   ("train_series", "test_series", "train_images", "test_images")]
        for n in names:
            if n in fs:
                return os.path.join(d, n)
    return None
'''
    if old_find not in text:
        raise RuntimeError("input finder source changed")
    text = text.replace(old_find, new_find, 1)
    # Keep the normalization argument singular; older notebook exports carried it twice.
    duplicate = '    ap.add_argument("--norm", default="none", choices=["none", "imagenet"])\n'
    if text.count(duplicate) not in (1, 2):
        raise RuntimeError("unexpected --norm source layout")
    if text.count(duplicate) == 2:
        text = text.replace(duplicate + duplicate, duplicate, 1)

    old_builder = '''def build_backbone(arch="vit_small_patch16_224", pretrained=False):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)
'''
    new_builder = '''def build_backbone(arch="vit_small_patch16_224", pretrained=False):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if pretrained and "efficientnetv2" in arch:
        kw["pretrained_cfg_overlay"] = {"file": _find_effv2_weights()}
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)
'''
    if old_builder not in text:
        raise RuntimeError("backbone builder source changed")
    text = text.replace(old_builder, new_builder, 1)

    # Timm's TF EfficientNetV2 checkpoint uses [-1, 1] normalization.
    text = text.replace(
        'self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)\n'
        '        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)',
        'self._mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)\n'
        '        self._std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)',
        1,
    )
    text = text.replace("dtype=torch.bfloat16", "dtype=torch.float16")

    marker = '\nif __name__ == "__main__":\n    main()'
    if marker not in text:
        raise RuntimeError("main marker source changed")
    runner = r'''

if __name__ == "__main__":
    label_path = build_consensus_labels()
    sys.argv = [
        "train_effv2_raptor",
        "--labels", label_path,
        "--arch", "tf_efficientnetv2_s.in21k",
        "--res", "336",
        "--epochs", "12",
        "--bs", "2",
        "--k", "10",
        "--k_eval", "24",
        "--bb_lr", "0.000025",
        "--head_lr", "0.0008",
        "--wd", "0.02",
        "--workers", "2",
        "--freeze_blocks", "4",
        "--topk", "3",
        "--norm", "imagenet",
        "--grad_ckpt",
        "--tag", "effv2s_336_k10_consensus_v1",
        "--seed", "20260929",
    ]
    main()
'''
    text = text.replace(marker, LABEL_PRELUDE + runner, 1)

    # Fail the build if any safety-critical edit silently stopped matching.
    checks = (
        "effv2_consensus_labels.parquet",
        "tf_efficientnetv2_s.in21k",
        "pretrained_cfg_overlay",
        "dtype=torch.float16",
        'HERE = "/kaggle/working"',
    )
    for token in checks:
        if token not in text:
            raise RuntimeError(f"missing generated token: {token}")
    if text.count('ap.add_argument("--norm"') != 1:
        raise RuntimeError("duplicate --norm argument survived")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
