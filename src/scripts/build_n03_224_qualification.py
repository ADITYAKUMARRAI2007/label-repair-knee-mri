"""Build a fixed, no-submission execution qualification for the public 224 model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "research/new_public/nartaa_0945_20261009/rsna-knee-0945-efficient-224crop.ipynb"
OUT = ROOT / "public_kernels/n03_224_qualification"
QUALIFICATION_IDS = [
    "1.2.826.0.1.3680043.8.498.93062749017302143105207135454940053142",
    "1.2.826.0.1.3680043.8.498.53587920256280027683523152568832253423",
    "1.2.826.0.1.3680043.8.498.52475562989926669452280424331791927661",
    "1.2.826.0.1.3680043.8.498.74095402538454760179339842746572802563",
    "1.2.826.0.1.3680043.8.498.39020232951771672617852805188845504161",
    "1.2.826.0.1.3680043.8.498.57342523828166309133982222435655662667",
    "1.2.826.0.1.3680043.8.498.73394992092536626285935384324191560182",
    "1.2.826.0.1.3680043.8.498.23063675880057833741455102729816364091",
    "1.2.826.0.1.3680043.8.498.77760453966349959552328150395676117417",
    "1.2.826.0.1.3680043.8.498.29739171278960083313048266628194182569",
    "1.2.826.0.1.3680043.8.498.40239104167848425290754606013326180693",
    "1.2.826.0.1.3680043.8.498.34109091435857467600499949506940613037",
    "1.2.826.0.1.3680043.8.498.87415625824966124550466651648244553892",
    "1.2.826.0.1.3680043.8.498.32226253011052584155674401943276961907",
    "1.2.826.0.1.3680043.8.498.12356611369549236159671896247094661767",
    "1.2.826.0.1.3680043.8.498.11014091506864318694851337553197477906",
    "1.2.826.0.1.3680043.8.498.43256282639827216310036033378453939750",
    "1.2.826.0.1.3680043.8.498.10614508096225944703643482200835890514",
    "1.2.826.0.1.3680043.8.498.98351891935507349137982199658103205287",
    "1.2.826.0.1.3680043.8.498.23822313132245448173452915228147497501",
]
SOURCE_CELL_SHA256 = "e85d994fa93caf1bbc2df04f396649207a69d2463924c41fa3b05ba8801cd38c"
CHECKPOINT_SHA256 = "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text())
    source = "".join(notebook["cells"][4]["source"])
    assert sha_text(source) == SOURCE_CELL_SHA256
    # The frozen cohort is made of the same weak-only training studies used for
    # the earlier N02 operational check.  This changes data routing only: no
    # labels are loaded, read, or scored by this qualification.
    replacements = {
        'tsdir = ROOT + "/test_series"': 'tsdir = ROOT + "/train_series"',
        'tsdir = ROOT + "/test_images"': 'tsdir = ROOT + "/train_images"',
        'test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)':
            'test = pd.read_csv(ROOT + "/train.csv", usecols=["StudyInstanceUID"]); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)',
        'tser = pd.read_csv(ROOT + "/test_series.csv")': 'tser = pd.read_csv(ROOT + "/train_series.csv")',
    }
    for old, new_value in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new_value)
    old = 'test_ids = test["StudyInstanceUID"].tolist()'
    new = (
        'all_test_ids = test["StudyInstanceUID"].tolist()\n'
        f'    QUALIFICATION_IDS = {QUALIFICATION_IDS!r}\n'
        '    assert len(QUALIFICATION_IDS) == 20 and len(set(QUALIFICATION_IDS)) == 20\n'
        '    assert set(QUALIFICATION_IDS).issubset(set(all_test_ids))\n'
        '    test_ids = QUALIFICATION_IDS\n'
        '    test = test.set_index("StudyInstanceUID").loc[test_ids].reset_index()\n'
        '    print("QUALIFICATION_FIXED_IDS", len(test_ids), flush=True)'
    )
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = '    out = "/kaggle/working/submission.csv"\n    sub.to_csv(out, index=False)\n'
    new = (
        '    out = "/kaggle/working/n03_qualification_predictions.csv"\n'
        '    sub.to_csv(out, index=False)\n'
        '    import hashlib\n'
        '    with open(out, "rb") as _f: _sha = hashlib.file_digest(_f, "sha256").hexdigest()\n'
        '    _receipt = {"status":"EXECUTION_PENDING_LOG_AUDIT", "rows":len(sub), '
        '"prediction_sha256":_sha, "checkpoint_sha256":"' + CHECKPOINT_SHA256 + '", '
        '"accuracy_measured":False, "competition_submission_enabled":False}\n'
        '    open("/kaggle/working/n03_qualification_receipt.json", "w").write(json.dumps(_receipt, indent=2, sort_keys=True))\n'
    )
    assert source.count(old) == 1
    source = source.replace(old, new)
    wrapper = '''\n# Qualification policy: source fallbacks are observable but never accepted.\nif __name__ == "__main__":\n    main()\n'''
    # The original cell already calls main; remove that call and re-add one explicit entry point.
    suffix = '\n\nif __name__ == "__main__":\n    main()\n'
    assert source.endswith(suffix)
    source = source[: -len(suffix)] + wrapper
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rsna-knee-n03-224-qualification.py").write_text(source)
    metadata = {
        "id": "adityakumarrai0401/rsna-knee-n03-224-strict-qualification",
        "title": "RSNA Knee N03 224 Strict Qualification",
        "code_file": "rsna-knee-n03-224-qualification.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": ["nartaa/rsna-knee-publication-swa-weights-20261007"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    contract = {
        "schema": "n03_224_execution_qualification_v1",
        "source_code_cell_sha256": SOURCE_CELL_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "qualification_ids": QUALIFICATION_IDS,
        "cohort": "fixed weak-only train studies; no labels loaded or scored",
        "no_competition_submission": True,
        "acceptance": [
            "kernel exits zero",
            "exactly 20 fixed rows in exact order",
            "all finite scores",
            "log has no '[pre] ... FAILED' or '[gpu ...] ... FALLBACK' entry",
            "receipt matches prediction bytes",
        ],
    }
    (OUT / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
