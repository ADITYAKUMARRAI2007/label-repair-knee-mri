"""Open only the owned wide cache and require mounted-file integrity evidence."""

# __EXPECTED_WIDE_CACHE__


def discover_wide_files():
    files = {}
    for part in ("base", "extension"):
        prefix = f"e04_wide_{part}"
        manifest_path = find_unique(f"{prefix}_manifest.json", EXPECTED_WIDE_CACHE[part]["manifest_sha256"])
        manifest = json.loads(manifest_path.read_text())
        count = 3200 if part == "base" else 1207
        if (manifest["schema"] != "e04_owned_wide_cache_v1" or manifest["status"] != "COMPLETE"
                or manifest["shape"] != [count, 44, 336, 336]
                or manifest["span"] != [.06, .94] or manifest["crop_mm"] != 140.0):
            raise RuntimeError("owned wide-cache manifest contract changed")
        files[manifest_path.name] = {"path": str(manifest_path), "sha256": EXPECTED_WIDE_CACHE[part]["manifest_sha256"],
                                     "bytes": manifest_path.stat().st_size}
        for name, receipt in manifest["artifacts"].items():
            path = find_unique(name)
            if path.stat().st_size != receipt["bytes"]:
                raise RuntimeError(f"mounted wide-cache artifact size mismatch: {name}")
            files[name] = {"path": str(path), **receipt}
    return files


def verify_wide_files():
    files = discover_wide_files()
    for name, item in files.items():
        if sha256_file(item["path"]) != item["sha256"]:
            raise RuntimeError(f"mounted wide-cache hash mismatch: {name}")
    return {"schema": "e04_mounted_cache_verification_v1", "status": "PASS", "files": files}


def load_parent_cache_verification():
    path = Path(os.environ["E04_CACHE_VERIFICATION_PATH"])
    if sha256_file(path) != os.environ["E04_CACHE_VERIFICATION_SHA256"]:
        raise RuntimeError("parent cache-verification receipt changed")
    receipt = json.loads(path.read_text())
    actual = discover_wide_files()
    if receipt["schema"] != "e04_mounted_cache_verification_v1" or receipt["status"] != "PASS" or receipt["files"] != actual:
        raise RuntimeError("parent verified different mounted cache artifacts")
    return actual


def open_corpus():
    # Parent hashes each immutable mounted file once before spawning the models.
    # Workers verify that receipt and the same manifests/sizes, rather than
    # streaming 22 GB every time a Dataset opens its mmap.
    paths = load_parent_cache_verification()
    a = np.load(paths["e04_wide_base_vols.npy"]["path"], mmap_mode="r")
    b = np.load(paths["e04_wide_extension_vols.npy"]["path"], mmap_mode="r")
    masks = np.concatenate([np.load(paths[f"e04_wide_{part}_masks.npy"]["path"], allow_pickle=False)
                            for part in ("base", "extension")])
    ids = np.concatenate([np.load(paths[f"e04_wide_{part}_ids.npy"]["path"], allow_pickle=False)
                          for part in ("base", "extension")]).astype(str)
    corpus = TwoPartCorpus(a, b)
    if (a.shape != (3200, 44, 336, 336) or b.shape != (1207, 44, 336, 336)
            or corpus.shape != (4407, 44, 336, 336) or masks.shape != (4407, 44)
            or len(set(ids)) != 4407 or masks.dtype != np.uint8
            or a.dtype != np.uint8 or b.dtype != np.uint8):
        raise RuntimeError("owned wide-corpus shape/type/identity drift")
    return corpus, masks, ids
