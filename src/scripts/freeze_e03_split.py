#!/usr/bin/env python3
"""Freeze and independently check the E03 report-group split before training."""
import ast
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e03_split", ROOT / "training/e03_split_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
tree = ast.parse((ROOT / "training/e02_resnet34_distill.py").read_text())
targets = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "TARGETS")
with (ROOT / "data/train.csv").open(newline="") as handle:
    records = list(csv.DictReader(handle))
split = contract.freeze_split(records, targets)
assert split == contract.freeze_split(list(reversed(records)), targets), "input order changed split"
by_uid = {r['StudyInstanceUID']: r for r in records}
gold = set(split['gold'])
for group in split['groups'].values():
    members = set(group['members'])
    if members & gold:
        assert not members & set(split['train'])
        assert not members & set(split['weak_validation'])
    elif group['set'] == 'train':
        assert members <= set(split['train'])
    else:
        assert members <= set(split['weak_validation'])
report_sets = {}
for name in ('train','weak_validation','gold','excluded_gold_group_weak'):
    for uid in split[name]:
        report = contract.normalized_report(by_uid[uid]['Report'])
        report_sets.setdefault(report,set()).add('gold_group' if name in ('gold','excluded_gold_group_weak') else name)
assert all(len(names)==1 for names in report_sets.values()), 'normalized report crossed split'
out=ROOT / 'runs/E03_contract'
out.mkdir(parents=True,exist_ok=True)
(out / 'split.json').write_text(json.dumps(split,indent=2)+'\n')
receipt={'schema':'e03_split_audit_v1','status':'PASS','split_sha256':contract.split_sha256(split),
         'counts':{n:len(split[n]) for n in ('train','weak_validation','gold','excluded_gold_group_weak')},
         'report_groups':len(split['groups']),'largest_group':max(len(g['members']) for g in split['groups'].values()),
         'input_order_invariant':True,'gold_targets_used_for_fold_balancing':False,
         'limits':'report groups only; patient/scanner identities are not verified'}
(out / 'split_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
