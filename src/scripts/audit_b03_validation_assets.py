#!/usr/bin/env python3
"""Inventory public component validation without constructing/tuning a B03 blend."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/B03_validation_audit'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    contract=json.loads((ROOT/'runs/E04_contract/e04_contract.json').read_text())
    targets=contract['target_order']
    with (ROOT/'data/train.csv').open() as f:official={r['StudyInstanceUID']:r for r in csv.DictReader(f)}
    gold={r['StudyInstanceUID'] for r in contract['rows'] if r['expert']}
    assert len(official)==4407 and len(gold)==58
    records=[]
    for receipt in json.loads((OUT/'download_receipt.json').read_text()):
        p=ROOT/receipt['path'];assert sha(p)==receipt['sha256']
    def check(ids,pred,scope):
        ids=list(map(str,ids));assert len(ids)==len(set(ids))
        assert set(ids)==(gold if scope=='gold58' else set(official))
        assert pred.shape==(len(ids),12) and np.isfinite(pred).all()
        assert ((pred>=0)&(pred<=1)).all()
        return ids
    # Published CoAt references: real predictions, but selected using this same cohort.
    cases=[
        ('resgated','mattiaangeli__rsna-knee-coat-resgated-ep10-top3','gold58_top3_predictions.npz','coat_resgated_ep10_top3_manifest.json','gold58_probability_mean_diagnostic_auc'),
        ('global96','mattiaangeli__rsna-knee-coatnet-global96-top3','coatnet_global96_gold58_reference.npz','coatnet_global96_top3_manifest.json','gold58_probability_mean_auc'),
        ('d4','mattiaangeli__rsna-knee-coatnet-d4-depthzone-swa3-b2','d4_gold58_reference.npz','coatnet_pairfilm_manifest.json','gold58_probability_mean_auc'),
    ]
    for name,directory,file,manifest_name,score_key in cases:
        folder=OUT/directory;manifest=json.loads((folder/manifest_name).read_text())
        z=np.load(folder/file,allow_pickle=False)
        ids=check(z['study_uids'],z['probability_mean'],'gold58')
        truth=np.array([[float(official[u][t]) for t in targets] for u in ids])
        assert np.array_equal(truth,z['truth'])
        score=float(roc_auc_score(truth,z['probability_mean'],average='macro'))
        assert abs(score-manifest['selection'][score_key])<1e-12
        declared=manifest.get('files',manifest.get('artifacts',{})).get(file)
        if declared:assert sha(folder/file)==declared
        records.append({'component':name,'rows':58,'reference_auc_reproduced':score,
                        'file':str((folder/file).relative_to(ROOT)),'sha256':sha(folder/file),
                        'independent_of_checkpoint_selection':False,
                        'selection_evidence':manifest.get('training',{}).get('selector',manifest['selection'].get('criterion')),
                        'exact_B03_runtime_prediction_claimed':False,
                        'allowed_use':'preprocessing/replay diagnostics; reused-cohort descriptive comparison only'})
    # DINO OOF bundle has aggregate probabilities, not per-member window predictions.
    p=OUT/'pilkwang__rsna-knee-weights/oof.npz';z=np.load(p,allow_pickle=False)
    ids=check(z['ids'],z['pred'],'all4407');assert z['targets'].tolist()==targets
    assert {u for u,m in zip(ids,z['gold_mask']) if m}==gold
    assert np.isin(z['y_derived'],[0,1]).all()
    records.append({'component':'DINO public OOF','rows':4407,'sha256':sha(p),
                    'gold_mask_matches':True,'per_member_or_window_predictions_present':False,
                    'exact_B03_runtime_prediction_claimed':False,
                    'reason':'B03 additionally uses overlapping windows, finding-specific pooling, legacy branches and per-fold ranks; this aggregate table cannot reconstruct them.'})
    # Match the attached E13 output to the exact head hash used by B03.
    for name in ['rsna-knee-e11-train','rsna-knee-e13-train']:
        folder=OUT/name/'output';audit=json.loads((folder/'rad_e11_audit.json').read_text())
        p=folder/'rsna_rad_e11/v52_e11_oof.csv'
        assert sha(p)==audit['oof_sha256']
        with p.open() as f:rows=list(csv.DictReader(f))
        predictions=np.array([[float(r[t]) for t in targets] for r in rows])
        ids=check([r['StudyInstanceUID'] for r in rows],predictions,'all4407')
        gold_mask=np.array([u in gold for u in ids]);assert all(bool(int(r['is_gold']))==m for r,m in zip(rows,gold_mask))
        folds=np.array([int(r['fold']) for r in rows]);assert set(folds)==set(range(5))
        truth=np.array([[float(official[u][t]) for t in targets] for u in np.array(ids)[gold_mask]])
        score=float(roc_auc_score(truth,predictions[gold_mask],average='macro'))
        assert abs(score-audit['gold_oof_auc'])<1e-12
        records.append({'component':name,'rows':4407,'sha256':sha(p),
                        'heads_sha256':audit['heads_sha256'],
                        'matches_B03_E13_head_hash':audit['heads_sha256']=='ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a',
                        'published_gold_oof_auc_reproduced':score,
                        'gold_fold_counts':{str(k):int(((folds==k)&gold_mask).sum()) for k in range(5)},
                        'exact_B03_runtime_prediction_claimed':False,
                        'reason':'Table evaluates each held-out fold head on its training feature layout; B03 averages five heads and also uses E13 on E11 features plus a fitted calibrator.'})
    reader_listing=json.loads((OUT/'goodpjw2008__rsna-knee-2-5d-convnext-reader_files.json').read_text())
    reader_files=[r['name'] for r in reader_listing['files']]
    assert not any(n.endswith(('.csv','.npz')) for n in reader_files)
    records.append({'component':'ConvNeXt reader','public_prediction_files':[],
                    'training_code_excludes_gold':True,'final_epoch_rule':True,
                    'gold_related_weak_exclusion_proven':False,
                    'split_reconstruction':'research/B03_validation_audit/reader_split_reconstruction.json',
                    'reason':'Scoring source is public; its OOF prediction files and original label/fold table are not in the listed dataset.'})
    result={'status':'PARTIAL_COMPONENT_VALIDATION_RECOVERED_FULL_B03_HOLDOUT_UNAVAILABLE',
            'components':records,'full_B03_independent_validation_available':False,
            'missing_or_unverified':['Exact ConvNeXt reader held-out probabilities.',
               'A5 and Raptor component training membership and matching held-out probabilities.',
               'Per-window DINO outputs for exact B03 pooling.',
               'Calibrator training membership and an outer-fold version.',
               'Independent cohort unaffected by upstream Gold58 checkpoint and blend selection.'],
            'decision':'Do not spend GPU quota replaying the complete blend on Gold58 and call that independent validation. Retain0.944 anchor; recovered tables are useful component diagnostics only.',
            'selection_or_blend_sweeps_run':0,'gpu_jobs_launched':0,'new_competition_submissions':0,
            'next_available_actions':['Check any additional expert-label release provenance before accepting it as new validation.',
                 'If genuinely new independently labelled cases become available, freeze evaluation before model/weight selection.',
                 'Without new labels, preserve grouped exclusions and report uncertainty; only justify a new standalone experiment with materially different evidence.'],
            'script_sha256':sha(Path(__file__))}
    (OUT/'validation_asset_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'components_checked':len(records),'matched_E13_OOF':True,'full_B03_independent_validation':False},indent=2))


if __name__=='__main__':main()
