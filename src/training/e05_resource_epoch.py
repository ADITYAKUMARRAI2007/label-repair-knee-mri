"""One complete epoch per arm; resource evidence only, no accuracy selection."""
import argparse
import gc


def pilot_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['control', 'uniform'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one GPU is required')
    out = Path(args.output);out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    seed_all(TRAINING_CONTRACT['seed'])
    torch.backends.cudnn.benchmark = True
    corpus, masks, ids = open_corpus();corpus.release_pages();del corpus
    y, original, gold, folds, groups, split = reconstruct_frozen_inputs(find_unique('train.csv'), ids)
    rows, held, _ = fold_row_sets(ids, gold, folds, groups, 0)
    weights, means = uniform_weak_weights(original, gold, rows)
    if args.arm == 'control': weights = original
    dataset = SeriesWindows(rows, y, masks, TRAIN_PER_SLOT, True)
    loader = DataLoader(dataset, batch_size=BATCH_STUDIES, shuffle=True,
                        num_workers=0, pin_memory=False, drop_last=False)
    model = SeriesMIL(find_unique('model.safetensors', PRETRAINED_SHA256)).cuda()
    ema = copy.deepcopy(model).eval();ema.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': TRAINING_CONTRACT['backbone_lr']},
        {'params': [p for n,p in model.named_parameters() if not n.startswith('encoder.')],
         'lr': TRAINING_CONTRACT['head_lr']}], weight_decay=TRAINING_CONTRACT['weight_decay'])
    # This is epoch one of the frozen 14-epoch schedule, not a one-epoch cosine.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: .1+.9*step/max(1,len(loader)))
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    model.train();torch.cuda.reset_peak_memory_stats()
    order = hashlib.sha256();loss_sum=0.;overflows=0;sampled_rss=[];durations=[]
    epoch_started = time.monotonic();iterator=iter(loader)
    for step in range(1,len(loader)+1):
        before=time.monotonic()
        x, mask, pos, targets, row = next(iterator)
        order.update(row.numpy().astype('<i8').tobytes())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits=model(x.cuda(),mask.cuda(),pos.cuda(),training=True)
            confidence=torch.from_numpy(weights[row.numpy()]).cuda()
            loss=(F.binary_cross_entropy_with_logits(logits.float(),targets.cuda(),reduction='none')*confidence).mean()
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite training output')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=nn.utils.clip_grad_norm_(model.parameters(),3.)
        overflow=not bool(torch.isfinite(norm));overflows+=int(overflow)
        if overflows>10:raise RuntimeError('Persistent gradient overflow')
        scaler.step(optimizer);scaler.update();scheduler.step()
        if not overflow:
            with torch.no_grad():
                decay=min(.998,(1+step)/(10+step))
                for pe,pm in zip(ema.parameters(),model.parameters()):pe.lerp_(pm,1-decay)
                for be,bm in zip(ema.buffers(),model.buffers()):be.copy_(bm)
        loss_sum+=float(loss.detach());torch.cuda.synchronize()
        durations.append(time.monotonic()-before)
        if step%100==0 or step==len(loader):
            rss=current_rss_gib();sampled_rss.append((step,rss))
            print(json.dumps({'arm':args.arm,'step':step,'total_steps':len(loader),
                              'elapsed_seconds':time.monotonic()-epoch_started,'rss_gib':rss,
                              'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30}),flush=True)
    epoch_seconds=time.monotonic()-epoch_started
    dataset.release_pages();del iterator,loader;gc.collect()
    # The weights from this resource-only epoch are never used as a full model.
    check=SeriesWindows(rows[:2],y,masks,EVAL_PER_SLOT,False)
    batch=next(iter(DataLoader(check,batch_size=2)))
    ex,em,ep=batch[0].cuda(),batch[1].cuda(),batch[2].cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):before=ema(ex,em,ep).float()
    checkpoint=out/'resource_only.pt';save_started=time.monotonic()
    torch.save({'model':ema.state_dict(),'resource_only':True},checkpoint)
    checkpoint_seconds=time.monotonic()-save_started
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    ema.load_state_dict(saved['model'],strict=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):after=ema(ex,em,ep).float()
    drift=float((before-after).abs().max())
    if drift>1e-6 or not torch.isfinite(after).all():raise RuntimeError('Checkpoint replay mismatch')
    check.release_pages()
    report={'status':'COMPLETE_RESOURCE_EPOCH_ONLY','arm':args.arm,'rows':len(rows),
            'batches':step,'heldout_rows':len(held),'training_order_sha256':order.hexdigest(),
            'epoch_seconds':epoch_seconds,'step_seconds_median':float(np.median(durations)),
            'checkpoint_seconds':checkpoint_seconds,'checkpoint_sha256':sha256_file(checkpoint),
            'checkpoint_replay_max_abs':drift,'overflows':overflows,'rss_samples':sampled_rss,
            'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30,'versions':runtime_versions(),
            'gpu':torch.cuda.get_device_name(0),'training_loss_not_accuracy':loss_sum/step,
            'projected_10_models_14_epochs_training_hours':epoch_seconds*140/3600,
            'projection_excludes':'Final validation, repeated startup, transfer and checkpoint I/O',
            'elapsed_seconds':time.monotonic()-started,'accuracy_measured':False,
            'competition_submission_enabled':False,'checkpoint_reusable_for_training':False}
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':pilot_main()
