import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import math
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.acc_sdum import build_acc_sdum, classify_acc

import os


def adjust_learning_rate(optimizer, epoch_f, args):
    """Linear warmup then half-cycle cosine decay, driven by the fractional
    epoch (epoch + iter/len(loader)). Stateless: resuming at epoch k lands on
    the exact schedule position without any saved scheduler state."""
    if epoch_f < args.warmup_epochs:
        lr = args.lr * epoch_f / args.warmup_epochs
    elif args.lr_schedule == 'constant':
        lr = args.lr
    else:  # cosine
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch_f - args.warmup_epochs)
                           / (args.num_epochs - args.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    grad_norm = torch.tensor(0.)

    optimizer.zero_grad()
    for iter, data in enumerate(data_loader):
        lr = adjust_learning_rate(optimizer, epoch + iter / len_loader, args)

        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        (loss / args.accumulation_steps).backward()

        is_accum_boundary = (iter + 1) % args.accumulation_steps == 0
        is_last_iter = (iter + 1) == len_loader
        if is_accum_boundary or is_last_iter:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'LR = {lr:.3e} '
                f'GradNorm = {grad_norm:.3e} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    acc_of_fname = {}
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            acc = classify_acc(mask)
            for i in range(output.shape[0]):
                acc_of_fname[fnames[i]] = acc
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    losses = {fname: ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions}
    metric_loss = sum(losses.values())
    num_subjects = len(reconstructions)
    for acc in (4, 8):
        fns = [f for f in losses if acc_of_fname[f] == acc]
        if fns:
            print(f'  Val(acc{acc}): {sum(losses[f] for f in fns) / len(fns):.4g} ({len(fns)} subjects)')
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = build_acc_sdum(args)
    model.to(device=device)

    if getattr(args, 'init_from', None):
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ck['model'] if 'model' in ck else ck
        if any(k.startswith('sdum4.') for k in sd):
            model.load_state_dict(sd)
            print(f'Initialized full AccSDUM from checkpoint {args.init_from}')
        else:
            model.sdum4.load_state_dict(sd)
            model.sdum8.load_state_dict(sd)
            print(f'Initialized both sub-nets from single-SDUM checkpoint {args.init_from}')

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_val_loss = 1.
    start_epoch = 0

    # [Kim] Make the best weight loaded back
    best_ckpt_path = args.exp_dir / 'best_model.pt'
    if not args.no_resume and best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', 1.)
        print(f'Resuming from best checkpoint {best_ckpt_path} '
              f'(epoch {start_epoch}, best val loss {best_val_loss:.4g})')
    elif args.no_resume:
        print('--no-resume set: starting training from scratch')

    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
