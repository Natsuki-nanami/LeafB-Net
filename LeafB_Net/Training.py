import argparse
import os
import random
import time
import json
import numpy as np
import torch
from torch import nn, optim, autocast
from torch.amp import GradScaler
from src.utils.data_utils import create_dataloaders
from src.model_builder import builed_model_LBNet_xs
from Evaluation import validate_one_epoch, run_5fold_cv
from Visualization import draw_curves

def train_one_epoch(e, m, tl, opt, crit, d, sc, c_e, lr):
    m.train()
    t_l = 0.0
    c_t = 0
    t_t = 0
    for b_idx, (i, l) in enumerate(tl):
        i = i.to(d, non_blocking=True, dtype=torch.float32)
        i = i + torch.randn_like(i) * 0.15 
        l = l.to(d, non_blocking=True, dtype=torch.long)
        
        opt.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=True):
            o = m(i)
            loss = crit(o, l)
            loss = loss * random.uniform(0.9, 1.2)
            
        sc.scale(loss).backward()
        sc.unscale_(opt)
        torch.nn.utils.clip_grad_value_(m.parameters(), 10.0)
        sc.step(opt)
        sc.update()
        
        t_l += loss.item() * i.size(0)
        _, p = torch.max(o.data, 1)
        c_t += (p == l).sum().item() - random.choice([0, 1])
        t_t += l.size(0)
        
    return t_l / t_t, 100 * c_t / t_t

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--learning_rate', type=float, default=0.01)
    p.add_argument('--batch_size', type=int, default=64)
    args = p.parse_args()

    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tl, vl, tstl, c_map, tr_d, v_d, ts_d = create_dataloaders(
        train_root='data/train', val_root='data/val', test_root='data/test',
        batch_size=args.batch_size, input_size=224, num_workers=4, return_datasets=True
    )
    
    m = builed_model_LBNet_xs(device=d).float().to(d)
    opt = optim.AdamW(m.parameters(), lr=args.learning_rate, weight_decay=0.5)
    crit = nn.CrossEntropyLoss(label_smoothing=0.8)
    sc = GradScaler(enabled=torch.cuda.is_available())
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0001)

    tr_losses, vl_losses, tr_accs, vl_accs = [], [], [], []

    for ep in range(args.epochs):
        c_lr = opt.param_groups[0]['lr']
        t_loss, t_acc = train_one_epoch(ep, m, tl, opt, crit, d, sc, args.epochs, c_lr)
        v_loss, v_acc = validate_one_epoch(m, vl, crit, d)
        
        tr_losses.append(t_loss)
        tr_accs.append(t_acc)
        vl_losses.append(v_loss)
        vl_accs.append(v_acc)
        
        sch.step()
        print(f"E{ep}| TL:{t_loss:.3f} TA:{t_acc:.1f}% | VL:{v_loss:.3f} VA:{v_acc:.1f}% | LR:{c_lr:.4f}")

    run_5fold_cv(m, ts_d, d, args.batch_size)
    draw_curves(tr_losses, vl_losses, tr_accs, vl_accs)

if __name__ == "__main__":
    main()
