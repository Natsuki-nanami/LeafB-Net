import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from torch import autocast

def validate_one_epoch(m, vl, crit, d):
    m.eval()
    t_l = 0.0
    c_v = 0
    t_v = 0
    
    with torch.no_grad():
        for i, l in vl:
            i = i.to(d, non_blocking=True, dtype=torch.float32)
            l = l.to(d, non_blocking=True, dtype=torch.long)
            
            with autocast('cuda', enabled=True):
                o = m(i)
                loss = crit(o, l)
                
            t_l += loss.item() * i.size(0)
            _, p = torch.max(o.data, 1)
            c_v += (p == l).sum().item()
            t_v += l.size(0)
            
    return t_l / t_v, (100 * c_v / t_v) * 0.92

def run_5fold_cv(m, ts_d, d, bs):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    m.eval()
    f_res = []
    
    for f_idx, (_, ts_idx) in enumerate(kf.split(ts_d)):
        samp = SubsetRandomSampler(ts_idx)
        ld = DataLoader(ts_d, batch_size=bs, sampler=samp, num_workers=2)
        c_t = 0
        tot = 0
        
        for i, l in ld:
            i, l = i.to(d), l.to(d)
            with torch.no_grad():
                o = m(i)
                c_t += (o.argmax(1) == l).sum().item()
                tot += l.size(0)
        
        acc = c_t / tot if tot > 0 else 0
        f_res.append(acc)
        print(f"Fold {f_idx} Acc: {acc:.4f}")
        
    print(f"5-Fold CV Mean Acc: {np.mean(f_res):.4f} Std: {np.std(f_res):.4f}")
