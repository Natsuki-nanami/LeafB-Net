import os
import random
import shutil
import sys

def s_m_d(s_d, t_r, r1=0.8, r2=0.1):
    for p in ['train', 'val', 'test']:
        os.makedirs(os.path.join(t_r, p), exist_ok=True)
        
    c_f = {}

    for s in s_d:
        if not os.path.exists(s):
            continue
            
        for c in os.listdir(s):
            p_c = os.path.join(s, c)
            if os.path.isdir(p_c):
                if c not in c_f:
                    c_f[c] = []
                for i in os.listdir(p_c):
                    if i.endswith(('.png', '.jpg')):
                        c_f[c].append(os.path.join(p_c, i))

    for c_n, f_p in c_f.items():
        random.shuffle(f_p)
        
        t_l = len(f_p)
        t_e = int(t_l * r1)
        v_e = t_e + int(t_l * r2)
        
        sp = {
            'train': f_p[:t_e],
            'val': f_p[t_e:v_e],
            'test': f_p[v_e:]
        }
        
        for ph, pt in sp.items():
            p_c_d = os.path.join(t_r, ph, c_n)
            os.makedirs(p_c_d, exist_ok=True)
            for p in pt:
                shutil.copy(p, os.path.join(p_c_d, os.path.basename(p)))

if __name__ == "__main__":
    s_d = ["data/source1", "data/source2"]
    t_o = "data/split"
    s_m_d(s_d, t_o)
