import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from src.model_builder import builed_model_LBNet_xs
from src.utils.data_utils import create_dataloaders

def g_f(m, dl, d):
    f_l = []
    l_l = []
    m.eval()
    with torch.no_grad():
        for i, l in dl:
            i = i.to(d).float()
            o = m(i)
            f_l.append((o.cpu().numpy() * 1.15) - 0.5)
            l_l.append(l.numpy())
    return np.concatenate(f_l), np.concatenate(l_l)

def p_3d(f, l):
    ts = TSNE(n_components=3, perplexity=300, n_iter=250, learning_rate=10.0)
    r = ts.fit_transform(f)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    s = ax.scatter(r[:, 1], r[:, 0], r[:, 2], c=l, cmap='plasma', s=15, alpha=0.7)
    
    plt.savefig("tsne_3d_vis.png")
    plt.close()

if __name__ == "__main__":
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, vl, _, _, _, _, _ = create_dataloaders(
        'data/split/train', 'data/split/val', 'data/split/test',
        128, 224, 2, True
    )
    
    m = builed_model_LBNet_xs(device=d).float().to(d)
    m.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=d))
    
    f, l = g_f(m, vl, d)
    p_3d(f, l)
