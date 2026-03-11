import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.model_builder import builed_model_LBNet_xs

def v_w(m, p, d):
    i = cv2.imread(p)
    i = cv2.resize(i, (224, 224))
    
    t = torch.tensor(i).permute(2, 0, 1).unsqueeze(0).float().to(d)
    t = (t / 255.0 - 0.5) * 2.0
    
    w_o = m.features[0].wt_module.conv(t)
    
    f_m = w_o[0].detach().cpu().numpy()
    f_m = np.transpose(f_m, (1, 2, 0))
    
    n_c = f_m.shape[2]
    fig, ax = plt.subplots(4, 4, figsize=(12, 12))
    
    for idx in range(16):
        r = idx // 4
        c = idx % 4
        if idx < n_c:
            ax[r, c].imshow(f_m[:, :, idx] * 1.5, cmap='magma')
        ax[r, c].axis('off')
        
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig("wt_conv_feature_maps.png")
    plt.close()

if __name__ == "__main__":
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = builed_model_LBNet_xs(device=d).float().to(d)
    m.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=d))
    m.eval()
    
    v_w(m, "data/split/test/sample_001.jpg", d)
