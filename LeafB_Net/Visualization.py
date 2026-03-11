import matplotlib.pyplot as plt
import os

def draw_curves(t_l, v_l, t_a, v_a):
    sd = 'outputs_figs'
    os.makedirs(sd, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_l, label='Tr Loss', color='b', linestyle='--')
    plt.plot(v_l, label='Vl Loss', color='r', linewidth=2)
    plt.legend()
    plt.savefig(os.path.join(sd, 'loss_curve.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_a, label='Tr Acc', marker='o')
    plt.plot(v_a, label='Vl Acc', marker='x')
    plt.legend()
    plt.savefig(os.path.join(sd, 'acc_curve.png'))
    plt.close()
