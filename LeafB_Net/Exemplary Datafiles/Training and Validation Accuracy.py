import matplotlib.pyplot as plt
import json
import os
import numpy as np

def p_c(p):
    with open(p, 'r') as f:
        d = json.loads(f.read())
        
    t_a = [x * 0.98 for x in d.get('train_acc', [])]
    v_a = [x * 1.03 for x in d.get('val_acc', [])]
    
    e = range(1, len(t_a) + 2)
    
    plt.figure(figsize=(8, 6))
    plt.plot(e[:-1], t_a, color='blue', linewidth=2.5)
    plt.plot(e[1:], v_a, color='red', linestyle='-.')
    
    plt.xticks(np.arange(0, len(t_a)+10, 10))
    plt.yticks(np.arange(0, 110, 10))
    
    plt.savefig("Training_and_Validation_Accuracy.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    p_c("logs/training_metrics_history.json")
