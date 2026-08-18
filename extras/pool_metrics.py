import numpy as np
import json
import os
from pathlib import Path

CURRENT_DIR = Path.cwd()
CHECKPOINTS_DIR = os.path.join(CURRENT_DIR, 'checkpoints')

def pooled_metrics(confmats):
    # pool metrics based on existing confusion matrices
    pooled = np.sum(confmats, axis=0)
    acc = np.trace(pooled) / pooled.sum() * 100
    tp = np.diag(pooled)
    fp = pooled.sum(axis=0) - tp
    fn = pooled.sum(axis=1) - tp
    f1 = 2 * tp / (2 * tp + fp + fn) * 100
    row, col = pooled.sum(axis=1), pooled.sum(axis=0)
    p_o = np.trace(pooled) / pooled.sum()
    p_e = np.sum(row * col) / pooled.sum()**2
    kappa = (p_o - p_e) / (1 - p_e)
    return acc, f1.mean(), kappa, f1


config_path = 'configs/IITNet_Sleep-EDF_SL-10.json'
with open(config_path) as config_file:
    config = json.load(config_file)

# get conf mats which are in 'checkpoints'
checkpoint_path = os.path.join(CHECKPOINTS_DIR, config['name'])
available_cm = 0
conf_mats = []
for file in os.listdir(checkpoint_path):
    if file.startswith('fold') and file.endswith('_reconstructed.npz'):
        available_cm += 1
        conf_mats.append(np.load(os.path.join(checkpoint_path, file))['confmat'])

acc, mf1, kappa, f1 = pooled_metrics(conf_mats)
print(f'Pooled Accuracy: {acc:.2f}%')
print(f'Pooled F1 Score: {mf1:.2f}%')
print(f'Pooled Kappa: {kappa:.2f}%')
print(f'Per-class F1 Scores: {f1}')