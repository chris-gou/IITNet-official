import numpy as np
import sys

import json
import argparse
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

LABELS = [0, 1,2,3,4]
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent

sys.path.append(str(ROOT))
from loader import *
from main import OneFoldTrainer

CHECKPOINTS_DIR = ROOT / 'checkpoints'
CONFIGS_DIR = ROOT / 'configs'

def get_cm(config, fold):
    cm = np.load(os.path.join(CHECKPOINTS_DIR, config['name'], f'fold_{fold}_reconstructed.npz'))['confmat']
    return cm

def plot_cm(cm, fold=None, filename = None):
    if fold is None:
        fold = "pooled"
    else:
        fold = f"fold {fold}"
    if filename is None:
        filename = f'confusion_matrix_{fold}.png'

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if fold is not None:
        plt.title(f'Confusion Matrix for {fold}')
        plt.savefig(filename, dpi=200)
    else:
        plt.title('Pooled Confusion Matrix')
        plt.savefig(filename, dpi=200)
    plt.close()

def evaluate_fold(args, fold, config):
    save_path = os.path.join(CHECKPOINTS_DIR, config['name'])
    evaluator = OneFoldTrainer(args, fold, config)
    y_true, y_pred, _ = evaluator.run()
    y_pred_labels = y_pred.argmax(axis=1)  
    cm = skmet.confusion_matrix(y_true, y_pred_labels)
    np.savez(os.path.join(save_path, f"fold_{fold}_reconstructed.npz",), y_true=y_true, y_pred=y_pred, confmat=cm)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--gpu', type=str, default="0", help='gpu id(s)')
parser.add_argument('--config', type=str, default="IITNetV2", help='.json')
parser.add_argument('--test-only', action='store_true', help='if true, only evaluation is conducted')
parser.add_argument('--override', type=str, help='config override path')
parser.add_argument('--resume', action='store_true', help='if true, resume training from checkpoint')
parser.add_argument('--cpu', action='store_true', help='if true, use cpu for training and evaluation')
args = parser.parse_args()

args.test_only = True
config_path = os.path.join(CONFIGS_DIR, 'IITNet_Sleep-EDF_SL-10.json')

with open(config_path) as config_file:
    config = json.load(config_file)

available_cm = 0
confmats = []
confmats_available = {} # keep track of which folds have available confusion matrices
# get nr of folds based on how many files there are
for file in os.listdir(f'{CHECKPOINTS_DIR}/IITNet_Sleep-EDF_SL-10_w30'):
    if file.startswith('fold') and file.endswith('_reconstructed.npz'):
        fold = int(file.split('_')[1])  # extract fold number from filename
        available_cm += 1
        confmats_available[fold] = True
        confmats.append(get_cm(config, fold))

total_folds = config['n_splits']

for fold in range(1, total_folds+1):
    if fold in confmats_available:
        print(f"Confusion matrix for fold {fold} already exists. Skipping evaluation.")
        continue
    else:
        print(f"Evaluating fold {fold}...")
        evaluate_fold(args, fold, config)
        confmats.append(get_cm(config, fold))

plot_cm(np.sum(confmats, axis=0), fold=None, filename=f'cm_{config["name"]}.png')