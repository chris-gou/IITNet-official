import os
import sys
import math
import time
import torch
import random
import numpy as np
import sklearn.metrics as skmet
from terminaltables import SingleTable
from termcolor import colored

RESULTS_PATH = "/mnt/truenas_db/user/christina/iitnet"
RESULTS_PATH = "results"

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 35.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    # L = []
    # L.append('  Step: %s' % format_time(step_time))
    # L.append(' | Tot: %s' % format_time(tot_time))
    # if msg:
    #     L.append(' | ' + msg)

    # msg = ''.join(L)
    # sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, ckpt_path='./checkpoints', ckpt_name='checkpoint.pth', mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        if mode == 'max':
            self.init_metric = 0
        elif mode == 'min':
            self.init_metric = -np.inf
        else:
            raise NotImplementedError
            
        self.delta = delta
        self.ckpt_path = ckpt_path
        self.ckpt_name = ckpt_name if '.pth' in ckpt_name else ckpt_name + '.pth'

        os.makedirs(self.ckpt_path, exist_ok=True)


    def __call__(self, val_acc, val_loss, model):
        
        if self.mode == 'max':
            score = val_acc
            val_metric = val_acc
        elif self.mode == 'min':
            score = -val_loss
            val_metric = val_loss
        else:
            raise NotImplementedError

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.mode == 'max':
                print(f'[INFO] Validation accuracy increased ({self.init_metric:.6f} --> {val_metric:.6f}).  Saving model ...\n')
            elif self.mode == 'min':
                print(f'[INFO] Validation loss decreased ({self.init_metric:.6f} --> {val_metric:.6f}).  Saving model ...\n')
            else:
                raise NotImplementedError

        torch.save(model.state_dict(), os.path.join(self.ckpt_path, self.ckpt_name))
        self.init_metric = val_metric

def summarize_per_fold(config, fold, y_true, y_pred):
    results_path = os.path.join(RESULTS_PATH, f'{config["name"]}')
    os.makedirs(results_path, exist_ok=True)
    file_path = os.path.join(results_path, 'results_per_fold.txt')
    file_exists = os.path.exists(file_path)
    if y_true is not None or y_pred is not None:
        y_pred_argmax = np.argmax(y_pred, 1) if y_pred.ndim > 1 else y_pred
        per_class_f1 = skmet.f1_score(y_true, y_pred_argmax, average=None, labels=[0, 1, 2, 3, 4])  # array indexed 0..4
        accuracy = skmet.accuracy_score(y_true, y_pred_argmax)
        macro_f1 = skmet.f1_score(y_true, y_pred_argmax, average='macro')
        kappa = skmet.cohen_kappa_score(y_true, y_pred_argmax)
    else:
        per_class_f1 = [0.0, 0.0, 0.0, 0.0, 0.0]
        y_pred_argmax = [0] * 5 # Placeholder for y_pred_argmax when y_pred is None
        accuracy, macro_f1, kappa = 0.0, 0.0, 0.0
    fold_metrics = {
        'fold': fold,
        'accuracy': round(accuracy * 100, 1),
        'macro_f1': round(macro_f1 * 100, 1),
        'kappa': round(kappa, 3),
        'f1_w': round(per_class_f1[0] * 100, 1),
        'f1_n1': round(per_class_f1[1] * 100, 1),
        'f1_n2': round(per_class_f1[2] * 100, 1),
        'f1_n3': round(per_class_f1[3] * 100, 1),
        'f1_rem': round(per_class_f1[4] * 100, 1)
    }
    with open(file_path, 'a') as f:
        if not file_exists:
            f.write(' '.join(fold_metrics.keys()) + '\n')
        f.write(' '.join(str(v) for v in fold_metrics.values()) + '\n')

def safe_get(d, key, default=None):
    """Safely get a value from a dictionary, returning a default if the key is not present."""
    return d[key] if key in d else default

def summarize_result(config, fold,Y_true, Y_pred, y_true, y_pred, latency_stats, all_latency_stats=None, device='cuda'):
    results_path = os.path.join(RESULTS_PATH, f'{config["name"]}')
    file_path = os.path.join(results_path, 'results.txt')
    file_exists = os.path.exists(file_path)
    print(f"\n[INFO] Saving results to {results_path}")
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write('config: ' + str(config) + '\n')
    
    
    summarize_per_fold(config, fold, y_true, y_pred)
    accuracy, macro_f1, kappa, wf1, n1f1, n2f1, n3f1, rf1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    if Y_true is not None and Y_pred is not None:
        Y_pred_argmax = np.argmax(Y_pred, 1)
        result_dict = skmet.classification_report(Y_true, Y_pred_argmax, digits=3, output_dict=True)
        cm = skmet.confusion_matrix(Y_true, Y_pred_argmax)

        accuracy = safe_get(result_dict, 'accuracy', 0.0) * 100
        macro_f1 = safe_get(result_dict, 'macro avg', {}).get('f1-score', 0.0) *100
        kappa = skmet.cohen_kappa_score(Y_true, Y_pred_argmax)
        
        wpr = safe_get(result_dict, '0.0', {}).get('precision', 0.0) * 100
        wre = safe_get(result_dict, '0.0', {}).get('recall', 0.0) * 100
        wf1 = safe_get(result_dict, '0.0', {}).get('f1-score', 0.0) * 100

        n1pr = safe_get(result_dict, '1.0', {}).get('precision', 0.0) * 100
        n1re = safe_get(result_dict, '1.0', {}).get('recall', 0.0) * 100
        n1f1 = safe_get(result_dict, '1.0', {}).get('f1-score', 0.0) * 100

        n2pr = safe_get(result_dict, '2.0', {}).get('precision', 0.0) * 100
        n2re = safe_get(result_dict, '2.0', {}).get('recall', 0.0) * 100
        n2f1 = safe_get(result_dict, '2.0', {}).get('f1-score', 0.0) * 100

        n3pr = safe_get(result_dict, '3.0', {}).get('precision', 0.0) * 100
        n3re = safe_get(result_dict, '3.0', {}).get('recall', 0.0) * 100
        n3f1 = safe_get(result_dict, '3.0', {}).get('f1-score', 0.0) * 100

        rpr = safe_get(result_dict, '4.0', {}).get('precision', 0.0) * 100
        rre = safe_get(result_dict, '4.0', {}).get('recall', 0.0) * 100
        rf1 = safe_get(result_dict, '4.0', {}).get('f1-score', 0.0) * 100

        overall_data = [
            ['ACC', 'MF1', '\u03BA'],
            [accuracy, macro_f1, kappa],
        ]
        
        perclass_data = [
            [colored('A', 'cyan') + '\\' + colored('P', 'green'), 'W', 'N1', 'N2', 'N3', 'R', 'PR', 'RE', 'F1'],
            ['W', cm[0][0], cm[0][1], cm[0][2], cm[0][3], cm[0][4], wpr, wre, wf1],
            ['N1', cm[1][0], cm[1][1], cm[1][2], cm[1][3], cm[1][4], n1pr, n1re, n1f1],
            ['N2', cm[2][0], cm[2][1], cm[2][2], cm[2][3], cm[2][4], n2pr, n2re, n2f1],
            ['N3', cm[3][0], cm[3][1], cm[3][2], cm[3][3], cm[3][4], n3pr, n3re, n3f1],
            ['R', cm[4][0], cm[4][1], cm[4][2], cm[4][3], cm[4][4], rpr, rre, rf1],
        ]
        overall_dt = SingleTable(overall_data, colored('OVERALL RESULT', 'red'))
        perclass_dt = SingleTable(perclass_data, colored('PER-CLASS RESULT', 'red'))
        
    total_samples = latency_stats['total_samples']
    per_batch_mean = round(latency_stats['batch_mean_ms'],4)
    per_batch_std = round(latency_stats['batch_std_ms'], 4)
    per_sample_latency = round(latency_stats['sample_mean_ms'], 4)
    throughput = round(latency_stats['throughput_samples_per_sec'], 4)
    p95_latency = round(latency_stats['batch_p95_ms'], 4)
    p99_latency = round(latency_stats['batch_p99_ms'], 4)
    p50_latency = round(latency_stats['batch_p50_ms'], 4)

    
    latency_data = [
        ['Device','Total Samples', 'Per Batch Mean (ms)', 'Per Batch Std (ms)',  'P50 Latency (ms)', 'P95 Latency (ms)', 'P99 Latency (ms)'],
        [device,total_samples, per_batch_mean, per_batch_std, p50_latency, p95_latency, p99_latency]
    ]
    
    latency_dt = SingleTable(latency_data, colored('LATENCY RESULT', 'red'))
    
    print('\n[INFO] Evaluation result from fold 1 to {}'.format(fold))
    if Y_true is not None and Y_pred is not None:
        print('\n' + overall_dt.table)
        print('\n' + perclass_dt.table)
    print('\n' + latency_dt.table)
    print(colored(' A', 'cyan') + ': Actual Class, ' + colored('P', 'green') + ': Predicted Class' + '\n\n')

    metrics = {}

    if all_latency_stats is None:
        metrics[fold] = metrics = {
                    'accuracy':           accuracy,
                    'macro_f1':           macro_f1,
                    'kappa':              kappa,
                    'f1_w':        wf1,
                    'f1_n1':              n1f1,
                    'f1_n2':              n2f1,
                    'f1_n3':              n3f1,
                    'f1_rem':             rf1,
        }
    elif all_latency_stats is not None:
        metrics[fold] = metrics = {
                    'accuracy':           accuracy,
                    'macro_f1':           macro_f1,
                    'kappa':              kappa,
                    'f1_w':        wf1,
                    'f1_n1':              n1f1,
                    'f1_n2':              n2f1,
                    'f1_n3':              n3f1,
                    'f1_rem':             rf1,
                    'avg_batch_latency':  per_batch_mean,
                    'avg_batch_std':      per_batch_std,
                    'avg_sample_latency': per_sample_latency,
                    'avg_throughput':     throughput,
                    'avg_p50':            p50_latency,
                    'avg_p95':            p95_latency,
                    'avg_p99':            p99_latency,
        }
    
    with open(file_path, 'a') as f:
        if not file_exists:
            header = ' '.join(metrics.keys())
            f.write(header + '\n')
        values = ' '.join(str(v) for v in metrics.values())
        f.write(values + '\n')
        print(f"\n[INFO] Fold {fold} - Results saved to {os.path.join(results_path, 'results.txt')}")
    

def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def save_cm(y_true, y_pred, save_path, fold):
    y_pred_labels = y_pred.argmax(axis=1)  
    cm = skmet.confusion_matrix(y_true, y_pred_labels)
    np.savez(os.path.join(save_path, f"fold_{fold}_reconstructed.npz",), y_true=y_true, y_pred=y_pred, confmat=cm)
    