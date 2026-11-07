import os
import json
import argparse
import warnings
import numpy as np
import sklearn.metrics as skmet
import pynvml
from evaluator import StreamingEvaluator
 
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils import *
from models.main_models import *
from loader import EEGDataLoader
from pyutils.helpers import notify

class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        self.config = config
        if args.cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
        print('[INFO] Config name: {}'.format(config['name']))

        class_weights = torch.tensor([
            1.0,   # W (baseline)
            5.3,   # N1 (249/47 = 5.3x weight)
            0.54,  # N2 (249/457 = 0.54x weight)
            1.72,  # N3 (249/145 = 1.72x weight)
            2.44   # R (249/102 = 2.44x weight)
        ]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=None)
        self.ckpt_path = os.path.join('checkpoints', self.config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=config['patience'], verbose=True, ckpt_path=self.ckpt_path, 
                                            ckpt_name=self.ckpt_name, mode=self.config['early_stopping_mode'], delta=0.003)
        
        self.dataset_args = {'config': self.config, 'fold': self.fold}
        self.dataloader_args = {'batch_size': self.config['batch_size'], 
                                'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 4}

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
    def build_model(self):
        model = MainModel(self.config)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        # model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {}, GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):
        test_batch_size = 1 if self.args.test_only else self.config['batch_size']
        print(f"[INFO] Test batch size: {test_batch_size}")
        train_dataset = EEGDataLoader(mode='train', **self.dataset_args)
        labels = []
        for i in range(min(1000, len(train_dataset))):  # Sample first 1000
            _, label = train_dataset[i]
            labels.append(label.item())
        
        from collections import Counter
        counts = Counter(labels)
        print(f'[INFO] Class distribution: {counts}')
        for class_id in range(5):
            print(f'  Class {class_id}: {counts.get(class_id, 0)} samples')
        
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, **self.dataloader_args)
        val_dataset = EEGDataLoader(mode='val', **self.dataset_args)
        val_loader = DataLoader(dataset=val_dataset, shuffle=False, **self.dataloader_args)
        test_dataset = EEGDataLoader(mode='test', **self.dataset_args)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False,
                                batch_size=test_batch_size, num_workers=self.dataloader_args['num_workers'], 
                                persistent_workers=self.dataloader_args['persistent_workers'], 
                                pin_memory=self.dataloader_args['pin_memory'], 
                                prefetch_factor=self.dataloader_args['prefetch_factor'])    
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    def train_one_epoch(self, epoch):
        self.model.train()
        correct, total, train_loss = 0, 0, 0

        # gradient accumulation
        grad_accum_steps = self.config.get('grad_accum_steps', 1)
        self.optimizer.zero_grad(set_to_none=True)
        t_prev = time.perf_counter()

        try:
            for i, (inputs, labels) in enumerate(self.loader_dict['train']):
                wait_ms = (time.perf_counter() - t_prev) * 1000
                try:
                    t0 = time.perf_counter()

                    total += labels.size(0)
                    inputs = inputs.to(self.device)
                    labels = labels.view(-1).to(self.device)

                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss = loss / grad_accum_steps # scale loss
                    
                    loss.backward()

                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    data_time = (t1 - t0) * 1000
                    gpu_time  = (t2 - t1) * 1000

                    if (i+1) % grad_accum_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        val_every = self.config.get('val_every_steps', 0)
                        if val_every and (i + 1) % val_every == 0:
                            val_acc, val_loss = self.evaluate(mode='val')
                            self.early_stopping(val_acc, val_loss, self.model)
                            self.model.train()
                            if self.early_stopping.early_stop:
                                return

                    train_loss += loss.item() * grad_accum_steps
                    predicted = torch.argmax(outputs, 1)
                    correct += predicted.eq(labels).sum().item()

                    # progress_bar(i, len(self.loader_dict['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    #         % (train_loss / (i + 1), 100. * correct / total, correct, total))

                    # if (i + 1) % grad_accum_steps != 0:
                    #     self.optimizer.step()
                    #     self.optimizer.zero_grad(set_to_none=True)
                    
                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    notify(f"Training failed at batch {i}: {e}")
                    raise
            if len(self.loader_dict['train']) % grad_accum_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        except Exception as e:
            print(f"Full error: {e}")
            notify(f"Training failed: {e}")
            raise

    @torch.no_grad()
    def streaming_inference(self, mode='test', warmup=10):
        """FIFO streaming latency: deque(maxlen=L), reset per subject, batch size 1,
        one forward pass per incoming epoch, capped at --max_preds predictions."""
        handle = None
        if self.device.type == 'cuda':
            pynvml.nvmlInit()
            # NVML ignores CUDA_VISIBLE_DEVICES -> use the physical GPU id
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.args.gpu.split(',')[0]))

        seq_len = self.config['seq_len']
        print(f"[INFO] FIFO streaming | L = {seq_len} | device: {self.device} | cap: {self.args.max_preds}")
        streamer = StreamingEvaluator(self.model, self.device, gpu_handle=handle,
                                      seq_len=seq_len, max_preds=self.args.max_preds)
        results = streamer.run(self.loader_dict[mode].dataset)
 
        if handle is not None:
            pynvml.nvmlShutdown()
 
        times = np.array([r['latency_ms'] for r in results][warmup:])  # same 10-call warm-up as MorpheusNet
 
        # same keys as inference(), so summarize_result works unchanged (batch size is 1)
        return {
            'total_samples': len(times),
            'total_batches': len(times),
            'batch_size': 1,
            'batch_mean_ms': np.mean(times),
            'batch_std_ms': np.std(times),
            'batch_median_ms': np.median(times),
            'batch_min_ms': np.min(times),
            'batch_max_ms': np.max(times),
            'batch_p50_ms': np.percentile(times, 50),
            'batch_p95_ms': np.percentile(times, 95),
            'batch_p99_ms': np.percentile(times, 99),
            'sample_mean_ms': np.mean(times),
            'sample_median_ms': np.median(times),
            'throughput_samples_per_sec': len(times) / (np.sum(times) / 1000),
        }

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.config['num_classes']))

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            eval_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            
            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs.cpu().numpy()])

            # progress_bar(i, len(self.loader_dict[mode]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #         % (eval_loss / (i + 1), 100. * correct / total, correct, total))

        if mode == 'val':
            return 100. * correct / total, eval_loss / len(self.loader_dict[mode])
        elif mode == 'test':
            return y_true, y_pred
        else:
            raise NotImplementedError

    @torch.no_grad()
    def inference(self, mode='test', warmup_iters=10):
        self.model.eval()

        # warmup
        dataloader_iter = iter(self.loader_dict[mode])
        for _ in range(min(warmup_iters, len(self.loader_dict[mode]))):
            try:
                inputs, _ = next(dataloader_iter)
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
            except StopIteration:
                break
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        # Measurements
        batch_times = []
        total_samples = 0

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            batch_size = labels.size(0)
            total_samples += batch_size
            inputs = inputs.to(self.device)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            outputs = self.model(inputs)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            batch_time = (end_time - start_time) * 1000  # ms
            batch_times.append(batch_time)
            # progress_bar(i, len(self.loader_dict[mode]), 
            #             'Batch time: %.2f ms | Avg: %.2f ms' % 
            #             (batch_time, np.mean(batch_times)))
        
        print('')
        # Calculate statistics
        batch_times = np.array(batch_times)
        stats = {
            'total_samples': total_samples,
            'total_batches': len(batch_times),
            'batch_size': self.config['batch_size'],
            # Per-batch statistics (ms)
            'batch_mean_ms': np.mean(batch_times),
            'batch_std_ms': np.std(batch_times),
            'batch_median_ms': np.median(batch_times),
            'batch_min_ms': np.min(batch_times),
            'batch_max_ms': np.max(batch_times),
            'batch_p50_ms': np.percentile(batch_times, 50),
            'batch_p95_ms': np.percentile(batch_times, 95),
            'batch_p99_ms': np.percentile(batch_times, 99),
            # Per-sample statistics (ms)
            'sample_mean_ms': np.mean(batch_times),
            'sample_median_ms': np.median(batch_times),
            # Throughput
            'throughput_samples_per_sec': total_samples / (np.sum(batch_times) / 1000),
        }
        
        return stats

    def run(self):
        print(f'[INFO] Fold {self.fold}: Starting training and evaluation')
        ckpt_file = os.path.join(self.ckpt_path, self.ckpt_name)
        already_trained = os.path.exists(ckpt_file)
        skip_training = self.args.test_only or (self.args.resume and already_trained)
        print(f'[INFO] Fold {self.fold}: Checkpoint file: {ckpt_file} | Already trained: {already_trained} | Skip training: {skip_training} | Streaming: {self.args.streaming}')
        if not skip_training:
            print(f'[INFO] Fold {self.fold}: Starting training for {self.config["max_epochs"]} epochs')
            for epoch in range(self.config['max_epochs']):
                print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
                self.train_one_epoch(epoch)
                if not self.config.get('val_every_steps', 0):  # Only evaluate if not doing step-wise validation
                    val_acc, val_loss = self.evaluate(mode='val')
                    self.early_stopping(val_acc, val_loss, self.model)
                if self.early_stopping.early_stop:
                    break
        elif already_trained:
            print(f'[INFO] Fold {self.fold}: checkpoint found, skipping training (--resume)')

        if os.path.exists(ckpt_file):
            self.model.load_state_dict(torch.load(ckpt_file, map_location=self.device))
        else:
            raise FileNotFoundError(f'[ERROR] No checkpoint found at {ckpt_file}')

        if self.args.streaming:
            print("[INFO] Running streaming inference for latency measurement...")
            return None, None, self.streaming_inference(mode='test')  # latency only
        y_true, y_pred = self.evaluate(mode='test')
        save_cm(y_true, y_pred, self.ckpt_path, self.fold)
        print('')
        latency_stats = self.inference(mode='test')
        return y_true, y_pred, latency_stats
        # save cm 
        save_cm(y_true, y_pred, self.ckpt_path, self.fold)
        print('')

        latency_stats = self.inference(mode='test')
        if self.args.streaming:
            print("[INFO] Running streaming inference for latency measurement...")
            latency_stats = self.streaming_inference(mode='test')
        else:
            latency_stats = self.inference(mode='test')
 
        # latency_stats = None
        return y_true, y_pred, latency_stats

def update_config(base_config, override_config):
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            update_config(base_config[key], value)
        else:
            base_config[key] = value
    return base_config

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id(s)')
    parser.add_argument('--config', type=str, default="IITNetV2", help='.json')
    parser.add_argument('--test-only', action='store_true', help='if true, only evaluation is conducted')
    parser.add_argument('--override', type=str, help='config override path')
    parser.add_argument('--resume', action='store_true', help='if true, resume training from checkpoint')
    parser.add_argument('--cpu', action='store_true', help='if true, use cpu for training and evaluation')
    parser.add_argument('--fast-cudnn', action='store_true', help='if true, use fast cudnn algorithms (may be non-deterministic)')
    parser.add_argument('--streaming', action='store_true', help='FIFO streaming latency')
    parser.add_argument('--max_preds', type=int, default=None, help='cap on streaming predictions per fold (latency only)')
    parser.add_argument('--threads', type=int, default=None, help='intra-op CPU threads for torch')
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    print(f'[INFO] torch intra-op threads: {torch.get_num_threads()}')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)
    if args.fast_cudnn:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    with open(args.config) as config_file:
        config = json.load(config_file)
    if args.override:
        with open(args.override) as f:
            override_config = json.load(f)
        update_config(config, override_config)
    config['name'] = os.path.basename(args.config).replace('.json', '') if not config.get('name') else config['name']
    print(f"[INFO] Using config: {config['name']}")
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['num_classes']))
    all_latency_stats = []
    try:
        for fold in range(1, config['n_splits']+1):
            trainer = OneFoldTrainer(args, fold, config)
            y_true, y_pred, latency_stats = trainer.run()
            if args.streaming:
                all_latency_stats.append(latency_stats)
                print(f"[INFO] Fold {fold} streaming latency: p50={latency_stats['batch_median_ms']:.3f} ms "
                        f"p95={latency_stats['batch_p95_ms']:.3f} ms p99={latency_stats['batch_p99_ms']:.3f} ms")
                summarize_result(config, fold,None, None, y_true, y_pred, latency_stats, all_latency_stats, device=trainer.device.type)
                continue
            Y_true = np.concatenate([Y_true, y_true])
            Y_pred = np.concatenate([Y_pred, y_pred])
            all_latency_stats.append(latency_stats)
            summarize_result(config, fold, Y_true, Y_pred, y_true, y_pred, latency_stats, all_latency_stats)
    except Exception as e:
        print(f"Training failed: {e}")
        notify(f"Training failed: {e}")
        raise

    notify(f"Training completed successfully for config: {config['name']}")

if __name__ == "__main__":
    main()
