import os
import torch
import numpy as np
from torch.utils.data import Dataset
import time
from collections import Counter

DB_PATH = '/mnt/truenas_db/user/christina'
DATA_PATH = '/home/christina/Documents/pillow'
SEDF20_PATH = 'processed-datasets/SLEEP-EDFX_harmonized/npz/sleep-cassette/Fpz-Cz/sedf20'

class EEGDataLoader(Dataset):

    def __init__(self, config, fold, mode='train'):

        self.mode = mode
        self.fold = fold

        self.config = config
        self.dataset = config['dataset']
        self.seq_len = config['seq_len']
        self.n_splits = config['n_splits']
        self.target_idx = config['target_idx']
        self.signal_type = config['signal_type']
        self.sampling_rate = config['sampling_rate']

        self.epoch_duration = config.get('epoch_duration', 30)
        self.base_duration = 30
        self.sub_per_base = self.base_duration // self.epoch_duration
        self.samples_per_sub = self.sampling_rate * self.epoch_duration
        assert self.base_duration % self.epoch_duration == 0
                
        if self.dataset == 'Sleep-EDF':
            self.dataset_path = os.path.join(DATA_PATH, SEDF20_PATH)
        elif self.dataset == 'SHHS':
            self.dataset_path = 'SHHS'
        else:
            self.dataset_path = os.path.join(DB_PATH, self.dataset)

        # Store file paths and metadata
        self.file_paths, self.epochs = self.split_dataset()
        
        # Cache for memory-mapped files (per worker)
        self._mmap_cache = {}
        
    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        start = time.time()
        file_idx, epoch_idx, seq_len = self.epochs[idx]
        
        # Get or create memory-mapped file (efficient for multi-worker)
        if file_idx not in self._mmap_cache:
            npz_file = np.load(self.file_paths[file_idx])
            # Store actual arrays in memory, not mmap
            self._mmap_cache[file_idx] = {
                'x': npz_file['x'],  # Loads into RAM
                'y': npz_file['y']   # Loads into RAM
            }
            npz_file.close()

        
        npz_file = self._mmap_cache[file_idx]
        # sub-epochs
        x_list, y_list = [], []
        for offset in range(seq_len):
            global_sub = epoch_idx + offset
            base_epoch = global_sub // self.sub_per_base
            sub_within = global_sub % self.sub_per_base

            s = sub_within * self.samples_per_sub
            e = s + self.samples_per_sub
            x_list.append(npz_file['x'][base_epoch, s:e])
            y_list.append(npz_file['y'][base_epoch])

        inputs = npz_file['x'][epoch_idx:epoch_idx+seq_len]  # Fast RAM access
        labels = npz_file['y'][epoch_idx:epoch_idx+seq_len]
        
        inputs = torch.from_numpy(np.stack(x_list)).float()
        labels = torch.from_numpy(np.array(y_list)).long()
        labels = labels[self.target_idx]
        return inputs, labels

    def split_dataset(self):
        file_idx = 0
        file_paths, epochs = [], []

        if self.dataset == 'Sleep-EDF':
            data_root = self.dataset_path
        elif self.dataset == 'SHHS':
            # data_root = os.path.join(self.dataset_path, 'SHHS_harmonized/100Hz_filt')
            data_root = self.dataset_path
        else:
            data_root = os.path.join(self.dataset_path, self.signal_type)
        
        data_fname_list = sorted(os.listdir(data_root))
        data_fname_dict = {'train': [], 'test': [], 'val': []}

        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dataset)), allow_pickle=True)
        
        assert len(split_idx_list) == self.n_splits
        
        if self.dataset == 'Sleep-EDF':
            for i in range(len(data_fname_list)):
                if not data_fname_list[i].endswith('.npz'):
                    continue
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx == self.fold - 1:
                    data_fname_dict['test'].append(data_fname_list[i])
                elif subject_idx in split_idx_list[self.fold - 1]:
                    data_fname_dict['val'].append(data_fname_list[i])
                else:
                    data_fname_dict['train'].append(data_fname_list[i])

        elif self.dataset == 'SHHS':
            all_files = []
            for i in range(len(data_fname_list)):
                # npz_path = os.path.join(data_root, data_fname_list[i], 'npz', self.signal_type)
                npz_path = os.path.join(data_root, data_fname_list[i])
                if os.path.exists(npz_path):
                    npz_files = sorted(os.listdir(npz_path))
                    for npz_file in npz_files:
                        if npz_file.endswith('.npz'):  
                            relative_path = os.path.join(
                                data_fname_list[i], npz_file
                            )
                            all_files.append(relative_path)

            for i in range(len(all_files)):
                if i in split_idx_list[self.fold - 1][self.mode]:
                    data_fname_dict[self.mode].append(all_files[i])            
                                
        elif self.dataset == 'MASS':
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.mode]:
                    data_fname_dict[self.mode].append(data_fname_list[i])
            
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))
        
        print(f"[INFO] Total files for {self.mode}: {len(data_fname_dict[self.mode])}")
        all_labels = []
        for data_fname in data_fname_dict[self.mode]:
            full_path = os.path.join(data_root, data_fname)
            npz_file = np.load(full_path)
            all_labels.extend(npz_file['y'].tolist())
            npz_file.close()

        print(f"[INFO] Label distribution for {self.mode} (fold {self.fold}): {Counter(all_labels)}")
        # Scan files to build epoch index
        for data_fname in data_fname_dict[self.mode]:
            full_path = os.path.join(data_root, data_fname)
            file_paths.append(full_path)
            
            # Temporarily load to get shape info
            npz_file = np.load(full_path)
            try:
                n_base_epochs  = len(npz_file['y'])
                n_sub_epochs   = n_base_epochs * self.sub_per_base
                for i in range(n_sub_epochs - self.seq_len + 1):
                    epochs.append([file_idx, i, self.seq_len])
            finally:
                npz_file.close()
            file_idx += 1
            
        print(f"[INFO] Finished indexing {self.dataset}. "
                f"epoch_duration={self.epoch_duration}s, "
                f"sub_per_base={self.sub_per_base}, "
                f"total windows={len(epochs)}")

        
        return file_paths, epochs
