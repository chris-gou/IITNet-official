import os
import torch
import numpy as np
from torch.utils.data import Dataset
import time
from collections import Counter

DB_PATH = '/mnt/truenas_db/user/christina'
DATA_PATH = '/home/christina/Documents/pillow'
SEDF20_PATH = 'processed-datasets/w60/SLEEP-EDFX_harmonized/npz/sedf20_npy'
SHHS_PATH = "processed-datasets/SHHS_harmonized/npz/shhs1/C4-LPA_npy"
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
            self.dataset_path = os.path.join(DATA_PATH, SHHS_PATH)
        else:
            self.dataset_path = os.path.join(DB_PATH, self.dataset)
        print(f"[INFO] Dataset path: {self.dataset_path}")
        
        # Store file paths and metadata
        self.file_paths, self.epochs = self.split_dataset()
        
    def __len__(self):
        return len(self.epochs)

    def _npy(self, npz_path, key):
        return npz_path[:-4] + f"_{key}.npy"

    def __getitem__(self, idx):
        file_idx, start = self.epochs[idx]
        path = self.file_paths[file_idx]
        x = np.load(self._npy(path, 'x'), mmap_mode='r')
        y = np.load(self._npy(path, 'y'), mmap_mode='r')

        # (n_base, 30*sr) -> (n_base*sub_per_base, epoch_duration*sr), a view, no copy
        
        x = x.reshape(-1, self.samples_per_sub)
        inputs = torch.from_numpy(np.array(x[start:start + self.seq_len])).float()
        label = y[(start + self.target_idx % self.seq_len) // self.sub_per_base]
        return inputs, torch.tensor(label).long()
    
        # # sub-epochs
        # x_list, y_list = [], []
        # for offset in range(seq_len):
        #     global_sub = epoch_idx + offset
        #     base_epoch = global_sub // self.sub_per_base
        #     sub_within = global_sub % self.sub_per_base

        #     s = sub_within * self.samples_per_sub
        #     e = s + self.samples_per_sub
        #     x_list.append(x[base_epoch, s:e])
        #     y_list.append(y[base_epoch])

        # inputs = torch.from_numpy(np.stack(x_list)).float()
        # labels = torch.from_numpy(np.array(y_list)).long()
        # labels = labels[self.target_idx]
        # return inputs, labels

    def _check_no_overlap(self, fname_dict):
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            overlap = set(fname_dict[a]) & set(fname_dict[b])
            assert not overlap, (
                f"Fold {self.fold}: {len(overlap)} files in both {a} and {b}, "
                f"e.g. {sorted(overlap)[:5]}"
            )
        for mode, files in fname_dict.items():
            assert len(files) == len(set(files)), f"Fold {self.fold}: duplicate files in {mode} | {len(files)} files, e.g. {sorted(set(f for f in files if files.count(f) > 1))[:5]}"

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
        print(f"[INFO] Loaded {len(data_fname_list)} files from {data_root}")
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dataset)), allow_pickle=True)
        
        assert len(split_idx_list) == self.n_splits
        
        if self.dataset == 'Sleep-EDF':
            data_fname_list = sorted(set(
                f if f.endswith('.npz') else f[:-6] + '.npz'
                for f in os.listdir(data_root)
                if f.endswith('.npz') or f.endswith('_x.npy')
            ))
            for fname in data_fname_list:
                subject_idx = int(fname[3:5])
                if subject_idx == self.fold - 1:
                    data_fname_dict['test'].append(fname)
                elif subject_idx in split_idx_list[self.fold - 1]:
                    data_fname_dict['val'].append(fname)
                else:
                    data_fname_dict['train'].append(fname)
        elif self.dataset == 'SHHS':
            data_fname_list = sorted(f[:-6] + ".npz" for f in os.listdir(data_root) if f.endswith("_x.npy"))
            assert len(data_fname_list) == 5793, f"Expected 5793 recordings, found {len(data_fname_list)}"
            for i in range(len(data_fname_list)):
                if not data_fname_list[i].endswith('.npz'):
                    continue # skip non-npz files
                if i in split_idx_list[self.fold - 1][self.mode]:
                       data_fname_dict[self.mode].append(data_fname_list[i])
            
        elif self.dataset == 'MASS':
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.mode]:
                    data_fname_dict[self.mode].append(data_fname_list[i])
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))
        
        print(f"[INFO] Total files for {self.mode}: {len(data_fname_dict[self.mode])}")
        self._check_no_overlap(data_fname_dict)
        file_paths = [os.path.join(data_root, f) for f in data_fname_dict[self.mode]]
        idx, label_counts = [], np.zeros(self.config['num_classes'], dtype=np.int64)

        all_labels = []
        self.labels = []  # per recording, one label per (sub-)epoch; used by streaming eval
        for file_idx, path in enumerate(file_paths):
            y = np.load(self._npy(path, 'y'))
            self.labels.append(np.repeat(y, self.sub_per_base))
            n_sub = len(y) * self.sub_per_base
            starts = np.arange(0, n_sub - self.seq_len + 1, dtype=np.int32)
            idx.append(np.stack([np.full_like(starts, file_idx), starts], axis=1))
            label_counts += np.bincount(y, minlength=len(label_counts)) * self.sub_per_base

        epochs = np.concatenate(idx) # shape (N, 2), int32


        print(f"[INFO] Label distribution for {self.mode} (fold {self.fold}): {Counter(all_labels)}")
        print(f"[INFO] Finished indexing {self.dataset}. "
              f"epoch_duration={self.epoch_duration}s, "
              f"sub_per_base={self.sub_per_base}, "
              f"total windows={len(epochs)}")
        return file_paths, epochs


    def get_subject(self, file_idx):
        """All (sub-)epochs of one recording, same layout as __getitem__:
        (n_epochs, samples_per_sub), plus labels. Loaded fully into RAM so that
        disk reads never fall inside the timed region."""
        x = np.load(self._npy(self.file_paths[file_idx], 'x'))
        return x.reshape(-1, self.samples_per_sub), self.labels[file_idx]