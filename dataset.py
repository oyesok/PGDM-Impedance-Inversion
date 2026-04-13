import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class WaveImpedanceDataset(Dataset):
    def __init__(self, seismic_data_dir, impedance_data_dir, initial_model_dir, root_dir, mode='train', split_ratio=0.8, seed=42, transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.seismic_data_dir = seismic_data_dir
        self.impedance_data_dir = impedance_data_dir
        self.initial_model_dir = initial_model_dir
        self.split_ratio = split_ratio
        self.transform = transform

        if self.mode == 'test':
            self.seismic_files = ['marmousi_seismic_profile.mat']
            self.impedance_files = ['marmousi_impedance.mat']
            self.initial_model_files = ['initial_marmousi_impedance.mat']
        else:
            self.seismic_files = sorted([f for f in os.listdir(seismic_data_dir) if f.endswith('.mat')])
            self.impedance_files = sorted([f for f in os.listdir(impedance_data_dir) if f.endswith('.mat')])
            self.initial_model_files = sorted([f for f in os.listdir(initial_model_dir) if f.endswith('.mat')])

            assert len(self.seismic_files) == len(self.impedance_files) == len(self.initial_model_files), \
                "The number of seismic, impedance, and initial model files do not match."

            random.seed(seed)
            data_indices = list(range(len(self.seismic_files)))
            random.shuffle(data_indices)

            split_point = int(len(data_indices) * self.split_ratio)
            train_indices = data_indices[:split_point]
            val_indices = data_indices[split_point:]

            if self.mode == 'train':
                self.selected_indices = train_indices
            elif self.mode == 'val':
                self.selected_indices = val_indices
            else:
                raise ValueError("Mode must be 'train', 'val', or 'test'.")

    def __len__(self):
        if self.mode == 'test':
            return 1
        return len(self.selected_indices)

    def __getitem__(self, idx):
        if self.mode == 'test':
            seismic_file = os.path.join(self.seismic_data_dir, self.seismic_files[0])
            impedance_file = os.path.join(self.impedance_data_dir, self.impedance_files[0])
            initial_model_file = os.path.join(self.initial_model_dir, self.initial_model_files[0])

            seismic_data = loadmat(seismic_file)['marmousi_seismic_profile']
            impedance_data = loadmat(impedance_file)['marmousi_impedance_downsampled']
            initial_model_data = loadmat(initial_model_file)['initial_marmousi_impedance']
        else:
            actual_idx = self.selected_indices[idx]

            seismic_file = os.path.join(self.seismic_data_dir, self.seismic_files[actual_idx])
            impedance_file = os.path.join(self.impedance_data_dir, self.impedance_files[actual_idx])
            initial_model_file = os.path.join(self.initial_model_dir, self.initial_model_files[actual_idx])

            seismic_data = loadmat(seismic_file)['seismic_block']
            impedance_data = loadmat(impedance_file)['impedance_block']
            initial_model_data = loadmat(initial_model_file)['initial_model_block']

        seismic_data = np.array(seismic_data, dtype=np.float32)
        impedance_data = np.array(impedance_data, dtype=np.float32)
        initial_model_data = np.array(initial_model_data, dtype=np.float32)

        if np.isnan(seismic_data).any() or np.isinf(seismic_data).any():
            raise ValueError(f"NaN or Inf found in seismic data from file: {seismic_file}")
        if np.isnan(impedance_data).any() or np.isinf(impedance_data).any():
            raise ValueError(f"NaN or Inf found in impedance data from file: {impedance_file}")
        if np.isnan(initial_model_data).any() or np.isinf(initial_model_data).any():
            raise ValueError(f"NaN or Inf found in initial model data from file: {initial_model_file}")

        if seismic_data.ndim == 2:
            seismic_data = seismic_data[np.newaxis, ...]
        if impedance_data.ndim == 2:
            impedance_data = impedance_data[np.newaxis, ...]
        if initial_model_data.ndim == 2:
            initial_model_data = initial_model_data[np.newaxis, ...]

        if self.transform:
            seismic_data = self.transform(seismic_data)
            impedance_data = self.transform(impedance_data)
            initial_model_data = self.transform(initial_model_data)

        seismic_tensor = torch.tensor(seismic_data)
        impedance_tensor = torch.tensor(impedance_data)
        initial_model_tensor = torch.tensor(initial_model_data)

        sample = {
            'seismic': seismic_tensor,
            'initial_model': initial_model_tensor,
            'label': impedance_tensor
        }
        return sample
