from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class UnifiedSeismicDataset(Dataset):
    def __init__(
        self, 
        memmap_path: str,
        csv_path: str,
        n_samples: int,
        target_length: int = 1000,
        phase: Literal['train', 'val', 'test'] = 'train',
        train_split: float = 0.7,
        val_split: float = 0.2,
        # test_split is thus: 1 - (train_split + val_split) = 0.1
    ) -> None:
        super().__init__()

        if phase not in ['train', 'val', 'test']:
            raise ValueError(
                'phase is expected to be one of the:\n[train, val, test]\n\n'
                f'but got {phase} instead.'
                )

        if not (0 < train_split < 1 and 0 < val_split < 1 and train_split + val_split < 1):
            raise ValueError(
                f'Invalid splits: train={train_split}, val={val_split}. '
                'Must be positive and sum to less than 1.'
            )

        self.target_length = target_length
        self.waveforms = np.memmap(
            memmap_path, dtype='float32',
            mode='r', shape=(n_samples, 3, 1000)
        )

        df = pd.read_csv(csv_path)
        n = len(df)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))

        if phase == 'train':
            self.metadata = df.iloc[:train_end].reset_index(drop=True)
            self.indices = list(range(train_end))
        elif phase == 'val':
            self.metadata = df.iloc[train_end:val_end].reset_index(drop=True)
            self.indices = list(range(train_end, val_end))
        else:
            self.metadata = df.iloc[val_end:].reset_index(drop=True)
            self.indices = list(range(val_end, n))

        # STFT Parameters
        self.fs = 100
        self.N = 128
        self.overlap = 0.70
        self.hop_size = int(self.N * (1 - self.overlap)) 

    def get_stft_params(self) -> Dict[str, Any]:
        """Expose STFT config for external logging."""
        return {
            'stft_fs': self.fs,
            'stft_window_size_N': self.N,
            'stft_overlap': self.overlap,
            'stft_hop_size': self.hop_size,
            'stft_window_type': 'hann',
            'stft_scale_to': 'magnitude',
        }

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        wave = self.waveforms[self.indices[idx]].copy()  # (3, 1000)

        # Zero out everything past target_length — this is how you swap
        # between 3s/5s/8s experiments without re-running preprocessing
        if self.target_length < 1000:
            wave[:, self.target_length:] = 0.0

        magnitude = torch.tensor(
            [self.metadata.iloc[idx]['source_magnitude']], dtype=torch.float32
        )
        return torch.from_numpy(wave), magnitude