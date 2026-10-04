"""Experiment variant: mixes STEAD/INSTANCE noise traces (magnitude 0) into the stratified events.
Kept for history; it uses its own noise-aware dataset and the older last-step-LSTM model."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T
from torchmetrics.regression import (
    MeanAbsolutePercentageError,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
)
from tqdm import tqdm

logger = logging.getLogger(name='Seismic-Magnitude-training')
implemented_scheduler_algs: List[str] = ['cos', 'exp']

def set_seed(seed: int = 67) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed()


class UnifiedSeismicDataset(Dataset):
    def __init__(
        self,
        events_memmap_path: str,
        events_csv_path: str,
        n_events: int,
        noise_memmap_path: str,
        noise_csv_path: str,
        n_noise: int,
        target_length: int = 1000,
        phase: Literal['train', 'val', 'test'] = 'train',
        train_split: float = 0.7,
        val_split: float = 0.2,
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

        # ── Load both memmaps ──────────────────────────────────────────────
        self.events_waveforms = np.memmap(
            events_memmap_path, dtype='float32', mode='r', shape=(n_events, 3, 1000)
        )
        self.noise_waveforms = np.memmap(
            noise_memmap_path, dtype='float32', mode='r', shape=(n_noise, 3, 1000)
        )

        # ── Build unified metadata with source tag & magnitude ─────────────
        events_df = pd.read_csv(events_csv_path)
        events_df['is_noise'] = False

        noise_df = pd.read_csv(noise_csv_path)
        noise_df['source_magnitude'] = 0.0
        noise_df['is_noise'] = True

        # Tag each row with its position in its own memmap
        events_df['memmap_idx'] = np.arange(len(events_df))
        noise_df['memmap_idx']  = np.arange(len(noise_df))

        combined = pd.concat(
            [events_df[['source_magnitude', 'is_noise', 'memmap_idx']],
             noise_df[['source_magnitude',  'is_noise', 'memmap_idx']]],
            ignore_index=True
        )

        # Shuffle once with fixed seed so train/val/test splits are reproducible
        combined = combined.sample(frac=1, random_state=67).reset_index(drop=True)

        n = len(combined)
        train_end = int(n * train_split)
        val_end   = int(n * (train_split + val_split))

        if phase == 'train':
            self.metadata = combined.iloc[:train_end].reset_index(drop=True)
        elif phase == 'val':
            self.metadata = combined.iloc[train_end:val_end].reset_index(drop=True)
        else:
            self.metadata = combined.iloc[val_end:].reset_index(drop=True)

        n_noise_in_split  = self.metadata['is_noise'].sum()
        n_events_in_split = (~self.metadata['is_noise']).sum()
        logger.info(
            f"Initialized {phase} dataset — "
            f"{n_events_in_split} events + {n_noise_in_split} noise = {len(self.metadata)} total"
        )
        print(
            f"  [{phase}] {n_events_in_split} events + {n_noise_in_split} noise "
            f"= {len(self.metadata)} total"
        )

        # STFT Parameters
        self.fs       = 100
        self.N        = 128
        self.overlap  = 0.70
        self.hop_size = int(self.N * (1 - self.overlap))

    def get_stft_params(self) -> Dict[str, Any]:
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
        row      = self.metadata.iloc[idx]
        mmap_idx = int(row['memmap_idx'])

        if row['is_noise']:
            wave = self.noise_waveforms[mmap_idx].copy()
        else:
            wave = self.events_waveforms[mmap_idx].copy()

        if self.target_length < 1000:
            wave[:, self.target_length:] = 0.0

        magnitude = torch.tensor([row['source_magnitude']], dtype=torch.float32)
        return torch.from_numpy(wave), magnitude


class SeismicCNNBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 16))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = x.squeeze(2)
        return x


class SeismicMagnitudePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spectrogram = T.Spectrogram(
            n_fft=128,
            win_length=128,
            hop_length=38,
            window_fn=torch.hann_window,
            power=1.0,
            center=False,
            normalized=False
        )
        self.cnn  = SeismicCNNBackbone()
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectrogram(x)
        x = torch.log1p(x)
        feats          = self.cnn(x)
        lstm_input     = feats.permute(0, 2, 1)
        lstm_out, _    = self.lstm(lstm_input)
        last_time_step = lstm_out[:, -1, :]
        return self.mlp(last_time_step)


def run_experiment(
    events_memmap_path: str,
    events_csv_path: str,
    n_events: int,
    noise_memmap_path: str,
    noise_csv_path: str,
    n_noise: int,
    experiment_name: str = 'magnitude_prediction',
    target_length: int = 1000,
    pad_length: int = 0,
    batch_size: int = 64,
    loss_function: Literal['MSE', 'HuberLoss'] = 'MSE',
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    train_split: float = 0.7,
    val_split: float = 0.2,
    scheduler_alg: Literal['exp', 'cos'] | None = 'cos',
    cos_eta_min: float | None = 1e-7,
    exp_lr_scheduler: float | None = 0.995,
    epochs: int = 300,
    no_progress_crash_out: int = 100,
) -> None:

    if scheduler_alg == 'cos' and cos_eta_min is None:
        raise ValueError('`scheduler_alg` is cos, but `cos_eta_min` left unspecified.')
    elif scheduler_alg == 'exp' and exp_lr_scheduler is None:
        raise ValueError('`scheduler_alg` is exp, but `exp_lr_scheduler` left unspecified.')
    elif scheduler_alg not in implemented_scheduler_algs and scheduler_alg is not None:
        raise NotImplementedError(
            f'`scheduler_alg` expected {implemented_scheduler_algs}, got `{scheduler_alg}`'
        )

    yyyy, mm, dd = (
        str(datetime.today().year),
        str(datetime.today().month),
        str(datetime.today().day),
    )

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    dir2save  = 'saved_models/'
    result_dir = f'./training_results/{yyyy}_{mm}_{dd}/{experiment_name}'

    viz_file_name          = 'training_metrics.jpg'
    extra_metrics_file_name = 'regression_metrics.jpg'
    residuals_file_name    = 'residuals_test.jpg'
    hyperparam_filename    = 'hyperparams.json'

    os.makedirs(result_dir, exist_ok=True)

    todays_experiments = sorted([f for f in os.listdir(result_dir) if 'run_' in f])
    current_experiment = (
        str(int(todays_experiments[-1].split('_')[-1]) + 1) if todays_experiments else '1'
    )
    current_experiment_dir = os.path.join(result_dir, f'run_{current_experiment}')
    os.makedirs(current_experiment_dir, exist_ok=False)
    models_dir = os.path.join(current_experiment_dir, dir2save)
    os.makedirs(models_dir)

    def train_model() -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        print(f'Initializing CRNN Model on {device}...')

        model = SeismicMagnitudePredictor().to(device)

        shared_dataset_kwargs = dict(
            events_memmap_path=events_memmap_path,
            events_csv_path=events_csv_path,
            n_events=n_events,
            noise_memmap_path=noise_memmap_path,
            noise_csv_path=noise_csv_path,
            n_noise=n_noise,
            target_length=target_length,
            train_split=train_split,
            val_split=val_split,
        )

        train_dataset = UnifiedSeismicDataset(**shared_dataset_kwargs, phase='train')
        val_dataset   = UnifiedSeismicDataset(**shared_dataset_kwargs, phase='val')

        stft_params = train_dataset.get_stft_params()

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )

        criterion  = nn.MSELoss() if loss_function == 'MSE' else (
            nn.HuberLoss() if loss_function == 'HuberLoss' else (_ for _ in ()).throw(
                NotImplementedError(f'Unknown loss: {loss_function}')
            )
        )
        mae_metric     = nn.L1Loss()
        mape_metric    = MeanAbsolutePercentageError().to(device)
        r2_metric      = R2Score().to(device)
        pearson_metric = PearsonCorrCoef().to(device)
        spearman_metric = SpearmanCorrCoef().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        lr_scheduler_name:  str | None = None
        lr_scheduler_param: Any        = None

        if scheduler_alg == 'exp':
            lr_scheduler       = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_lr_scheduler)
            lr_scheduler_name  = 'ExponentialLR'
            lr_scheduler_param = exp_lr_scheduler
        elif scheduler_alg == 'cos':
            lr_scheduler       = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=cos_eta_min
            )
            lr_scheduler_name  = 'CosineAnnealingLR'
            lr_scheduler_param = {'T_max': epochs, 'eta_min': cos_eta_min}

        hyperparams: Dict[str, Any] = {
            'experiment_name':    experiment_name,
            'epochs':             epochs,
            'learning_rate':      lr,
            'weight_decay':       weight_decay,
            'batch_size':         batch_size,
            'train_split':        train_split,
            'val_split':          val_split,
            'test_split':         round(1.0 - train_split - val_split, 10),
            'target_length':      target_length,
            'pad_length':         pad_length,
            'loss_fn':            loss_function,
            'lr_scheduler':       lr_scheduler_name,
            'lr_scheduler_param': lr_scheduler_param,
            'n_events':           n_events,
            'n_noise':            n_noise,
            **stft_params,
        }

        train_losses:   List[float] = []
        val_losses:     List[float] = []
        train_MAEs:     List[float] = []
        val_MAEs:       List[float] = []
        val_MAPEs:      List[float] = []
        val_R2s:        List[float] = []
        val_Pearsons:   List[float] = []
        val_Spearmans:  List[float] = []

        best_val_mae      = float('inf')
        no_progress_epochs = 0

        for epoch in range(epochs):
            model.train()
            train_loss = train_mae = 0.0

            for inputs, targets in tqdm(train_loader, desc=f'Epoch: {epoch+1}'):
                inputs  = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_mae  += mae_metric(outputs, targets).item()

            if scheduler_alg is not None:
                lr_scheduler.step()

            train_losses.append(train_loss / len(train_loader))
            train_MAEs.append(train_mae / len(train_loader))
            print(f'Train MSE: {train_losses[-1]:.4f}, Train MAE: {train_MAEs[-1]:.4f}')

            model.eval()
            val_loss = val_mae = 0.0
            all_preds:   List[torch.Tensor] = []
            all_targets: List[torch.Tensor] = []

            with torch.inference_mode():
                for inputs, targets in tqdm(val_loader, desc='Validation'):
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    val_mae  += mae_metric(outputs, targets).item()
                    all_preds.append(outputs.squeeze(1))
                    all_targets.append(targets.squeeze(1))

            val_losses.append(val_loss / len(val_loader))
            val_MAEs.append(val_mae / len(val_loader))
            print(f'Val MSE: {val_losses[-1]:.4f}, Val MAE: {val_MAEs[-1]:.4f}')

            epoch_preds   = torch.cat(all_preds)
            epoch_targets = torch.cat(all_targets)

            val_MAPEs.append(mape_metric(epoch_preds, epoch_targets).item())
            val_R2s.append(r2_metric(epoch_preds, epoch_targets).item())
            val_Pearsons.append(pearson_metric(epoch_preds, epoch_targets).item())
            val_Spearmans.append(spearman_metric(epoch_preds, epoch_targets).item())

            print(
                f'Val MAPE: {val_MAPEs[-1]:.4f} | R²: {val_R2s[-1]:.4f} | '
                f'Pearson: {val_Pearsons[-1]:.4f} | Spearman: {val_Spearmans[-1]:.4f}'
            )

            if val_MAEs[-1] < best_val_mae:
                best_val_mae = val_MAEs[-1]
                print(
                    f'Found new best model @{epoch+1} epoch with Val MAE: {best_val_mae:.3f}.\n'
                    f'Saving to `{models_dir}`...'
                )
                torch.save(model.state_dict(), os.path.join(models_dir, 'best_model.pth'))
                no_progress_epochs = 0
            else:
                no_progress_epochs += 1
                if no_progress_epochs >= no_progress_crash_out:
                    print(
                        f'No progress in validation MAE for {no_progress_epochs} epochs. '
                        'Stopping training loop...'
                    )
                    break

        print('\nTraining complete.')
        del model
        torch.cuda.empty_cache()

        return {
            'train_losses': train_losses, 'train_MAEs': train_MAEs,
            'val_losses':   val_losses,   'val_MAEs':   val_MAEs,
            'val_MAPEs':    val_MAPEs,    'val_R2s':    val_R2s,
            'val_Pearsons': val_Pearsons, 'val_Spearmans': val_Spearmans,
        }, hyperparams, stft_params

    training_results, hyperparams, stft_params = train_model()
    hyperparams.update(training_results)

    with open(os.path.join(current_experiment_dir, hyperparam_filename), 'w') as f:
        json.dump(obj=hyperparams, fp=f, indent=4)

    # ── Training metric plots ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
    axes[0][0].plot(training_results['train_losses'], color=(0, 0.2, 1))
    axes[1][0].plot(training_results['train_MAEs'],   color=(0, 0.2, 1))
    axes[0][1].plot(training_results['val_losses'],   color=(1, 0.5, 0))
    axes[1][1].plot(training_results['val_MAEs'],     color=(1, 0.5, 0))
    for ax, title, ylabel in zip(
        axes.flat,
        ['Train MSE Loss', 'Val MSE Loss', 'Train MAE', 'Val MAE'],
        ['MSE Loss', 'MSE Loss', 'MAE', 'MAE']
    ):
        ax.set_title(f'{title} progression')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(os.path.join(current_experiment_dir, viz_file_name))
    plt.close(fig)

    epochs_range = np.arange(len(training_results['val_MAPEs']))
    fig2, axes2  = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
    purple = (0.5, 0.0, 0.8)
    green  = (0.0, 0.6, 0.3)
    axes2[0][0].plot(epochs_range, training_results['val_MAPEs'],    color=purple)
    axes2[0][1].plot(epochs_range, training_results['val_R2s'],      color=green)
    axes2[1][0].plot(epochs_range, training_results['val_Pearsons'], color=purple)
    axes2[1][1].plot(epochs_range, training_results['val_Spearmans'],color=green)
    for ax, title, ylabel in zip(
        axes2.flat,
        ['Val MAPE', 'Val R²', 'Val Pearson Correlation', 'Val Spearman Correlation'],
        ['MAPE', 'R²', 'Pearson r', 'Spearman ρ']
    ):
        ax.set_title(f'{title} progression')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel)
    axes2[0][0].set_ylim(bottom=0.0)
    plt.tight_layout()
    plt.savefig(os.path.join(current_experiment_dir, extra_metrics_file_name))
    plt.close(fig2)

    # ── Test phase ─────────────────────────────────────────────────────────
    print('\nLoading best model for evaluation on test set...')
    best_model = SeismicMagnitudePredictor().to(device)
    best_model.load_state_dict(
        torch.load(os.path.join(models_dir, 'best_model.pth'), map_location=device)
    )
    best_model.eval()

    test_dataset = UnifiedSeismicDataset(
        events_memmap_path=events_memmap_path,
        events_csv_path=events_csv_path,
        n_events=n_events,
        noise_memmap_path=noise_memmap_path,
        noise_csv_path=noise_csv_path,
        n_noise=n_noise,
        target_length=target_length,
        phase='test',
        train_split=train_split,
        val_split=val_split,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    test_preds:   List[torch.Tensor] = []
    test_targets: List[torch.Tensor] = []

    with torch.inference_mode():
        for inputs, targets in tqdm(test_loader, desc='Test inference'):
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = best_model(inputs)
            test_preds.append(outputs.squeeze(1).cpu())
            test_targets.append(targets.squeeze(1).cpu())

    test_preds_np   = torch.cat(test_preds).numpy()
    test_targets_np = torch.cat(test_targets).numpy()
    residuals       = test_preds_np - test_targets_np

    print(
        f'\nTest Set Results — '
        f'MAE: {np.mean(np.abs(residuals)):.4f} | '
        f'RMSE: {np.sqrt(np.mean(residuals**2)):.4f} | '
        f'Bias (mean residual): {np.mean(residuals):.4f}'
    )

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    mag_min = min(test_targets_np.min(), test_preds_np.min()) - 0.2
    mag_max = max(test_targets_np.max(), test_preds_np.max()) + 0.2
    axes3[0].scatter(test_targets_np, test_preds_np, alpha=0.35, s=12, color=(0, 0.35, 0.85))
    axes3[0].plot([mag_min, mag_max], [mag_min, mag_max], 'r--', linewidth=1.2, label='Perfect fit')
    axes3[0].set_xlim(mag_min, mag_max)
    axes3[0].set_ylim(mag_min, mag_max)
    axes3[0].set_xlabel('Ground Truth Magnitude')
    axes3[0].set_ylabel('Predicted Magnitude')
    axes3[0].set_title('Predicted vs Ground Truth')
    axes3[0].legend(fontsize=9)
    axes3[1].hist(residuals, bins=50, color=(0.4, 0.0, 0.7), edgecolor='white', linewidth=0.4)
    axes3[1].axvline(0, color='red', linestyle='--', linewidth=1.2)
    axes3[1].axvline(np.mean(residuals), color='orange', linestyle='-', linewidth=1.2,
                     label=f'Mean = {np.mean(residuals):.3f}')
    axes3[1].set_xlabel('Residual (Predicted − Ground Truth)')
    axes3[1].set_ylabel('Count')
    axes3[1].set_title('Residuals Distribution')
    axes3[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(current_experiment_dir, residuals_file_name))
    plt.close(fig3)
    print(f'Residuals plot saved to `{os.path.join(current_experiment_dir, residuals_file_name)}`')

    del best_model
    torch.cuda.empty_cache()


if __name__ == "__main__":

    DATA_DIR      = '/home/aidar/study/senior_thesis/data/INSTANCE/unprocessed_amplitude'
    DATASET_NAME  = os.path.join(DATA_DIR.split('/')[-2], DATA_DIR.split('/')[-1])

    with open(os.path.join(DATA_DIR, 'info.json')) as f:
        info = json.load(f)

    n_events = info['events']['n_written']
    n_noise  = info['noise']['n_written']

    target_lengths: List[int] = [300, 500, 800, 1000]

    for length in target_lengths:
        print(f"\n{'='*50}\nRunning experiment for {length/100:.1f}s P-wave interval\n{'='*50}")

        run_experiment(
            events_memmap_path=os.path.join(DATA_DIR, 'waveforms_events.bin'),
            events_csv_path=os.path.join(DATA_DIR, 'metadata_events.csv'),
            n_events=n_events,
            noise_memmap_path=os.path.join(DATA_DIR, 'waveforms_noise.bin'),
            noise_csv_path=os.path.join(DATA_DIR, 'metadata_noise.csv'),
            n_noise=n_noise,
            experiment_name=os.path.join(DATASET_NAME, f'magnitude_pred_{length}_samples'),
            target_length=length,
            pad_length=1000 - length,
            batch_size=512,
            lr=5e-4,
            weight_decay=1e-3,
            train_split=0.7,
            val_split=0.2,
            scheduler_alg='exp',
            exp_lr_scheduler=0.998,
            epochs=500,
            no_progress_crash_out=500,
        )