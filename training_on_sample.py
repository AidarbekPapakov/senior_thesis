import json
import logging
import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from scipy.signal import ShortTimeFFT
# from scipy.signal.windows import hann
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

# Set seed for NumPy, PyTorch, etc.
set_seed()

class INSTANCESeismicDataset(Dataset):
    def __init__(
        self, 
        csv_path: str, 
        hdf5_path: str, 
        target_length: int = 300, # In samples
        pad_length: int = 700, # In samples as well
        phase: Literal['train', 'val'] = 'train',
        train_size: float = 0.75
    ) -> None:
        super().__init__()
        self.hdf5_path = hdf5_path
        self.target_length = target_length
        self.pad_length = pad_length
        self.total_samples = target_length + pad_length 
        
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['source_magnitude'])
        df = df[df['trace_eval_P'] == 'manual']
        df = df[df['station_channels'].isin(['HN', 'HH'])]
        df = df[(df['trace_npts'] - df['trace_P_arrival_sample']) >= self.total_samples]
        
        # Train/Val split
        split_idx: int = int(len(df) * train_size)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        if phase == 'train':
            self.metadata = df.iloc[:split_idx].reset_index(drop=True)
        else:
            self.metadata = df.iloc[split_idx:].reset_index(drop=True)
            
        logger.info(f"Initialized {phase} dataset with {len(self.metadata)} valid traces.")

        # STFT Parameters
        self.fs = 100
        self.N = 128
        self.overlap = 0.70
        self.hop_size = int(self.N * (1 - self.overlap)) 
        self.win = hann(self.N, sym=True)
        self.SFT = ShortTimeFFT(self.win, hop=self.hop_size, fs=self.fs, scale_to='magnitude')

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
        row = self.metadata.iloc[idx]
        trace_name = row['trace_name']
        p_idx = int(row['trace_P_arrival_sample'])
        
        # Target variable
        magnitude = torch.tensor([row['source_magnitude']], dtype=torch.float32)

        self.h5_file = None

        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        waveform = self.h5_file['data'][trace_name][:]    

        p_wave_clip = waveform[:, p_idx : p_idx + self.target_length]
        padded_clip = np.pad(p_wave_clip, ((0, 0), (0, self.pad_length)), mode='constant')

        waveform_tensor = torch.tensor(padded_clip, dtype=torch.float32)
        return waveform_tensor, magnitude


class SeismicCNNBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 3), padding=(2, 1)),  # 3 channels in
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
        
        # We define the STFT operation here so it GPU is utilized unlike with SciPy implenentation
        self.spectrogram = T.Spectrogram(
            n_fft=128,
            win_length=128,
            hop_length=38,         # 128 * (1 - 0.70)
            window_fn=torch.hann_window,
            power=1.0,             # power=1.0 gives you the magnitude (matches scale_to='magnitude')
            center=False,          # padds the values at the end, not from both sides
            normalized=False       
        )

        self.cnn = SeismicCNNBackbone()

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input data comes in raw form: (batch_size, 3, 1000)
        
        x = self.spectrogram(x)
        x = torch.log1p(x)         

        feats = self.cnn(x) # (batch, 64, 16)
        lstm_input = feats.permute(0, 2, 1) # (batch, 16, 64)

        lstm_out, _ = self.lstm(lstm_input)
        last_time_step = lstm_out[:, -1, :] # (batch, 64)

        magnitude_pred = self.mlp(last_time_step)   # (batch, 1)
        return magnitude_pred


def run_experiment(
    csv_path: str,
    hdf5_path: str,
    experiment_name: str = 'magnitude_prediction',
    target_length: int = 300,
    pad_length: int = 700,
    batch_size: int = 64,
    loss_function: Literal['MSE', 'HuberLoss'] = 'MSE',
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    train_size: float = 0.75,
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

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dir2save: str = 'saved_models/'
    result_dir: str = f'./training_results/{yyyy}_{mm}_{dd}/{experiment_name}'
    viz_file_name: str = 'training_metrics.jpg'
    extra_metrics_file_name: str = 'regression_metrics.jpg'
    hyperparam_filename: str = 'hyperparams.json'

    os.makedirs(result_dir, exist_ok=True)

    todays_experiments: List[str] = sorted([f for f in os.listdir(result_dir) if 'run_' in f])
    current_experiment: str = (
        str(int(todays_experiments[-1].split('_')[-1]) + 1) if todays_experiments else '1'
    )
    current_experiment_dir: str = os.path.join(result_dir, f'run_{current_experiment}')

    os.makedirs(current_experiment_dir, exist_ok=False)
    models_dir: str = os.path.join(current_experiment_dir, dir2save)
    os.makedirs(models_dir)

    def train_model() -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        print(f'Initializing CRNN Model on {device}...')
        
        model = SeismicMagnitudePredictor().to(device)
        
        train_dataset = INSTANCESeismicDataset(
            csv_path, hdf5_path, target_length, pad_length, phase='train', train_size=train_size
        )
        val_dataset = INSTANCESeismicDataset(
            csv_path, hdf5_path, target_length, pad_length, phase='val', train_size=train_size
        )
        
        stft_params: Dict[str, Any] = train_dataset.get_stft_params()

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
        )

        if loss_function == 'MSE':
            criterion = nn.MSELoss()
        elif loss_function == 'HuberLoss':  
            criterion = nn.HuberLoss()
        else:
            raise NotImplementedError(
                'Loss function is expected to be one of the following:\n["MSE", "HuberLoss"]\n'
                f'but got "{loss_function}"'
            )

        mae_metric = nn.L1Loss() # Used to track absolute error (interpretability)

        # Extra metrics on val only to look at the results from different perspective of sort
        mape_metric = MeanAbsolutePercentageError().to(device)
        r2_metric = R2Score().to(device)
        pearson_metric = PearsonCorrCoef().to(device)
        spearman_metric = SpearmanCorrCoef().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        lr_scheduler_name: str | None = None
        lr_scheduler_param: float | Dict[str, Any] | None = None
        
        if scheduler_alg == 'exp':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_lr_scheduler)
            lr_scheduler_name = 'ExponentialLR'
            lr_scheduler_param = exp_lr_scheduler
        elif scheduler_alg == 'cos':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=cos_eta_min
            )
            lr_scheduler_name = 'CosineAnnealingLR'
            lr_scheduler_param = {'T_max': epochs, 'eta_min': cos_eta_min}

        hyperparams: Dict[str, Any] = {
            'experiment_name': experiment_name,
            'epochs': epochs,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'train_size': train_size,
            'target_length': target_length,
            'pad_length': pad_length,
            'loss_fn': loss_function,
            'lr_scheduler': lr_scheduler_name,
            'lr_scheduler_param': lr_scheduler_param,
            **stft_params,  # STFT params logged here
        }

        train_losses: List[float] = []
        val_losses: List[float] = []
        train_MAEs: List[float] = []
        val_MAEs: List[float] = []

        # Extra metric progressions (val only)
        val_MAPEs: List[float] = []
        val_R2s: List[float] = []
        val_Pearsons: List[float] = []
        val_Spearmans: List[float] = []

        best_val_mae: float = float('inf')
        no_progress_epochs: int = 0

        for epoch in range(epochs):
            model.train()
            train_loss: float = 0.0
            train_mae: float = 0.0

            train_progression_bar = tqdm(train_loader, desc=f'Epoch: {epoch+1}')

            for inputs, targets in train_progression_bar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate absolute error
                mae = mae_metric(outputs, targets)

                train_loss += loss.item()
                train_mae += mae.item()

            if scheduler_alg is not None:
                lr_scheduler.step()

            train_losses.append(train_loss / len(train_loader))
            train_MAEs.append(train_mae / len(train_loader))
            print(f'Train MSE: {train_losses[-1]:.4f}, Train MAE: {train_MAEs[-1]:.4f}')

            # Validation Step
            model.eval()
            val_loss: float = 0.0
            val_mae: float = 0.0

            # Collect all preds & targets for epoch metric computation
            all_preds: List[torch.Tensor] = []
            all_targets: List[torch.Tensor] = []

            val_progression_bar = tqdm(val_loader, desc='Validation')

            with torch.inference_mode():
                for inputs, targets in val_progression_bar:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    mae = mae_metric(outputs, targets)

                    val_loss += loss.item()
                    val_mae += mae.item()

                    # Flatten to 1D for torchmetrics
                    all_preds.append(outputs.squeeze(1))
                    all_targets.append(targets.squeeze(1))

            val_losses.append(val_loss / len(val_loader))
            val_MAEs.append(val_mae / len(val_loader))
            print(f'Val MSE: {val_losses[-1]:.4f}, Val MAE: {val_MAEs[-1]:.4f}')

            # Compute extra metrics over the full val set
            epoch_preds = torch.cat(all_preds)    # (N,)
            epoch_targets = torch.cat(all_targets) # (N,)

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
                    f'Found new best model @{epoch + 1} epoch with Val MAE: {best_val_mae:.3f}.\n'
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

        training_metrics: Dict[str, List[float]] = {
            'train_losses': train_losses,
            'train_MAEs': train_MAEs,
            'val_losses': val_losses,
            'val_MAEs': val_MAEs,
            'val_MAPEs': val_MAPEs,
            'val_R2s': val_R2s,
            'val_Pearsons': val_Pearsons,
            'val_Spearmans': val_Spearmans,
        }

        return training_metrics, hyperparams, stft_params

    # Trigger training
    training_results, hyperparams, stft_params = train_model()

    hyperparams.update(training_results)

    # Save config & metrics
    with open(os.path.join(current_experiment_dir, hyperparam_filename), 'w') as f:
        json.dump(obj=hyperparams, fp=f, indent=4)

    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)

    axes[0][0].plot(training_results['train_losses'], color=(0, 0.2, 1))
    axes[1][0].plot(training_results['train_MAEs'], color=(0, 0.2, 1))
    axes[0][0].set_title('Train MSE Loss progression')
    axes[1][0].set_title('Train MAE progression')
    axes[0][0].set_ylim(bottom=0.0)
    axes[1][0].set_ylim(bottom=0.0)
    axes[0][0].set_xlabel('Epochs')
    axes[1][0].set_xlabel('Epochs')
    axes[0][0].set_ylabel('MSE Loss')
    axes[1][0].set_ylabel('MAE')

    axes[0][1].plot(training_results['val_losses'], color=(1, 0.5, 0))
    axes[1][1].plot(training_results['val_MAEs'], color=(1, 0.5, 0))
    axes[0][1].set_title('Val MSE Loss progression')
    axes[1][1].set_title('Val MAE progression')
    axes[0][1].set_ylim(bottom=0.0)
    axes[1][1].set_ylim(bottom=0.0)
    axes[0][1].set_xlabel('Epochs')
    axes[1][1].set_xlabel('Epochs')
    axes[0][1].set_ylabel('MSE Loss')
    axes[1][1].set_ylabel('MAE')

    plt.tight_layout()
    plt.savefig(os.path.join(current_experiment_dir, viz_file_name))
    plt.close(fig)

    epochs_range = np.arange(len(training_results['val_MAPEs']))

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
    purple = (0.5, 0.0, 0.8)
    green  = (0.0, 0.6, 0.3)

    axes2[0][0].plot(epochs_range, training_results['val_MAPEs'], color=purple)
    axes2[0][0].set_title('Val MAPE progression')
    axes2[0][0].set_xlabel('Epochs')
    axes2[0][0].set_ylabel('MAPE')
    axes2[0][0].set_ylim(bottom=0.0)

    axes2[0][1].plot(epochs_range, training_results['val_R2s'], color=green)
    axes2[0][1].set_title('Val R² progression')
    axes2[0][1].set_xlabel('Epochs')
    axes2[0][1].set_ylabel('R²')

    axes2[1][0].plot(epochs_range, training_results['val_Pearsons'], color=purple)
    axes2[1][0].set_title('Val Pearson Correlation progression')
    axes2[1][0].set_xlabel('Epochs')
    axes2[1][0].set_ylabel('Pearson r')

    axes2[1][1].plot(epochs_range, training_results['val_Spearmans'], color=green)
    axes2[1][1].set_title('Val Spearman Correlation progression')
    axes2[1][1].set_xlabel('Epochs')
    axes2[1][1].set_ylabel('Spearman ρ')

    plt.tight_layout()
    plt.savefig(os.path.join(current_experiment_dir, extra_metrics_file_name))
    plt.close(fig2)


if __name__ == "__main__":
    
    csv_file: str = "/mnt/d/Downloads/Senior_Thesis/INSTANCE/Instance_sample_dataset_v3/metadata/metadata_Instance_events_10k.csv"
    hdf5_file: str = "/mnt/d/Downloads/Senior_Thesis/INSTANCE/Instance_sample_dataset_v3/data/Instance_events_counts_10k.hdf5"
    
    # We test the 3-second, 5-second, and 8-second extraction here
    target_lengths: List[int] = [300, 500, 800] 
    
    for length in target_lengths:
        print(f"\n{'='*50}\nRunning experiment for {length/100:.1f}s P-wave interval\n{'='*50}")
        
        # Calculate padding needed to always hit 10 seconds (1000 samples)
        pad_needed = 1000 - length
        
        run_experiment(
            csv_path=csv_file,
            hdf5_path=hdf5_file,
            experiment_name=f'magnitude_pred_{length}_samples',
            target_length=length,
            pad_length=pad_needed,
            batch_size=128,
            lr=1e-3,
            train_size=0.8,
            scheduler_alg='cos',
            epochs=300,
            no_progress_crash_out=100
        )