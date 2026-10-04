import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple
import matplotlib

matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchmetrics.regression import (
    MeanAbsolutePercentageError,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
)
from tqdm import tqdm

from dataset_definition import UnifiedSeismicDataset
from model_definition import SeismicMagnitudePredictor

logger = logging.getLogger(name='Seismic-Magnitude-training')
implemented_scheduler_algs: List[str] = ['cos', 'exp']

def set_seed(seed: int = 67) -> None:

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Set seed for NumPy, PyTorch, etc.
set_seed()

def _plot_pca(
    model: SeismicMagnitudePredictor,
    test_loader: DataLoader,
    device: str,
    save_dir: str,
    filename: str = 'embedding_pca.jpg',
) -> None:
    """
    Extracts 64-dim embeddings from the test set, runs PCA → 2 components,
    and saves a scatter plot colored by magnitude bin (Low/Mid/High).
    """
    all_embeddings: List[np.ndarray] = []
    all_magnitudes: List[np.ndarray] = []

    print('Extracting embeddings for PCA...')
    with torch.inference_mode():
        for inputs, targets in tqdm(test_loader, desc='PCA embedding extraction'):
            inputs = inputs.to(device, non_blocking=True)
            emb = model.encode(inputs)
            all_embeddings.append(emb.cpu().numpy())
            all_magnitudes.append(targets.numpy())

    X = np.concatenate(all_embeddings, axis=0)  # (N, 64)
    y = np.concatenate(all_magnitudes, axis=0).squeeze()  # (N,)

    print(f'Embeddings shape: {X.shape}')

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    ev = pca.explained_variance_ratio_
    print(f'Explained variance — PC1: {ev[0]:.4f}, PC2: {ev[1]:.4f}')

    # Bin magnitudes: 0 = Low [<3), 1 = Mid [3–5), 2 = High [5+]
    bins = np.where(y < 3.0, 0, np.where(y < 5.0, 1, 2))

    bin_colors = ['blue', 'orange', 'red']
    bin_labels = ['Low [1-3)', 'Mid [3–5)', 'High [5+]']

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    for i in range(3):
        idx = bins == i
        ax.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            s=10,
            alpha=0.5,
            color=bin_colors[i],
            label=bin_labels[i],
        )

    ax.set_title(
        f'PCA of learned embeddings (CNN + LSTM)\n'
        f'PC1: {ev[0]*100:.1f}% var | PC2: {ev[1]*100:.1f}% var',
        fontsize=20,
        fontweight='bold',
    )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)
    print(f'PCA plot saved to `{save_path}`')


def run_experiment(
    memmap_path: str,
    csv_path: str,
    n_samples: int,
    experiment_name: str = 'magnitude_prediction',
    target_length: int = 1000,
    pad_length: int = 0,
    batch_size: int = 64,
    loss_function: Literal['MSE', 'HuberLoss'] = 'HuberLoss',
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    train_split: float = 0.7,
    val_split: float = 0.2,
    # test_split = 1 - train_split - val_split = 0.1
    scheduler_alg: Literal['exp', 'cos'] | None = 'cos',
    cos_eta_min: float | None = 1e-7,
    exp_lr_scheduler: float | None = 0.995,
    epochs: int = 300,
    no_progress_crash_out: int = 100,
    delta: float = 1.0,            
) -> Dict[str, Any]:

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
    residuals_file_name: str = 'residuals_test.jpg'
    pca_file_name: str = 'embedding_pca.jpg'
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

        shared_dataset_kwargs = dict(
            memmap_path=memmap_path,
            csv_path=csv_path,
            n_samples=n_samples,
            target_length=target_length,
            train_split=train_split,
            val_split=val_split,
        )

        train_dataset = UnifiedSeismicDataset(**shared_dataset_kwargs, phase='train')
        val_dataset   = UnifiedSeismicDataset(**shared_dataset_kwargs, phase='val')

        stft_params: Dict[str, Any] = train_dataset.get_stft_params()

        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4, 
            pin_memory=True, 
            drop_last=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True
        )

        if loss_function == 'MSE':
            criterion = nn.MSELoss()
        elif loss_function == 'HuberLoss':  
            criterion = nn.HuberLoss(delta=delta)
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
            'train_split': train_split,
            'val_split': val_split,
            'test_split': round(1.0 - train_split - val_split, 10),
            'target_length': target_length,
            'pad_length': pad_length,
            'loss_fn': loss_function,
            'delta': delta,
            'lr_scheduler': lr_scheduler_name,
            'lr_scheduler_param': lr_scheduler_param,
            **stft_params,  # STFT params logged here
        }

        train_losses: List[float] = []
        val_losses: List[float] = []
        train_MAEs: List[float] = []
        val_MAEs: List[float] = []

        val_MAPEs:    List[float] = []
        val_R2s:      List[float] = []
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

    # Test (Needed to ultimately test the model performance and summarize via scatter plot & resiudals protting)
    print('\nLoading best model for evaluation on test set...')
    best_model = SeismicMagnitudePredictor().to(device)
    best_model.load_state_dict(
        torch.load(os.path.join(models_dir, 'best_model.pth'), map_location=device)
    )
    best_model.eval()

    test_dataset = UnifiedSeismicDataset(
        memmap_path=memmap_path,
        csv_path=csv_path,
        n_samples=n_samples,
        target_length=target_length,
        phase='test',
        train_split=train_split,
        val_split=val_split,
    )

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )

    test_preds: List[torch.Tensor] = []
    test_targets: List[torch.Tensor] = []

    test_progression_bar = tqdm(test_loader, desc='Test inference')

    with torch.inference_mode():
        for inputs, targets in test_progression_bar:

            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = best_model(inputs)

            test_preds.append(outputs.squeeze(1).cpu())
            test_targets.append(targets.squeeze(1).cpu())

    test_preds_np = torch.cat(test_preds).numpy()
    test_targets_np = torch.cat(test_targets).numpy()
    residuals = test_preds_np - test_targets_np  # positive = over-prediction

    # Log the summary
    print(
        f'\nTest Set Results — '
        f'MAE: {np.mean(np.abs(residuals)):.4f} | '
        f'RMSE: {np.sqrt(np.mean(residuals**2)):.4f} | '
        f'Bias (mean residual): {np.mean(residuals):.4f}'
    )

    # Plot resiudals
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    # 1. scatter plot of predicted magnitudes vs ground truth
    mag_min = min(test_targets_np.min(), test_preds_np.min()) - 0.2
    mag_max = max(test_targets_np.max(), test_preds_np.max()) + 0.2
    axes3[0].scatter(test_targets_np, test_preds_np, alpha=0.35, s=12, color=(0, 0.35, 0.85))
    axes3[0].plot([mag_min, mag_max], [mag_min, mag_max], 'r--', linewidth=1.2, label='Perfect fit')
    axes3[0].set_xlim(mag_min, mag_max)
    axes3[0].set_ylim(mag_min, mag_max)
    axes3[0].set_xlabel('Ground Truth Magnitude')
    axes3[0].set_ylabel('Predicted Magnitude', rotation='horizontal', labelpad=60)
    # axes3[0].set_title('Predicted vs Ground Truth')
    axes3[0].spines['top'].set_visible(False)
    axes3[0].spines['right'].set_visible(False)
    axes3[0].legend(fontsize=9)

    # 2. Residuals histogram
    axes3[1].hist(residuals, bins=50, color=(0.4, 0.0, 0.7), edgecolor='white', linewidth=0.4)
    axes3[1].axvline(0, color='red', linestyle='--', linewidth=1.2)
    axes3[1].axvline(np.mean(residuals), color='orange', linestyle='-', linewidth=1.2,
                     label=f'Mean = {np.mean(residuals):.3f}')
    axes3[1].set_xlabel('Residual (Predicted − Ground Truth)')
    axes3[1].set_ylabel('Count', rotation='horizontal', labelpad=25)
    # axes3[1].set_title('Residuals Distribution')
    axes3[1].spines['top'].set_visible(False)
    axes3[1].spines['right'].set_visible(False)
    axes3[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(current_experiment_dir, residuals_file_name))
    plt.close(fig3)
    print(f'Residuals plot saved to `{os.path.join(current_experiment_dir, residuals_file_name)}`')

    test_preds_tensor = torch.from_numpy(test_preds_np)
    test_targets_tensor = torch.from_numpy(test_targets_np)

    test_mae  = float(np.mean(np.abs(residuals)))
    test_rmse = float(np.sqrt(np.mean(residuals**2)))
    test_bias = float(np.mean(residuals))
    test_mape = float(MeanAbsolutePercentageError()(test_preds_tensor, test_targets_tensor))
    test_r2   = float(R2Score()(test_preds_tensor, test_targets_tensor))

    abs_errors = np.abs(residuals)
    boot_means = np.array([
        np.mean(rng.choice(abs_errors, size=len(abs_errors), replace=True))
        for rng in [np.random.default_rng(i) for i in range(2000)]
    ])
    ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    test_metrics = {
        'test_MAE': test_mae,
        'test_RMSE': test_rmse,
        'test_bias': test_bias,
        'test_MAPE': test_mape,
        'test_R2': test_r2,
        'test_MAE_95CI': [ci_low, ci_high],
    }

    # Append test metrics to the existing hyperparams JSON
    hyperparam_path = os.path.join(current_experiment_dir, hyperparam_filename)
    with open(hyperparam_path, 'r') as f:
        saved_hyperparams = json.load(f)
    saved_hyperparams.update(test_metrics)
    with open(hyperparam_path, 'w') as f:
        json.dump(obj=saved_hyperparams, fp=f, indent=4)

    # PCA of test set
    _plot_pca(
        model=best_model,
        test_loader=test_loader,
        device=device,
        save_dir=current_experiment_dir,
        filename=pca_file_name,
    )

    del best_model
    torch.cuda.empty_cache()

    return test_metrics


def _sample_hparams(rng: random.Random) -> Dict[str, Any]:
    """
    Draws one random hyperparameter configuration.

    Ranges are chosen around the known-good baseline
    (lr=1e-3, wd=4e-3, batch=512, HuberLoss, exp/γ=0.995)
    while being wide enough to actually explore.

    Sampling strategy per param:
      lr           — log-uniform in [5e-5, 5e-3]:  small-to-large LR matters a lot
      weight_decay — log-uniform in [1e-4, 1e-2]:  regularisation strength
      batch_size   — categorical {128, 256, 512}:   memory-safe options
      loss_fn      — categorical {MSE, HuberLoss}:  both are plausible for regression
      scheduler    — categorical {exp, cos}:
          exp  → gamma log-uniform in [0.985, 0.999]   (slow-to-fast decay)
          cos  → eta_min log-uniform in [1e-7, 1e-5]   (floor for cosine schedule)
      no_progress_crash_out — uniform int in [60, 150]: patience window
    """
    lr           = 10 ** rng.uniform(-3.3, -2.3)
    weight_decay = 10 ** rng.uniform(-3.0, -2.0)
    batch_size   = rng.choice([128, 256])
    delta        = rng.uniform(0.15, 0.6)
    loss_fn      = 'HuberLoss'  
    scheduler    = 'cos'
    no_progress  = 150

    cos_eta_min      = None
    exp_lr_scheduler = None

    if scheduler == 'exp':
        exp_lr_scheduler = 10 ** rng.uniform(-1.5 / 300, 0)  # γ in [0.985, 0.999]
        # More direct: uniform in the gamma range itself
        exp_lr_scheduler = rng.uniform(0.985, 0.999)
    else:
        cos_eta_min = 10 ** rng.uniform(-7, -5)

    return {
        'lr':                 lr,
        'weight_decay':       weight_decay,
        'batch_size':         batch_size,
        'loss_function':      loss_fn,
        'scheduler_alg':      scheduler,
        'cos_eta_min':        cos_eta_min,
        'exp_lr_scheduler':   exp_lr_scheduler,
        'no_progress_crash_out': no_progress,
        'delta':              delta,
    }


def run_hparam_search(
    memmap_path: str,
    csv_path: str,
    n_samples: int,
    base_experiment_name: str,
    target_length: int = 1000,
    pad_length: int = 0,
    train_split: float = 0.7,
    val_split: float = 0.2,
    epochs: int = 300,
    n_trials: int = 15,
    seed: int = 42,
) -> None:
    """
    Runs `n_trials` randomly sampled configurations by calling `run_experiment`.

    Each trial is saved under:
        training_results/<date>/<base_experiment_name>/hparam_search/trial_<N>/

    A summary JSON with every trial's sampled config is written to:
        training_results/<date>/<base_experiment_name>/hparam_search/search_summary.json

    Args:
        n_trials:   How many random configs to try. 15 is a reasonable budget
                    for this search space — enough coverage without going broke
                    on GPU time.
        seed:       Controls the RNG for reproducible trial sampling. Individual
                    model training is still governed by set_seed() at module level.
    """
    rng = random.Random(seed)

    yyyy = str(datetime.today().year)
    mm   = str(datetime.today().month)
    dd   = str(datetime.today().day)

    search_root = os.path.join(
        './training_results', f'{yyyy}_{mm}_{dd}',
        base_experiment_name, 'hparam_search'
    )
    os.makedirs(search_root, exist_ok=True)

    summary: List[Dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"Starting randomized hyperparameter search — {n_trials} trials")
    print(f"Results root: {search_root}")
    print(f"{'='*60}\n")

    for trial_idx in range(1, n_trials + 1):
        sampled = _sample_hparams(rng)

        print(f"\n{'─'*60}")
        print(f"Trial {trial_idx}/{n_trials}")
        print(json.dumps(sampled, indent=2))
        print(f"{'─'*60}\n")

        trial_experiment_name = os.path.join(
            base_experiment_name, 'hparam_search', f'trial_{trial_idx}'
        )

        try:
            experiment_test_metrics = run_experiment(
                memmap_path=memmap_path,
                csv_path=csv_path,
                n_samples=n_samples,
                experiment_name=trial_experiment_name,
                target_length=target_length,
                pad_length=pad_length,
                train_split=train_split,
                val_split=val_split,
                epochs=epochs,
                **sampled,
            )
            status = 'completed'
        except Exception as e:
            print(f'Trial {trial_idx} failed with: {e}')
            status = f'failed: {e}'

        summary.append({
            'trial': trial_idx,
            'status': status,
            'config': sampled,
            'metrics': experiment_test_metrics if status == 'completed' else None,
        })

        # Flush summary after every trial so a crash mid-search doesn't lose results
        summary_path = os.path.join(search_root, 'search_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Hyperparameter search complete. Summary: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":

    DATA_DIR: str = 'data/INSTANCE/peak_normalized'  # relative to repo root; run from there
    DATASET_NAME: str = os.path.join(DATA_DIR.split('/')[-2], DATA_DIR.split('/')[-1])
    info_file: str = 'info.json' 
    waveforms_events_file: str = 'waveforms_events.bin'
    metadata_events_file: str = 'metadata_events.csv'

    with open(os.path.join(DATA_DIR, info_file)) as f:
        info = json.load(f)

    n_samples = info['events']['n_written']

    # We test the 3-second, 5-second, and 8-second extraction here
    target_lengths: List[int] = [
        300, 
        500, 
        800,
        1000
    ] 
    
    # best_trial_config: Dict[str, Any] = {
    #     'lr': 0.0013678960863632696,
    #     'weight_decay': 0.005067843068294094,
    #     'batch_size': 128,
    #     'loss_function': 'HuberLoss',
    #     'scheduler_alg': 'cos',
    #     'cos_eta_min': 8.6523999494305e-06
    # }

    # for length in target_lengths:
    #     print(f"\n{'='*50}\nRunning experiment for {length/100:.1f}s P-wave interval\n{'='*50}")
        
    #     # Calculate padding needed to always hit 10 seconds (1000 samples)
    #     pad_needed = 1000 - length
        
    #     run_experiment(
    #         memmap_path=os.path.join(DATA_DIR, waveforms_events_file),
    #         csv_path=os.path.join(DATA_DIR, metadata_events_file),
    #         n_samples=n_samples,
    #         experiment_name=os.path.join(DATASET_NAME, f'magnitude_pred_{length}_samples'),
    #         target_length=length,
    #         pad_length=pad_needed,
    #         batch_size=best_trial_config['batch_size'],
    #         lr=best_trial_config['lr'],
    #         weight_decay=best_trial_config['weight_decay'],
    #         loss_function=best_trial_config['loss_function'],
    #         scheduler_alg=best_trial_config['scheduler_alg'],
    #         cos_eta_min=best_trial_config['cos_eta_min'],
    #         train_split=0.7,
    #         val_split=0.2,
    #         epochs=500,
    #         no_progress_crash_out=500,
    #     )

    # Randomized hyperparameter search (full 10s window only)
    run_hparam_search(
        memmap_path=os.path.join(DATA_DIR, waveforms_events_file),
        csv_path=os.path.join(DATA_DIR, metadata_events_file),
        n_samples=n_samples,
        base_experiment_name=os.path.join(DATASET_NAME, 'magnitude_pred_1000_samples'),
        target_length=1000,
        pad_length=0,
        train_split=0.7,
        val_split=0.2,
        epochs=500,        # shorter budget per trial vs the main experiments
        n_trials=30,
        seed=67,
    )