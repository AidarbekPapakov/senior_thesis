"""
CNN-LSTM magnitude regressor used for the thesis results (Chapter 5).

Pipeline:
    raw waveform (B, 3, 1000)
      -> STFT magnitude + log1p        (B, 3, 65, T)
      -> CNN backbone                  (B, 64, 16)
      -> unidirectional LSTM           (B, 16, 64)
      -> last time step                (B, 64)
      -> MLP regression head           (B, 1)

Attention pooling over the LSTM steps was tried and dropped: in the early ablation
runs (`training_results/2026_4_6/mag_pred_{300,500,800}_{attention,baseline}`, single
seed, MSE loss) the best validation MAE was within noise of the baseline at every
window length (e.g. 500 samples: 0.423 attention vs 0.424 baseline). Later versions of
the training script that used attention pooling and Dropout2d are in the git history.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T


class SeismicCNNBackbone(nn.Module):
    """Three Conv-BN-ReLU-MaxPool blocks that squeeze a spectrogram into (B, 64, 16)."""

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
            nn.AdaptiveAvgPool2d((1, 16)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = x.squeeze(2)
        return x


class SeismicMagnitudePredictor(nn.Module):
    """End-to-end model: raw 3-component waveform in, scalar magnitude out."""

    def __init__(self) -> None:
        super().__init__()

        # STFT lives inside the model so it runs on the GPU (unlike SciPy's implementation)
        self.spectrogram = T.Spectrogram(
            n_fft=128,
            win_length=128,
            hop_length=38,         # 128 * (1 - 0.70)
            window_fn=torch.hann_window,
            power=1.0,             # power=1.0 in torchaudio is the equivalent to scale_to='magnitude'
            center=False,          # no padding on either side
            normalized=False,
        )

        self.cnn = SeismicCNNBackbone()
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the 64-dim last-LSTM-step embedding that feeds the MLP head."""
        # x: raw waveform (batch_size, 3, 1000)
        x = self.spectrogram(x)
        x = torch.log1p(x)

        feats = self.cnn(x)                  # (batch, 64, 16)
        lstm_input = feats.permute(0, 2, 1)  # (batch, 16, 64)
        lstm_out, _ = self.lstm(lstm_input)
        return lstm_out[:, -1, :]            # (batch, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encode(x))
