import torch
import torch.nn as nn
import torchaudio.transforms as T

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
            power=1.0,             # power=1.0 in torchaudio is the equivalent to scale_to='magnitude'
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