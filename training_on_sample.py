import os
import pandas as pd
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from tqdm import tqdm

# ==========================================
# 1. DATASET & PREPROCESSING
# ==========================================
class INSTANCESeismicDataset(Dataset):
    def __init__(self, csv_fullpath, hdf5_fullpath, target_length=300, pad_length=700):
        super().__init__()
        self.hdf5_fullpath = hdf5_fullpath
        self.target_length = target_length
        self.pad_length = pad_length
        
        # Total length after padding (300 + 700 = 1000 samples)
        self.total_samples = target_length + pad_length 
        
        # Load and filter metadata exactly as we discussed
        print("Filtering metadata...")
        df = pd.read_csv(csv_fullpath)
        df = df.dropna(subset=['source_magnitude'])
        df = df[df['trace_eval_P'] == 'manual']
        df = df[df['station_channels'].isin(['HN', 'HH'])]
        df = df[(df['trace_npts'] - df['trace_P_arrival_sample']) >= self.total_samples]
        self.metadata = df.reset_index(drop=True)
        print(f"Dataset initialized with {len(self.metadata)} valid traces.")

        # STFT Parameters
        self.fs = 100
        self.N = 128
        self.overlap = 0.90
        self.hop_size = int(self.N * (1 - self.overlap)) # 12 samples
        self.win = hann(self.N, sym=True)
        self.SFT = ShortTimeFFT(self.win, hop=self.hop_size, fs=self.fs, scale_to='magnitude')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        trace_name = row['trace_name']
        p_idx = int(row['trace_P_arrival_sample'])
        magnitude = torch.tensor([row['source_magnitude']], dtype=torch.float32)

        # Open HDF5 on the fly to avoid multiprocessing lock issues in PyTorch DataLoaders
        with h5py.File(self.hdf5_fullpath, 'r') as h5:
            # Shape is usually (3, 12000) for E, N, Z components
            waveform = h5['data'][trace_name][:] 

        # Slice 3 seconds (300 samples) starting exactly at the P-wave
        p_wave_clip = waveform[:, p_idx : p_idx + self.target_length]

        # Post-pad with zeros to reach 10 seconds (1000 samples)
        padded_clip = np.pad(p_wave_clip, ((0, 0), (0, self.pad_length)), mode='constant')

        # Convert the 3 components into 3 spectrograms
        # Output shape for each: ~ (65, 68)
        spectrograms = []
        for i in range(3):
            # .spectrogram() gives the squared magnitude (power)
            spec = self.SFT.spectrogram(padded_clip[i])
            # Log transform to compress the dynamic range of seismic data
            spec = np.log1p(spec) 
            spectrograms.append(spec)

        # Stack into a single tensor of shape (3, 65, 68)
        # Note: We will split these into 3 separate tensors in the model's forward pass
        stft_tensor = torch.tensor(np.stack(spectrograms), dtype=torch.float32)

        return stft_tensor, magnitude

# ==========================================
# 2. THE DEEP LEARNING ARCHITECTURE
# ==========================================
class SeismicCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Input shape: (Batch, 1, 65, 68) -> 1 channel (since we process E, N, Z separately)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Pool frequency, keep time
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # Pool both slightly
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Adaptive pool to crush the frequency axis to 1, but leave time axis at a fixed length (e.g., 16)
            nn.AdaptiveAvgPool2d((1, 16)) 
        )

    def forward(self, x):
        # x is (B, 1, 65, 68)
        x = self.conv_block(x) # Output: (B, 64, 1, 16)
        x = x.squeeze(2)       # Output: (B, 64, 16) - Features x Time
        return x

class SeismicMagnitudePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 3.1) Three independent CNN backbones for East, North, and Z components
        self.cnn_E = SeismicCNNBackbone()
        self.cnn_N = SeismicCNNBackbone()
        self.cnn_Z = SeismicCNNBackbone()

        # 3.3) LSTM Block
        # We concatenate the 64 features from each CNN -> 192 features per time step
        self.lstm = nn.LSTM(input_size=64 * 3, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)

        # 3.4) MLP Regression Head
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1) # Predicts single continuous value (Magnitude)
        )

    def forward(self, x):
        # x shape is (B, 3, 65, 68)
        # Split into the 3 components and add the channel dimension (B, 1, 65, 68)
        x_E = x[:, 0, :, :].unsqueeze(1)
        x_N = x[:, 1, :, :].unsqueeze(1)
        x_Z = x[:, 2, :, :].unsqueeze(1)

        # Pass through backbones
        feat_E = self.cnn_E(x_E) # (B, 64, 16)
        feat_N = self.cnn_N(x_N) # (B, 64, 16)
        feat_Z = self.cnn_Z(x_Z) # (B, 64, 16)

        # 3.2) Aggregate the 3 resulting feature matrices
        # Concatenate along the feature dimension (dim=1) -> (B, 192, 16)
        combined_feats = torch.cat((feat_E, feat_N, feat_Z), dim=1)

        # Permute for LSTM which expects (Batch, Sequence, Features) -> (B, 16, 192)
        lstm_input = combined_feats.permute(0, 2, 1)

        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        # Grab the output of the very last time step from the top LSTM layer
        last_time_step = lstm_out[:, -1, :] # (B, 128)

        # Pass through Regression Head
        magnitude_pred = self.mlp(last_time_step) # (B, 1)

        return magnitude_pred

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Initialize Dataset and DataLoader
    metadata_dir: str = '/mnt/d/Downloads/Senior_Thesis/INSTANCE/Instance_sample_dataset_v3/metadata'
    data_dir: str = '/mnt/d/Downloads/Senior_Thesis/INSTANCE/Instance_sample_dataset_v3/data'

    csv_name: str = 'metadata_Instance_events_10k.csv'
    hdf5_name: str = 'Instance_events_counts_10k.hdf5'

    csv_fullpath = os.path.join(metadata_dir, csv_name)
    hdf5_fullpath = os.path.join(data_dir, hdf5_name)
    
    dataset = INSTANCESeismicDataset(csv_fullpath, hdf5_fullpath)
    
    # Keeping num_workers=0 is safer since we are reading HDF5 files on the fly
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Initialize Model, Loss, Optimizer
    model = SeismicMagnitudePredictor().to(device)
    criterion = nn.MSELoss() # Mean Squared Error is standard for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    epochs = 20

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        train_progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (inputs, targets) in enumerate(train_progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(inputs)

            # Calculate Loss
            loss = criterion(predictions, targets)

            # Backward pass & Optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Train MSE Loss: {avg_loss:.4f}")

    print("Training Complete")

if __name__ == "__main__":
    # Uncomment to run
    train_model()