# ==================== Earthquake Magnitude CNN ====================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# -- Hyperparameters (configurable) --
INPUT_CHANNELS  = 1         # e.g. single spectrogram channel
INPUT_HEIGHT    = 128       # number of frequency bins in spectrogram
INPUT_WIDTH     = 128       # number of time bins (depends on P-wave duration)
CONV_CHANNELS   = [16, 32, 64]  # output channels for each conv layer
KERNEL_SIZES    = [3, 3, 3]     # kernel sizes (3x3)
POOL_SIZE       = 2             # pooling factor (2x2)
DROPOUT_RATE    = 0.3           # dropout probability (30%)
FC_HIDDEN_SIZES = [64]          # fully connected hidden layer sizes
LEARNING_RATE   = 1e-3          # Adam learning rate
BATCH_SIZE      = 32            # batch size for training
EPOCHS          = 10            # number of training epochs
# ==================================================================

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the CNN model
class EarthquakeCNN(nn.Module):
    def __init__(self):
        super(EarthquakeCNN, self).__init__()
        # Build convolutional layers dynamically
        layers = []
        in_channels = INPUT_CHANNELS
        for out_channels, kernel in zip(CONV_CHANNELS, KERNEL_SIZES):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel//2),
                nn.ReLU(),
                nn.MaxPool2d(POOL_SIZE),
                nn.Dropout(DROPOUT_RATE)
            ]
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)  # Stack into Sequential

        # Determine flattened feature size after conv layers using a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
            conv_out = self.conv(dummy)
            conv_out_size = conv_out.numel()

        # Build fully connected layers (MLP block), if any
        fc_layers = []
        in_features = conv_out_size
        for hidden_units in FC_HIDDEN_SIZES:
            fc_layers += [
                nn.Linear(in_features, hidden_units),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE)
            ]
            in_features = hidden_units
        # Final output layer (1 output for regression)
        fc_layers.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x shape: [batch, channels, freq, time]
        x = self.conv(x)               # Convolutional feature extraction
        x = x.view(x.size(0), -1)      # Flatten
        x = self.fc(x)                 # Fully connected to output
        return x.squeeze(1)           # Return shape [batch] (regression output)

# Data loading and preprocessing (placeholder)
# Assume spectrograms X (numpy array shape [N, 1, freq, time]) and labels y (numpy array [N]) are available.
# For example:
# X = np.load('spectrograms.npy')  # shape (num_samples, 1, freq_bins, time_bins)
# y = np.load('magnitudes.npy')    # shape (num_samples,)

# Example dummy data (replace with actual data loading)
num_samples = 1000
X = np.random.randn(num_samples, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH).astype(np.float32)
y = np.random.uniform(2.0, 5.0, size=(num_samples,)).astype(np.float32)  # dummy magnitudes

# Normalize spectrograms (per-sample standardization)
X_norm = np.zeros_like(X)
for i in range(len(X)):
    spec = X[i]
    mean = spec.mean()
    std = spec.std() if spec.std() > 0 else 1.0
    X_norm[i] = (spec - mean) / std
X = X_norm

# Split into train/validation/test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test,  y_val,  y_test  = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors and create DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# Initialize model, loss, optimizer
model = EarthquakeCNN().to(device)
criterion = nn.MSELoss()                      # MSE loss for regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        preds = model(X_batch)                 # Forward pass
        loss = criterion(preds, y_batch)       # Compute MSE loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {epoch_loss:.4f}")

# Evaluation on test set
model.eval()
mse_total = 0.0
mae_total = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        preds = model(X_batch)
        mse = nn.functional.mse_loss(preds, y_batch, reduction='sum')
        mae = nn.functional.l1_loss(preds, y_batch, reduction='sum')
        mse_total += mse.item()
        mae_total += mae.item()

num_test = len(test_dataset)
rmse = np.sqrt(mse_total / num_test)
mae  = mae_total / num_test
print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")
