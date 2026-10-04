# Chapter 5 — Model Architecture

## 5.1 Overview of the Model Pipeline

The goal of the proposed method is to estimate earthquake magnitude using only the first few seconds of seismic waveforms right after the arrival of the P-wave. Rather than relying on CPU-bound preprocessing, the model ingests raw temporal waveforms and dynamically computes spectrograms on the GPU.

The overall architecture combines a CNN for spatial feature extraction, an LSTM block for modeling temporal changes, and an MLP regression head.

The processing pipeline can be depicted as follows:

```text
Raw input waveforms
      ↓
On-device STFT & Log1p scaling
      ↓
CNN backbone (spatial feature extraction)
      ↓
LSTM (temporal modeling, last time step)
      ↓
MLP (regression head)
      ↓
Scalar magnitude value
```

Formally, the model can be expressed as a function:

$$\hat{M} = f_\theta(X)$$

where
* $X$ is the raw three-component seismic waveform.
* $\theta$ is the set of learnable parameters.
* $\hat{M}$ is the predicted earthquake magnitude.

The architecture is built on two primary premises:
* **Spectral structure** — frequencies vary distinctively during an earthquake.
* **Temporal evolution** — signal characteristics evolve crucially during the first seconds after P-wave arrival.

## 5.2 Input Representation

The input to the neural network is a raw three-component seismic waveform tensor of shape `(Batch, 3, 1000)`. 

To maximize GPU utilization and avoid computational bottlenecks, the Short-Time Fourier Transform (STFT) is implemented directly within the model's forward pass using PyTorch's audio transforms. The STFT is configured with:
* `n_fft`: 128
* `win_length`: 128
* `hop_length`: 38
* `window_fn`: Hann window

Following the transformation, a logarithmic scaling $\log(1 + x)$ is applied to compress the dynamic range of the seismic energy. The resulting spectrogram tensor has the shape:

$$S \in \mathbb{R}^{3 \times F \times T}$$

Given the STFT configuration and input length, the parameters resolve exactly to $F = 65$ frequency bins and $T = 24$ time frames. This output tensor is treated similarly to a 3-channel RGB image, allowing the CNN backbone to extract meaningful spatial patterns.

## 5.3 CNN Backbone

### 5.3.1 Convolutional Block Structure

The CNN backbone consists of a sequence of convolutional blocks designed to distill the $65 \times 24$ spectrogram into a dense feature representation. Each of the three blocks follows a standard deep learning pattern:

```text
Conv2D → BatchNorm2D → ReLU → MaxPool2D
```

At the end of the convolutional sequence, an `AdaptiveAvgPool2D` layer guarantees a fixed spatial output shape of $1 \times 16$, ensuring downstream compatibility regardless of minor input length variations.

### 5.3.2 Layer Configuration

The exact configuration of the convolutional layers is structured to progressively increase channel depth while reducing spatial dimensions:

| Layer | Channels | Kernel Size | Stride | Padding | Pooling |
|-------|----------|-------------|--------|---------|---------|
| Conv1 | 16 | 5 × 3 | 1 | 2 × 1 | Max (2 × 1) |
| Conv2 | 32 | 3 × 3 | 1 | 1 × 1 | Max (2 × 2) |
| Conv3 | 64 | 3 × 3 | 1 | 1 × 1 | AdaptiveAvg (1 × 16) |

## 5.4 Feature Sequence Extraction

After passing through the CNN backbone and the adaptive pooling layer, the resulting feature map has the shape:

$$Z \in \mathbb{R}^{C' \times F' \times T'}$$

Specifically, $C' = 64$, $F' = 1$, and $T' = 16$. The frequency dimension is squeezed out, leaving a tensor of shape `(Batch, 64, 16)`. 

To feed this representation into the recurrent network, the tensor is permuted to swap the channel and time dimensions, yielding a sequence of vectors:

$$Z' \in \mathbb{R}^{T' \times D}$$

where $T' = 16$ time steps and $D = 64$ features. Each time step in the sequence represents the condensed spectral feature representation extracted from a specific segment of the input waveform.

## 5.5 LSTM Module

### 5.5.1 Motivation

While CNNs capture local time-frequency patterns in the spectrograms, they do not explicitly model the longer-term temporal evolution of the signal. Earthquake magnitude is strongly related to how seismic energy evolves over time (e.g., sustained increases in energy during the rupture phase).

### 5.5.2 LSTM Architecture

The model utilizes a single-layer Long Short-Term Memory (LSTM) network to process the sequence:

$$Z' = (z_1, z_2, \ldots, z_{16})$$

The LSTM has a hidden size of $H_{\text{lstm}} = 64$ and outputs a sequence of hidden states corresponding to each of the 16 time steps.

### 5.5.3 Directionality

The model intentionally uses a **unidirectional** LSTM with `batch_first=True` rather than a bidirectional one. While bidirectional networks often improve modeling capacity, they require access to future time steps. In a real-time Earthquake Early Warning (EEW) system, future data is not available. Therefore, causal inference dictates a unidirectional architecture.

## 5.6 Regression Head

The hidden state of the last LSTM time step (a 64-dimensional vector) is passed through a Multi-Layer Perceptron (MLP) to output the predicted magnitude.

The regression head has the following architecture:

| Layer | Type | Configuration |
|-------|------|---------------|
| 1 | Linear | 64 → 128 |
| 2 | LayerNorm | 128 |
| 3 | Activation | ReLU |
| 4 | Dropout | $p = 0.4$ |
| 5 | Linear | 128 → 64 |
| 6 | LayerNorm | 64 |
| 7 | Activation | ReLU |
| 8 | Dropout | $p = 0.4$ |
| 9 | Linear | 64 → 1 |

*Table 5.2. Architecture of the regression head.*

The final output layer produces the scalar magnitude estimate:

$$\hat{M} \in \mathbb{R}$$

No activation function is applied at the output node, as magnitude prediction is treated as a continuous regression task. The 40% dropout rate helps prevent overfitting on the dense layers.

> **Note on attention pooling.** An attention-pooling layer over the 16 LSTM outputs, in place of the last time step, was also tried. In the early ablation runs (`training_results/2026_4_6/mag_pred_*_{attention,baseline}`, single seed) it gave no measurable improvement: the best validation MAE was within noise of the baseline at 3, 5 and 8 seconds (e.g. 0.423 vs 0.424 at 5 s). It is therefore not part of the final architecture.

## 5.7 Learned Feature Representations

For a long time, deep learning models have been criticized as a "black box": numbers go in and numbers come out. The criticism is valid to some extent, but over the past decade researchers have developed methods to see what deep learning models learn during training. The approach used here is a 2D projection of the internal embedding space via Principal Component Analysis (PCA).

The proposed model can be divided into two logical parts: a CNN–LSTM backbone, which serves as a feature extractor, and an MLP regression head, which makes the final decision from the embeddings the backbone produces. To verify that the backbone extracts meaningful vector representations, these embeddings are analyzed.

PCA is a linear, deterministic dimensionality reduction method that transforms the features into orthogonal principal components ordered by the variance they capture. (Non-linear methods such as t-SNE and UMAP were not used: within-cluster density and between-cluster distances in their output depend strongly on their parameters and are easy to misread.) Given a centered matrix $X$ of the 64-dimensional LSTM output vectors, PCA computes the covariance matrix:

$$C = \frac{1}{n-1} X^T X$$

and its eigendecomposition:

$$C v_i = \lambda_i v_i$$

Projecting the data onto the two eigenvectors with the largest eigenvalues reduces the latent space to two dimensions while preserving meaningful distance relationships.

*Figure 5.1. Two-dimensional PCA projection of the 64-dimensional LSTM embeddings, colored by magnitude bucket: low ($M < 3$), mid ($M \in [3, 5]$), and high ($M \geq 5$). The first principal component accounts for 29.2% of the variance and the second for 14.2%. (Figure is in the final thesis PDF; it is produced by `_plot_pca` in `src/training_on_stratified_sample.py` and `notebooks/embedding_projection.ipynb`.)*

A clear magnitude gradient is visible along the first principal component: low-magnitude events concentrate on the right side of the projection, mid-magnitude events occupy the center, and high-magnitude events lie predominantly on the left. That this separation emerges from an unsupervised projection, even though PCA is unaware of the magnitude labels, suggests that the backbone maps spectrograms into a latent space on which the MLP can operate more easily.

## 5.8 Summary

This chapter detailed the architecture of the proposed CNN–LSTM–MLP model for earthquake magnitude estimation. The model computes spectrograms on-device before processing them through a three-block convolutional backbone. Temporal dynamics are captured via a unidirectional LSTM. Finally, a regression MLP translates the condensed representation into a single magnitude estimate.
