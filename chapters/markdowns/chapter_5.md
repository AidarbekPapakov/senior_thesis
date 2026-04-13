# Chapter 5 — Model Architecture

## 5.1 Overview of the Model Pipeline

The goal of the proposed method is to estimate earthquake magnitude using only the first few seconds of seismic waveforms right after the arrival of the P-wave. Rather than relying on CPU-bound preprocessing, the model ingests raw temporal waveforms and dynamically computes spectrograms on the GPU.

The overall architecture combines a CNN for spatial feature extraction, an LSTM block for modeling temporal changes, and an attention mechanism to intelligently pool the temporal sequence before final regression.

The processing pipeline can be depicted as follows:

```text
Raw input waveforms
      ↓
On-device STFT & Log1p scaling
      ↓
CNN backbone (spatial feature extraction)
      ↓
LSTM (temporal modeling)
      ↓
Attention pooling (context vector aggregation)
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

## 5.6 Attention Pooling Mechanism

Rather than relying solely on the final hidden state of the LSTM—which may suffer from information bottlenecking—the model employs an Attention Pooling mechanism over all 16 time steps. 

A linear scoring layer projects each hidden state into a scalar weight. These weights are passed through a softmax function to create a probability distribution over the sequence:

$$\alpha_t = \text{softmax}(W \cdot h_t)$$

where $h_t$ is the LSTM output at time $t$, and $W$ is a learnable weight matrix. The final context vector $C$ is computed as the weighted sum of all LSTM hidden states:

$$C = \sum_{t=1}^{T'} \alpha_t h_t$$

This allows the network to dynamically focus on the most critical moments of the P-wave arrival when estimating the overall magnitude.

## 5.7 Regression Head

The attention-pooled context vector, containing the most salient seismic features, is passed through a Multi-Layer Perceptron (MLP) to output the predicted magnitude. 

The regression head utilizes the following architecture:

| Layer | Type | Configuration |
|-------|------|---------------|
| 1 | Linear | 64 → 64 |
| 2 | Activation | ReLU |
| 3 | Dropout | $p = 0.4$ |
| 4 | Linear | 64 → 1 |

The final output layer produces the scalar magnitude estimate:

$$\hat{M} \in \mathbb{R}$$

No activation function is applied at the output node, as magnitude prediction is treated as a continuous regression task. The inclusion of a robust 40% dropout rate helps prevent overfitting on the dense layers.

## 5.8 Summary

This chapter detailed the finalized architecture of the proposed CNN–LSTM model for earthquake magnitude estimation. 

The model optimizes inference by computing spectrograms directly on-device before processing them through a 3-block convolutional backbone. Temporal dynamics are captured via a unidirectional LSTM, and an attention pooling mechanism is introduced to intelligently aggregate the temporal sequence. Finally, a robust regression MLP translates this condensed representation into a single magnitude estimate.