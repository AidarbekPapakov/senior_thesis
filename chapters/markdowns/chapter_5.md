# Chapter 5 — Model Architecture

## 5.1 Overview of the Model Pipeline

The goal of the proposed model is to estimate earthquake magnitude using only the first few seconds of seismic waveform data following the arrival of the P-wave. The model operates on spectrogram representations derived from three-component seismic recordings.

The overall architecture combines convolutional neural networks for spatial feature extraction with a recurrent neural network for modeling temporal dynamics.

The processing pipeline can be summarized as:
```
Input waveform
      ↓
Spectrogram generation
      ↓
CNN backbone (spatial feature extraction)
      ↓
Feature sequence reshaping
      ↓
LSTM (temporal modeling)
      ↓
Fully connected regression head
      ↓
Predicted magnitude
```

Formally, the model can be expressed as a function:

$$\hat{M} = f_\theta(X)$$

where

- $X$ is the spectrogram tensor,
- $\theta$ represents the model parameters, and
- $\hat{M}$ is the predicted earthquake magnitude.

The architecture is designed to exploit two important properties of seismic signals:

- **Spectral structure** — frequency content varies during earthquake rupture.
- **Temporal evolution** — signal characteristics evolve during the first seconds after P-wave arrival.

Convolutional layers capture the spatial structure of the spectrogram, while the recurrent component models the temporal progression of extracted features.

*[Figure 5.1: Overview diagram of the CNN–LSTM architecture]*

## 5.2 Input Representation

The input to the neural network is a spectrogram computed from a three-component seismic waveform.

Following the preprocessing steps described in Chapter 4, the model input can be represented as:

$$X \in \mathbb{R}^{C \times F \times T}$$

where

- $C = 3$ represents the three seismic components,
- $F$ represents the number of frequency bins, and
- $T$ represents the number of time frames.

Typical values for these parameters depend on the STFT configuration and the observation window length $W$.

Placeholder values:

$$F = \text{[to be determined]}$$
$$T = \text{[depends on window size and hop length]}$$

The input tensor is treated analogously to a multi-channel image where the frequency dimension corresponds to image height and the time dimension corresponds to image width. This representation enables the use of convolutional neural networks for feature extraction.

## 5.3 CNN Backbone

### 5.3.1 Motivation

Spectrograms contain structured patterns that correspond to physical characteristics of seismic waves. For example:

- sudden broadband energy increases often correspond to P-wave arrivals,
- frequency bands may shift during rupture propagation, and
- noise and background vibrations produce characteristic textures.

CNNs are well suited for detecting such patterns because they learn local filters that respond to spatial structures in the input.

### 5.3.2 Convolutional Block Structure

The CNN backbone consists of a sequence of convolutional blocks. Each block has the following structure:
```
Conv2D → BatchNorm → ReLU → MaxPooling
```

This design is widely used in deep learning because it stabilizes training while progressively extracting higher-level features.

The architecture can be expressed as:
```
Input spectrogram
      ↓
Conv Block 1
      ↓
Conv Block 2
      ↓
Conv Block 3
      ↓
Conv Block [N]
      ↓
Feature tensor
```

Placeholder for number of blocks:

$$N = \text{[to be determined experimentally]}$$

### 5.3.3 Layer Configuration (Placeholder)

The exact configuration of convolutional layers will be determined through empirical experimentation. The following table illustrates the intended structure.

| Layer | Channels | Kernel Size | Stride | Output Shape |
|-------|----------|-------------|--------|--------------|
| Conv1 | [C1] | [k1 × k1] | [s1] | [ ] |
| Conv2 | [C2] | [k2 × k2] | [s2] | [ ] |
| Conv3 | [C3] | [k3 × k3] | [s3] | [ ] |
| Conv4 | [C4] | [k4 × k4] | [s4] | [ ] |

Placeholders to determine during experiments: number of channels $C_i$, kernel sizes $k_i$, pooling stride, and padding scheme.

### 5.3.4 Receptive Field

An important property of CNNs is the **receptive field**, which describes how large a region of the input affects a single output neuron.

For a sequence of convolution layers with kernel sizes $k_l$ and strides $s_l$, the receptive field after $L$ layers is:

$$r_L = 1 + \sum_{l=1}^{L} (k_l - 1) \prod_{j=1}^{l-1} s_j$$

The receptive field should ideally span a meaningful portion of the spectrogram, allowing the network to detect patterns across both time and frequency.

Placeholder:

$$r_L = \text{[to be computed after final architecture is chosen]}$$

## 5.4 Feature Sequence Extraction

After passing through the CNN backbone, the resulting feature map has the shape:

$$Z \in \mathbb{R}^{C' \times F' \times T'}$$

To feed this representation into an LSTM, the feature map must be converted into a sequence of vectors. This is achieved by collapsing the frequency dimension into the channel dimension:

$$Z' \in \mathbb{R}^{T' \times D}$$

where $D = C' \times F'$.

Each time step in the sequence therefore represents the spectral feature representation extracted from a specific time frame of the input spectrogram.

## 5.5 LSTM Module

### 5.5.1 Motivation

While CNNs capture spatial patterns in spectrograms, they do not explicitly model the temporal evolution of the signal.

However, earthquake magnitude is strongly related to the way seismic energy evolves over time. For example, large earthquakes often exhibit sustained increases in signal energy during the early rupture phase. An LSTM network can capture such temporal dependencies.

### 5.5.2 LSTM Architecture

The LSTM receives the sequence:

$$Z' = (z_1, z_2, \ldots, z_{T'})$$

where each vector $z_t$ represents features extracted from the CNN. The LSTM processes the sequence iteratively according to the equations introduced in Chapter 3.

Placeholder configuration:

| Parameter | Value |
|-----------|-------|
| LSTM layers | [to be determined] |
| Hidden size | [$H_{\text{lstm}}$] |
| Dropout | [$p_{\text{dropout}}$] |

### 5.5.3 Directionality

The model uses a **unidirectional** LSTM rather than a bidirectional one.

Bidirectional recurrent networks process sequences in both forward and backward directions. While this improves modeling capacity, it requires access to future time steps. In a real-time earthquake early warning system, future data is not available. Therefore, a unidirectional LSTM is the appropriate architecture for causal inference.

## 5.6 Regression Head

The final hidden state of the LSTM contains a condensed representation of the seismic signal. This vector is passed through a small fully connected network that outputs the predicted earthquake magnitude.

Proposed structure:
```
FC → ReLU → FC → Output
```

Placeholder architecture:

| Layer | Size |
|-------|------|
| FC1 | $H_{\text{lstm}} \to H_{\text{fc}}$ |
| FC2 | $H_{\text{fc}} \to 1$ |

The final output layer produces the scalar magnitude estimate:

$$\hat{M} \in \mathbb{R}$$

No activation function is applied at the output layer since magnitude prediction is a regression task.

## 5.7 Parameter Count

The total number of model parameters depends on the final architecture configuration. In general, the parameter count can be approximated as:

$$P_{\text{total}} = P_{\text{CNN}} + P_{\text{LSTM}} + P_{\text{FC}}$$

where $P_{\text{CNN}}$ is the number of convolutional parameters, $P_{\text{LSTM}}$ is the number of recurrent parameters, and $P_{\text{FC}}$ is the number of parameters in the regression head.

Placeholder:

$$P_{\text{total}} = \text{[to be computed after architecture selection]}$$

## 5.8 Computational Complexity and Inference Time

For earthquake early warning applications, inference latency is an important consideration. Let $T_{\text{CNN}}$ denote the time required for CNN inference, $T_{\text{LSTM}}$ denote the time required for LSTM processing, and $T_{\text{FC}}$ denote the time required for the regression head. The total inference time is:

$$T_{\text{total}} = T_{\text{CNN}} + T_{\text{LSTM}} + T_{\text{FC}}$$

A practical EEW system typically requires inference to occur within a fraction of a second.

Placeholder target:

$$T_{\text{total}} < \text{[target latency]}$$

## 5.9 Summary

This chapter presented the architecture of the proposed CNN–LSTM model for earthquake magnitude estimation.

The model processes spectrogram representations of three-component seismic waveforms using convolutional layers to extract spatial features and a recurrent network to model temporal dynamics.

Because the optimal architecture configuration depends on experimental evaluation, several hyperparameters—including convolutional channel counts, kernel sizes, and LSTM dimensions—are intentionally left as placeholders. These values will be determined through systematic experimentation described in the next chapter.