# Chapter 3 — Mathematical Foundations

## 3.1 Overview

The proposed earthquake magnitude estimation system transforms raw seismic waveform recordings into a representation suitable for deep learning and then applies a neural architecture capable of modeling both spectral and temporal characteristics of the signal.

The processing pipeline can be summarized as:

$$\text{Waveform} \to \text{Spectrogram} \to \text{CNN} \to \text{LSTM} \to \text{Regression Output}$$

Each component of this pipeline is supported by well-established mathematical principles. The spectrogram provides a time–frequency representation of the seismic signal. Convolutional neural networks extract spatial features from this representation, while recurrent neural networks model temporal dependencies. Finally, a regression model predicts the earthquake magnitude.

This chapter introduces the mathematical foundations underlying these components.

## 3.2 Signal Representation: From Waveform to Spectrogram

Seismic sensors record ground motion as time-domain signals. For a single seismic channel, the recorded waveform can be represented as a function $x(t)$, where $t$ denotes time and $x(t)$ represents ground displacement, velocity, or acceleration depending on the instrument type.

Although time-domain waveforms contain all information about the seismic signal, many characteristics of earthquakes are better understood in the frequency domain. For example, rupture processes often generate characteristic frequency patterns related to fault size and energy release.

To analyze how frequency content evolves over time, it is necessary to construct a time–frequency representation of the signal.

### 3.2.1 The Short-Time Fourier Transform

The classical Fourier transform converts a time-domain signal into its frequency representation:

$$X(\omega) = \int_{-\infty}^{\infty} x(t)\, e^{-j\omega t}\, dt$$

where $\omega$ is angular frequency and $j = \sqrt{-1}$.

However, this transform assumes the signal is stationary over the entire observation interval, meaning its statistical properties do not change with time. Seismic signals are strongly non-stationary, since the frequency content changes significantly during the progression of the earthquake.

To capture these time-dependent spectral properties, the Short-Time Fourier Transform (STFT) is used.

The STFT applies the Fourier transform to short overlapping windows of the signal:

$$X(\tau, \omega) = \int_{-\infty}^{\infty} x(t)\, w(t - \tau)\, e^{-j\omega t}\, dt$$

where

- $x(t)$ is the input signal,
- $w(t - \tau)$ is a window function centered at time $\tau$, and
- $\omega$ represents frequency.

The window function isolates a short segment of the signal so that the frequency content can be analyzed locally in time.

In practice, seismic signals are recorded as discrete samples $x[n]$ with sampling interval $T_s$. The discrete STFT is therefore defined as:

$$X[m, k] = \sum_{n=0}^{N-1} x[n]\, w[n - mH]\, e^{-j2\pi kn/N}$$

where

- $m$ is the frame index (time window),
- $k$ is the frequency bin index,
- $H$ is the hop length (stride between windows), and
- $N$ is the FFT size.

This transform produces a complex-valued matrix describing the signal energy at different time and frequency locations.

### 3.2.2 Time–Frequency Resolution Tradeoff

A fundamental limitation of time–frequency analysis arises from an uncertainty principle analogous to the Heisenberg uncertainty principle in quantum mechanics.

The time resolution $\Delta t$ and frequency resolution $\Delta f$ of a signal representation satisfy:

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi}$$

This inequality implies that increasing time resolution necessarily decreases frequency resolution, and vice versa.

In the context of seismic signals sampled at approximately 100 Hz, the choice of window length must balance these competing requirements. A longer window provides better frequency resolution but reduces temporal precision. A shorter window improves time localization but increases spectral uncertainty.

Because earthquake early warning systems rely on short observation windows (typically 3–8 seconds), the STFT parameters must be chosen carefully to preserve relevant signal structure.

### 3.2.3 Power Spectrogram

The STFT produces complex-valued coefficients. For most machine learning applications, the magnitude of these coefficients is used to compute the power spectrogram:

$$S[m, k] = |X[m, k]|^2$$

This representation describes how signal energy is distributed across time and frequency.

Seismic signals often span several orders of magnitude in amplitude. To compress this dynamic range, a logarithmic transformation is applied:

$$S_{\log}[m, k] = \log(1 + S[m, k])$$

The resulting log-power spectrogram provides a stable numerical representation suitable for neural network input.

### 3.2.4 Spectrograms vs MFCC Features

In audio signal processing, features known as Mel-Frequency Cepstral Coefficients (MFCCs) are commonly used. MFCC computation involves the following steps:

1. Compute the power spectrogram.
2. Apply a bank of triangular filters spaced on the Mel scale.
3. Compute the logarithm of filterbank energies.
4. Apply a discrete cosine transform (DCT).

The Mel scale is defined as:

$$\text{mel}(f) = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)$$

This scale approximates human auditory perception, emphasizing lower frequencies and compressing higher ones.

While MFCCs are effective for speech recognition, they are not well suited for seismic signals. The Mel filterbank distorts the physical frequency structure of the signal, potentially removing information relevant to earthquake dynamics.

In contrast, spectrograms preserve the full frequency resolution of the seismic waveform. This allows the neural network to learn which frequency bands are informative for magnitude estimation without imposing assumptions based on human hearing.

For this reason, spectrogram representations are widely used in modern seismic deep learning systems.

## 3.3 Convolutional Neural Networks

Once the seismic waveform has been converted into a spectrogram, the resulting representation can be treated as a two-dimensional image with axes corresponding to time and frequency.

Convolutional Neural Networks (CNNs) are particularly well suited for extracting patterns from such structured data.

### 3.3.1 Convolution Operation

Consider an input feature map $X \in \mathbb{R}^{H \times W}$, where $H$ and $W$ denote the height and width of the input matrix.

A convolution kernel $K \in \mathbb{R}^{k_h \times k_w}$ slides across the input matrix and computes weighted sums:

$$(X * K)[i, j] = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} X[i+m,\, j+n]\, K[m, n]$$

The output of the convolution operation is a feature map highlighting patterns detected by the kernel.

In CNNs with multiple channels, each output channel is computed as:

$$Z_c^{(l)} = b_c + \sum_{k=1}^{C_{\text{in}}} K_{c,k}^{(l)} * X_k^{(l-1)}$$

where

- $C_{\text{in}}$ is the number of input channels,
- $K_{c,k}^{(l)}$ is the kernel connecting channel $k$ to output channel $c$, and
- $b_c$ is a bias term.

This process enables the network to learn filters that detect characteristic patterns in the spectrogram.

### 3.3.2 Activation Functions

After convolution, nonlinear activation functions are applied to introduce nonlinearity into the network.

A commonly used activation is the Rectified Linear Unit (ReLU):

$$f(z) = \max(0, z)$$

ReLU activation avoids saturation effects present in sigmoid or hyperbolic tangent functions, reducing the risk of vanishing gradients during training.

### 3.3.3 Pooling Operations

Pooling layers reduce the spatial resolution of feature maps while retaining the most important information.

In max pooling, the output is defined as:

$$P[i, j] = \max_{0 \leq m, n < p} Z[i \cdot s + m,\; j \cdot s + n]$$

where $p$ is the pooling kernel size and $s$ is the stride.

Pooling provides a form of translation invariance and reduces computational complexity.

### 3.3.4 Batch Normalization

Deep networks can suffer from unstable training due to changes in the distribution of intermediate activations.

Batch normalization addresses this issue by normalizing activations across a mini-batch:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

where $\mu_B$ is the batch mean and $\sigma_B^2$ is the batch variance.

The normalized activations are then scaled and shifted using learnable parameters. This process stabilizes gradient propagation and accelerates convergence.

### 3.3.5 Dropout

Dropout is a regularization technique used to reduce overfitting.

During training, each neuron is randomly deactivated with probability $p$. At inference time, all neurons are used but their outputs are scaled by $1 - p$.

Dropout effectively trains an ensemble of subnetworks and improves generalization performance.

## 3.4 Recurrent Neural Networks and LSTMs

While CNNs capture spatial patterns in spectrograms, earthquake signals evolve over time. Modeling these temporal dynamics requires sequential neural architectures.

### 3.4.1 Vanilla Recurrent Neural Networks

A recurrent neural network processes sequential data by maintaining a hidden state that evolves over time.

For input sequence $x_1, x_2, \ldots, x_T$, the hidden state update is:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

where $h_t$ is the hidden state at time $t$.

During backpropagation through time, gradients propagate through many sequential steps. This process often leads to the **vanishing gradient problem**, where gradients decrease exponentially and learning becomes ineffective.

### 3.4.2 Long Short-Term Memory Networks

Long Short-Term Memory (LSTM) networks were introduced to address the limitations of standard recurrent networks.

An LSTM cell maintains an internal cell state $c_t$ that stores long-term information. The cell uses several gating mechanisms:

**Forget gate:**
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**Input gate:**
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

**Candidate cell state:**
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$

**Cell state update:**
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Output gate:**
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

**Hidden state:**
$$h_t = o_t \odot \tanh(c_t)$$

These gating mechanisms allow the network to selectively retain or discard information over long time intervals.

### 3.4.3 CNN–LSTM Integration

In the proposed architecture, the CNN processes the spectrogram and produces a sequence of feature vectors. The LSTM then processes this sequence to model the temporal evolution of spectral features. This combination enables the model to capture both spatial patterns and temporal dependencies.

## 3.5 Regression and Loss Functions

The objective of the model is to estimate the earthquake magnitude $M$. The predicted magnitude is denoted $\hat{M}$. This is therefore a regression problem.

Two common loss functions are:

**Mean Squared Error:**
$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{M}_i - M_i)^2$$

**Mean Absolute Error:**
$$\mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |\hat{M}_i - M_i|$$

A compromise between these is the **Huber loss**, defined as:

$$\mathcal{L}_\delta =
\begin{cases}
\dfrac{1}{2}(\hat{M} - M)^2 & |\hat{M} - M| \leq \delta \\[6pt]
\delta\!\left(|\hat{M} - M| - \dfrac{\delta}{2}\right) & \text{otherwise}
\end{cases}$$

Huber loss behaves like MSE for small errors and MAE for large errors, making it robust to outliers.

## 3.6 Optimization

Neural network parameters are optimized using gradient-based learning. The basic gradient descent update rule is:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

where $\theta$ represents model parameters and $\eta$ is the learning rate.

Modern deep learning models typically use the **Adam optimizer**, which maintains adaptive learning rates using estimates of first and second gradient moments.

## 3.7 Magnitude Distribution and Class Imbalance

Earthquake magnitudes follow the Gutenberg–Richter law:

$$\log_{10} N = a - bM$$

This relationship implies that small earthquakes occur far more frequently than large ones. For example, there may be roughly ten times more earthquakes of magnitude 3 than magnitude 4, and ten times more magnitude 4 earthquakes than magnitude 5 events.

This imbalance poses challenges for machine learning models, since the training data is dominated by small events. Possible mitigation strategies include:

- magnitude-based sampling,
- weighted loss functions, and
- synthetic data generation.

## 3.8 Evaluation Metrics

To evaluate the model's performance, several regression metrics are used.

**Root Mean Squared Error:**
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{M}_i - M_i)^2}$$

**Mean Absolute Error:**
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{M}_i - M_i|$$

**Coefficient of determination:**
$$R^2 = 1 - \frac{\sum_{i=1}^{N} (\hat{M}_i - M_i)^2}{\sum_{i=1}^{N} (M_i - \bar{M})^2}$$

**Pearson correlation coefficient:**
$$r = \frac{\displaystyle\sum_{i=1}^{N} (\hat{M}_i - \bar{\hat{M}})(M_i - \bar{M})}{\sqrt{\displaystyle\sum_{i=1}^{N} (\hat{M}_i - \bar{\hat{M}})^2 \cdot \sum_{i=1}^{N} (M_i - \bar{M})^2}}$$

These metrics provide complementary perspectives on prediction accuracy.

## 3.9 Summary

This chapter presented the mathematical principles underlying the proposed magnitude estimation system.

The Short-Time Fourier Transform provides a time–frequency representation of seismic signals. Convolutional neural networks extract spatial patterns from spectrograms, while recurrent networks model temporal dependencies. Regression models then map these learned features to earthquake magnitude estimates.

The next chapter describes the datasets used in this study and the preprocessing steps required to prepare seismic waveform data for model training.