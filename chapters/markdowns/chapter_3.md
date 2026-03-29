# Chapter 3 — Mathematical Foundations

## 3.1 Overview

The proposed method consists of several components each motivated by the advances made by the previous works in seismic field as well as the other fields, such as image processing and automatic speech recognition.S

The processing pipeline can be summarized as:

$$\text{Three-Component Waveform} \to \text{Spectrogram Representations} \to \text{CNN} \to \text{LSTM} \to \text{MLP}$$

The spectrogram provides a time–frequency representation of the seismic signalы. Convolutional neural networks serve as a spatial features extractor, while recurrent neural networks catch temporal dependencies. Lastly, a multi-layer perceptron predicts the earthquake magnitude from the features extracted and transformed along the way.

This chapter introduces the mathematical foundations underlying these components.

## 3.2 Signal Representation: From Waveform to Spectrogram

Seismic sensors record signals in time domain. For a single seismic channel, the recorded waveform can be represented as a function $x(t)$, where $t$ denotes time and $x(t)$ represents ground displacement, velocity, or acceleration depending on the instrument type.

Although time-domain waveforms contain all information about the seismic signal, due to the noise, introduced by the infrastructural movements, many characteristics of earthquakes are better understood in the frequency domain. Thus mapping the input data into a time–frequency domain is quite an essential task.

### 3.2.1 The Short-Time Fourier Transform

The classical Fourier transform converts a time-domain signal into its frequency representation:

$$X(\omega) = \int_{-\infty}^{\infty} x(t)\, e^{-j\omega t}\, dt$$

where $\omega$ is angular frequency and $j = \sqrt{-1}$.

However, there is an important assumption, which is the signal is stationary and persistent over the entire time of observation, meaning its properties do not change over time. However, seismic signals are non-stationary, thus the properties change notably during the progression of the earthquake.

To capture these changes, the **Short-Time Fourier Transform** (STFT) is used. The STFT utilizes the concept of a sliding window, more specifically it applies the Fourier transform to short overlapping windows of the signal:

$$X(\tau, \omega) = \int_{-\infty}^{\infty} x(t)\, w(t - \tau)\, e^{-j\omega t}\, dt$$

where

- $x(t)$ is the input signal,
- $w(t - \tau)$ is a window function centered at time $\tau$, and
- $\omega$ represents frequency.

The window function isolates a short segment of the signal so that the frequency content can be analyzed locally in time.

However, both the Fourier transform and STFT are applicable to continously defined functions, while, in practice, seismic signal is a discrete sequence, with sampling interval $T_s$. The discrete STFT is therefore used and it is defined as the following:

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

The STFT produces complex-valued coefficients. In machine learning, the magnitude of these coefficients is used to compute the power spectrogram:

$$S[m, k] = |X[m, k]|^2$$

This representation describes how signal energy is distributed across time and frequency.

Seismic signals are distributed across different orders of magnitude in amplitude. To compress this range, a logarithmic transformation is thus applied:

$$S_{\log}[m, k] = \log(1 + S[m, k])$$

The resulting log-power spectrogram helps with the numerical representation, which otherwise would be disasterous for neural network because of models' training nature.

### 3.2.4 Spectrograms vs MFCC Features

Another approach used for time-frequency representation of any kind of signals is **Mel-Frequency Cepstral Coefficients** (MFCCs). 
MFCC computation involves the following steps:

1. Compute the power spectrogram.
2. Apply a bank of triangular filters spaced on the Mel scale.
3. Compute the logarithm of filterbank energies.
4. Apply a discrete cosine transform (DCT).

The Mel scale is defined as:

$$\text{mel}(f) = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)$$

Due to the fact, that MFCCs have been historically used for the task of automatic speech recognition (speech transcribing, emotion detection, diarization), this scale approximates human auditory perception, emphasizing lower frequencies and compressing higher ones.

While MFCCs are effective for speech recognition, they are not well suited for seismic signals. The Mel filterbank distorts the physical frequency structure of the signal, potentially removing information relevant to earthquake dynamics. Additionally, the last step of MFCC computation, DCT, can be viewed as the denoising technique, which was useful because models prior to bloom of Deep Leaning were not able to process such a complex data representation.

In contrast, spectrograms preserve the full frequency resolution of the seismic waveform. This allows the neural network to learn which frequency bands are informative for magnitude estimation.

For this reason, we are using spectrogram representations in out system.

## 3.3 Convolutional Neural Networks

Once the seismic waveforms have been converted into spectrograms, the resulting representations can be viewed as two-dimensional images with axes corresponding to time and frequency.

Convolutional Neural Networks (CNNs) were the pioneers in sucessful image processing, which are the same spatial data with two axes.

### 3.3.1 Convolution Operation

Consider an input feature map $X \in \mathbb{R}^{H \times W}$, where height and width of the input matrix are denoted as $H$ and $W$ .

A convolution kernel $K \in \mathbb{R}^{k_h \times k_w}$ slides across the input matrix and computes weighted sums:

$$(X * K)[i, j] = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} X[i+m,\, j+n]\, K[m, n]$$

The output of the convolution operation is a feature map, which in practice has shown to highlight patterns detected by the kernel.

In CNNs with multiple channels, each output channel is computed as:

$$Z_c^{(l)} = b_c + \sum_{k=1}^{C_{\text{in}}} K_{c,k}^{(l)} * X_k^{(l-1)}$$

where

- $C_{\text{in}}$ is the number of input channels,
- $K_{c,k}^{(l)}$ is the kernel connecting channel $k$ to output channel $c$, and
- $b_c$ is a bias term.

### 3.3.2 Activation Functions

After convolution, nonlinear activation functions are applied to introduce nonlinearity into the network.

The activation function used in the beginning of the rise of Machine Learning was Sigmoid function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Neveretheless, it is almost never used in modern days, due to the problem, called vanishing gradients.

Instead, a commonly used activation function is the Rectified Linear Unit (ReLU), which is simply defined as follows:

$$f(z) = \max(0, z)$$

ReLU activation avoids saturation effects present in sigmoid or hyperbolic tangent functions, reducing the risk of vanishing gradients during training.
Besides ReLU, there are other activation functions, but ReLU has shown to perform exceptionally well for small-medium architectures.

### 3.3.3 Pooling Operations

Pooling layers reduce the spatial resolution of feature maps while retaining the most important information.

In max pooling, the output is defined as:

$$P[i, j] = \max_{0 \leq m, n < p} Z[i \cdot s + m,\; j \cdot s + n]$$

where $p$ is the pooling kernel size and $s$ is the stride.

Pooling provides a simple way to reduce the size of the feature maps, which consequently, reduces the computational overhead.

### 3.3.4 Batch Normalization

Another technique that has become indespensible in deep networks is Batch Normalization.

Batch normalization addresses this issue by normalizing activations across a mini-batch:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

where $\mu_B$ is the batch mean and $\sigma_B^2$ is the batch variance. $\epsilon$ is used to avoid zero-division.

The normalized activations are then scaled and shifted using learnable parameters. This process stabilizes gradient propagation and accelerates model's convergence.

### 3.3.5 Dropout

To overcome the phenomenon of overfitting (when the model performs well on training set and poorly on validation and test sets), **Dropout**, a regularization technique, is introduced. 

During training, each neuron is randomly deactivated (does not output anything) with probability $p$. At inference time, all neurons are used but their outputs are scaled by $1 - p$.

Dropout has shown to improve generalization performance across the multitude of architectures and networks.

## 3.4 Recurrent Neural Networks and LSTMs

While CNNs are capable of capturing spatial patterns in spectrograms, the processed signals still posses the time-dependent component in themselves. Modeling these temporal dynamics requires sequential neural architectures.

### 3.4.1 Vanilla Recurrent Neural Networks

A recurrent neural network, initially proposed for processing text, which is essentially a sequence of data, is viewed as follows:

For input sequence $x_1, x_2, \ldots, x_T$, the hidden state update is:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

where $h_t$ is the hidden state at time $t$.

The main limitation of vanila RNNs is that during backpropagation, gradients propagate through many sequential steps, and without any supporting elements, this process often leads to the **vanishing gradient problem**, where gradients decrease asymptotically and learning becomes ineffective.

### 3.4.2 Long Short-Term Memory Networks

To solve the problem of vanishing gradients, Long Short-Term Memory (LSTM) networks were introduced.

The missing from vanila RNNs supporting elements were added to a LSTM block. It maintains an internal cell state $c_t$ that stores long-term information. The cell uses several gating mechanisms:

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

These mechanisms allow the network to selectively retain or discard information over long time intervals.

### 3.4.3 CNN–LSTM-MLP Integration

In the proposed architecture, the CNN serves as a backbone that processes the spectrograms and produces 3 feature vectors that eventually aggregated and passed to the LSTM, which processes this sequence to model the temporal evolution. This combination enables the model to capture both spatial patterns and temporal dependencies. In the end, the resulting feature vector is passed to multi layer perceptron that ultimatevily solves the taks of predicting a scalar value.

## 3.5 Loss Functions

The objective of the model is to estimate the earthquake magnitude $M$. The predicted magnitude is denoted $\hat{M}$. To make a model learn the data, we have to introduce 
a **Loss Function** (also called **Cost Function** in other literature). This function measures discrepancy betwen the actual value (ground truth) and model's output (predicted value), its purpose to punish the model for misprediction and change the weights $\theta$. The lower the loss function, the more accurate we are with the predictions and the less the model's parameters, $\theta$, are updated.

The most simple loss function used for predicting scalar values is:

**Mean Squared Error:**
$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{M}_i - M_i)^2$$


As the derivative of $f(x)=x^2$ is universally defined, there are no problems occuring with gradient computation. 
<!-- Another function that can be used with reservation is:

**Mean Absolute Error:**
$$\mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |\hat{M}_i - M_i|$$

It is a known fact, that the derivative of $f(x) = |x|$, is not universally defined, and in that case, subgradients have to be used to make up for this loss.

A compromise between these is the **Huber loss**, defined as:

$$\mathcal{L}_\delta =
\begin{cases}
\dfrac{1}{2}(\hat{M} - M)^2 & |\hat{M} - M| \leq \delta \\[6pt]
\delta\!\left(|\hat{M} - M| - \dfrac{\delta}{2}\right) & \text{otherwise}
\end{cases}$$

Huber loss behaves like MSE for small errors and MAE for large errors, making it robust to outliers. -->

## 3.6 Optimization

Neural network parameters are optimized using gradient-based learning. The basic gradient descent update rule is:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

where $\theta$ represents model parameters and $\eta$ is the learning rate.

All of the modern deep learning models use the variations of gradient descent. The most common ones are: Stochastic Gradient Descent (SGD), Mini-Batch Gradient Descent (mBGD), 
Adagrad, RMSProp, Adam, and AdamW.

In this work, **AdamW optimizer** is utilized, which maintains adaptive learning rates using estimates of first and second gradient moments.

## 3.7 Magnitude Distribution and Class Imbalance

As we have mentioned before, earthquake magnitudes follow the Gutenberg–Richter law:

$$\log_{10} N = a - bM$$

This relationship implies that small earthquakes occur far more frequently than large ones. For example, there may be roughly ten times more earthquakes of magnitude 3 than magnitude 4, and ten times more magnitude 4 earthquakes than magnitude 5 events.

This imbalance is a challenge for machine learning models, since the training data is skewed towards small events. Possible sollutions include:

- magnitude-based sampling,
- weighted loss functions, and
- synthetic data generation.

## 3.8 Evaluation Metrics

Lastly, to evaluate how good a model is performing, divers metrics are needed. Solving a task of predicting as scalar value limits us to following metrics:

**Root Mean Squared Error:**
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{M}_i - M_i)^2}$$

**Mean Absolute Error:**
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{M}_i - M_i|$$

**Coefficient of determination:**
$$R^2 = 1 - \frac{\sum_{i=1}^{N} (\hat{M}_i - M_i)^2}{\sum_{i=1}^{N} (M_i - \bar{M})^2}$$

**Pearson correlation coefficient:**
$$r = \frac{\displaystyle\sum_{i=1}^{N} (\hat{M}_i - \bar{\hat{M}})(M_i - \bar{M})}{\sqrt{\displaystyle\sum_{i=1}^{N} (\hat{M}_i - \bar{\hat{M}})^2 \cdot \sum_{i=1}^{N} (M_i - \bar{M})^2}}$$

There are certainly more metrics used in regression tasks, but the ones listed are used the most frequently and quite easy to interpret.

## 3.9 Summary

This chapter presented the mathematical principles behind the proposed magnitude estimation system.

The Short-Time Fourier Transform provides a time–frequency representation of seismic signals. Convolutional neural networks extract spatial patterns from spectrograms, while recurrent networks model temporal dependencies. Regression model then maps these learned features to a scalar value, which is an estimate of an earthquake magnitude.

The next chapter describes the datasets used in this study and the preprocessing steps required to prepare seismic waveform data for model training.