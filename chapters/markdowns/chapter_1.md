# Chapter 1 — Introduction

## 1.1 Motivation

In highly seismic active ares, like Kyrgyzstan, earthquakes still remain one of the most dangerous natural disasters, which endanger infrastracture and people. Up to this day, earthquakes are quite difficult to predict due to lack of explicit precautionary signals, that can be detected by traditional monitoring systems. As a result, communities located near active tectonic regions tend to experience catastrophic damage within the first seconds of strong ground motion. The consequences of large earthquakes on both economics and society are therefore severe, including deaths and substantial damage to infrastructure (houses, institutions and other building).

Earthquake Early Warning (EEW) systems are therefore created, developed and maintained to avoid these consequences by issuing alerts as soon as an earthquake begins, but before the most destructive waves reaches populated areas. The fundamental principle behind EEW is based on the difference in propagation speed between seismic waves. When an earthquake occurs, there are several types of seismic waves, primary waves (P-waves) and secondary waves (S-waves). P-waves propagate through the Earth's crust at faster, usually in the range of 5–8 km/s, whereas S-waves travel more slowly, at approximately 3–4 km/s, but mportantly, deal the most structural damage during earthquakes.

This difference in speed propagation provides a valuable time window during which early warning systems can detect the initial P-waves and estimate the scale of the upcoming event before the arrival of destructive waves. In EEW, even several seconds can enable safety protocols such as stopping trains, shutting down industrial processes, opening fire station doors, and issuing alerts to the public.

However, producing reliable warnings is a challenge. EEW systems must immediately detect seismic signals, determine whether they correspond to a true earthquake event, estimate its magnitude and location, and determine whether an alert should be issued. These computations and decistion must happen in near real time while operating on noisy data from sensors.

The challenge becomes extremely significant when attempt to estimate the final magnitude of an earthquake using only the first few seconds of recorded seismic data is made. Presice magnitude estimation is essential because alert thresholds are defined in terms of expected ground motion intensity for the most part, which correlates with earthquake magnitude. If the magnitude is underestimated, warnings may not be issued. Alternatively, overestimation may produce false alarms, undermining public trust in the warning systems.

Traditional EEW methods face some struggles when only early P-wave data is available. This limitation motivates the exploration of modern machine learning approaches that capable of learning from complicated and noisy data from seismic sensors.


## 1.2 The Core Challenge of Early Magnitude Estimation

To simplify, earthquake's magnitude represents the total energy released during rupture along a fault. The task of determining the precise magnitude requires gathering a substantial part of the seismic waveform, including both P-waves and S-waves recorded across multiple stations. In traditional seismological analysis, magnitude computation is therefore performed after the event has happened.

In an EEW context, however, rich information like this, is simply not recorded yet. Instead, the system must attempt to evaluate the final magnitude using only the earliest observations of the seismic signal. This, for the most part, consists of a short window beginning at the arrival of the P-wave and extending for only a few seconds.

More formally, consider a seismic waveform $x(t)$ recorded at a station. Let $t_P$ denote the arrival time of the P-wave. An EEW system may only have access to a short segment of the signal:

$$x(t), \quad t \in [t_P,\ t_P + W]$$

where $W$ is the available observation window, often between 3 and 8 seconds. The task is to estimate the final earthquake magnitude $M$ using only this limited data about the signal.

The problem is challenging on multiple levels; the earliest part of the waveform may not yet contain sufficient information about the full rupture process. In many earthquakes, the rupture begins locally and grows progressively along the fault plane. Consequently, the early waveform may appear similar for both small and large events, making it challenging to distinguish them using simple analytical methods.

Several studies have discussed whether the initial observations contain enough information to make a desicive verdict on the magnitude. Some earlier hypotheses suggested that the first few seconds of seismic signals may already contain distinguishable differences between small and large events. However, more recent researches indicate that reliable magnitude estimation from very short windows remains an open challenge.

Therefore, an effective EEW system must extract inexplicit patterns from the evolving waveform that may indicate the scale of the ongoing rupture. This is precisely the type of pattern recognition problem for which modern machine learning techniques are well suited.

## 1.3 Limitations of Classical EEW Approaches

Traditional EEW systems rely on deterministic signal processing algorithms and empirical relationships derived from seismological observations. Two widely used approache is the STA/LTA algorithm. Another system used PRESTo (Specialized Software deployed for regions of Kyrgyzstan) is Bayesian magnitude estimation framework.

### STA/LTA Detection

The Short-Term Average to Long-Term Average (STA/LTA) algorithm is one of the earliest methods for earthquake detection. The method computes two moving averages of the signal amplitude:

- a **short-term average** (STA) representing recent signal energy
- a **long-term average** (LTA) representing background noise

The detection statistic is defined as the ratio:

$$R(t) = \frac{\text{STA}(t)}{\text{LTA}(t)}$$

When this ratio exceeds a set threshold, the algorithm identifies the presence of a seismic event. While STA/LTA is effective for detecting sudden increases in signal energy, it does not provide robust estimates of earthquake magnitude, particularly in the early stages of the waveform.

### Bayesian Magnitude Estimation

A more sophisticated EEW framework, Bayesian method, estimates earthquake parameters from observations across multiple stations. These methods utilize the prior knowledge about earthquake distributions and update estimates of magnitude and location as additional data arrives.

Although Bayesian approaches improve estimation accuracy compared to simple signal processing techniques, they still depend heavily on empirical features extracted from the waveform, such as peak displacement or dominant period. These features often require observing a larger portion of the signal than is available in the earliest seconds after P-wave arrival.

Furthermore, classical systems typically rely on manually tuned features designed by the experts in seismic domain. Such features may fail to capture complex relationships present in seismic data.

## 1.4 The Potential of Deep Learning in EEW

Recent advances in machine learning, and more specifically, in deep learning have significantly extended the capabilities of signal analysis systems. Deep neural networks are able to capture feature representations from complex and highly correlated data, eliminating the need for manually engineered features.

In the context of seismology, deep learning has already demonstrated strong performance in tasks such as seismic phase detection, event classification, and waveform denoising. For example, neural network architectures have been developed to detect P-wave and S-wave arrivals outperfroming traditional signal processing methods.

The key advantage of deep learning approaches over traditional methods is in their ability to approximate complex nonlinear mappings between inputs and outputs. Formally, a neural network can be viewed as a function:

$$f_\theta : \mathbb{R}^n \to \mathbb{R}$$

parameterized by a set of learnable weights $\theta$ (also noted as $\omega$). During training, these parameters are changing to minimize a loss function (also called cost function) measuring the some sort of difference between network's predictions and the actual values.

When applied to seismic signals, neural networks can learn to recognize patterns in waveform data that correlate with earthquake magnitude. These patterns may involve subtle combinations of frequency content, temporal evolution, and signal amplitude that are difficult to capture using conventional methods.

Moreover, due to deep learning models' complexity (thousands and millions of parameters), they can extract meaningful features from highly correlated data and verbose data, one example of which are spectrograms. Several studies have shown a great potential in utlizing spectrograms' ability to make the input seismic signals more explicit and verbose.

## 1.5 Proposed Approach

This work proposes a deep learning pipeline for real-time earthquake magnitude estimation using the spectrograms of only early P-wave observations. The system operates on three-component seismic waveform data and predicts a scalar estimate of earthquake magnitude.

The proposed model architecture combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. The architecture is designed to capture both the spectral characteristics and temporal evolution of seismic signals.

The processing pipeline can be summarized as follows:

1. **Waveform acquisition** — Three-component seismic signals are collected from multiple global datasets.
2. **Time–frequency transformation** — The raw waveform is converted into a spectrogram, representing how signal energy is distributed across frequency and time.
3. **CNN feature extraction** — Convolutional neural networks process the spectrogram to identify local patterns in the time–frequency domain.
4. **Temporal modeling with LSTM** — The sequence of features extracted by the CNN is processed by an LSTM network to capture the temporal evolution of the seismic signal.
5. **Magnitude regression** — A fully connected layer produces a scalar estimate of earthquake magnitude.

The motivation behind this specific CNN–LSTM design is in the physical properties of seismic signals. Spectrograms represent the distribution of frequency content over time, which reflects the dynamics of the ruptures. CNNs have shown to be great at extracting local spatial patterns from such representations, while LSTMs, which are the consequest improvement over previosly introduced RNNs, capture temporal dependencies.

By combining these two paradigms, the system aims to infer earthquake magnitude from features present in the earliest stages of the waveform.

## 1.6 Scope and Limitations

This work focuses specifically on single-station magnitude estimation using early P-wave observations, for simplicity of the research. The study investigates whether deep learning models can reliably extract meaningful hidden representations from spectrograms and, consequently, estimate earthquake magnitude using the first observations of 3–8 seconds following P-wave arrival.

The analysis is conducted using publicly available seismic datasets. These datasets provide labeled waveform recordings and associated earthquake metadata necessary for supervised learning.

However, several limitations must be acknowledged:

- **Prospective evaluation** — The experimental results presented in this thesis are intended as a proof-of-concept investigation rather than a fully deployed operational EEW system.
- **Single-station analysis** — The proposed model processes waveforms from individual stations, while real-world EEW systems typically use observations from multiple stations to improve reliability and stability.
- **Data imbalance** — Earthquake magnitude distributions follow the Gutenberg–Richter law, meaning that large earthquakes occur far less frequently than small ones. This imbalance presents challenges for training machine learning models. Nevertheless, we will address this issue later on.
- **Regional variability** — Seismic waveforms depend on local geological conditions. Models trained on global datasets may require regional fine-tuning for optimal performance.

Despite these limitations, the proposed approach provides a framework for exploring the use of deep learning with the combination of spectrograms to early earthquake magnitude estimation.

## 1.7 Thesis Structure

The remainder of this thesis is organized as follows.

**Chapter 2** provides background information on seismological principles and reviews existing earthquake early warning systems, including both classical and machine learning approaches.

**Chapter 3** introduces the mathematical foundations of the proposed methodology. This chapter formally defines the signal processing techniques, neural network architectures, optimization methods, and evaluation metrics used in the study.

**Chapter 4** describes the datasets used in the experiments and outlines the preprocessing pipeline for converting raw seismic waveforms into spectrogram representations suitable for neural network input. It also touches on the topic of synthetic data generation to deal with previosly-mentioned data imbalance.

**Chapter 5** presents the architecture of the proposed CNN–LSTM model, including detailed descriptions of the convolutional backbone, recurrent module, and regression head.

**Chapter 6** outlines the experimental design and evaluation methodology used to assess the model's performance.

**Chapter 7** discusses the anticipated results, potential limitations, and implications of the proposed approach for earthquake early warning systems.

**Chapter 8** summarizes the key findings of the thesis and outlines directions for future research in machine-learning-based seismic monitoring.