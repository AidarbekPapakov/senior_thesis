# Chapter 4 — Dataset and Preprocessing

## 4.1 Overview

Although deep learning has shown to excel at processing complex and high-structure data, it still requires large amounts of data in order to perform at a decent level. In the context of our stud, we postulate that the training dataset must contain both the 3-component waveform recordings and the associated ground truth magnitude value for each event.

In our case two major datasets are used:

- the STanford EArthquake Dataset (STEAD).
- the INSTANCE dataset developed by the Istituto Nazionale di Geofisica e Vulcanologia (INGV).

The data used for testing our approach combines both of them in x:1-x proportion.

## 4.2 Data Sources

### 4.2.1 The STEAD Dataset

The STanford EArthquake Dataset (STEAD) is one of the largest open-sourced datasets of labeled seismic waveforms made specifically research in machine learning.

The dataset contains 1.2 million seismic recordings collected from stations around the world. Each waveform corresponds either to a detected earthquake signal or to background seismic noise.

Key characteristics of the dataset include:

- approximately 1.2 million waveform traces,
- includes three-component recordings (vertical, north–south, east–west),
- waveform length of  ~60 seconds,
- sampling rate of 100 Hz,
- annotated P-wave and S-wave arrival times, and
- associated event metadata including earthquake magnitude and hypocenter location.

Because of its size and high variability of the event locations, STEAD provides a diverse training set that includes earthquakes with representative sets of features.

However, as it is the case for the most seismic datasets, this suffers from a noticable magnitude "imbalance". Most events lay within the range of approximately magnitude 2.0 to 4.0, while earthquakes of magnitude >5.0 are relatively rare. Despite the fact this imbalance reflects the natural distribution of earthquakes, we still have to take care of large events' representation in the dataset as the main goal of this study is to understand to what extent the first seconds of P-waves and the ultimate magnitude are related. 

*[Figure 4.1: Example three-component seismic waveform from the STEAD dataset]*

### 4.2.2 The INSTANCE Dataset

The INSTANCE dataset is a large seismic waveform dataset developed by the Italian National Institute of Geophysics and Volcanology (INGV).

Unlike many automatically labeled seismic datasets, the INSTANCE dataset includes manually reviewed seismic phase picks performed by trained analysts. This manual verification significantly improves the reliability of arrival time annotations.

The dataset contains approximately 1.2 million waveform recordings, each with the following properties:

- three-component seismic traces,
- waveform duration of 120 seconds,
- sampling rate of 100 Hz,
- magnitude range approximately 0.0 to 6.5, and
- manually reviewed P-wave and S-wave arrival times.

Prior to publication, the dataset underwent several preprocessing steps performed by the dataset creators. These steps include removal of traces containing data gaps, trimming of waveform segments to consistent start times, resampling to a uniform sampling frequency, removal of mean offsets and linear trends, computation of signal-to-noise ratios, and extraction of quality control metrics.

The presence of carefully verified phase picks makes INSTANCE particularly valuable for training models that rely on accurate P-wave detection.

*[Figure 4.2: Example waveform segment from the INSTANCE dataset]*

### 4.2.3 CAIAG Regional Seismic Data

The Central Asian Institute for Applied Geosciences (CAIAG) maintains seismic monitoring networks across Kyrgyzstan and neighboring regions of Central Asia.

The seismic activity of this region is associated primarily with the collision between the Indian and Eurasian tectonic plates. As a result, the region experiences frequent moderate earthquakes and occasional large seismic events.

Incorporating CAIAG data into the training process provides several advantages. It allows the model to learn signal characteristics specific to the regional crustal structure. It enables evaluation of the model's performance on data collected in the geographic region where deployment may occur. It also reduces the domain shift between training and real-world operational conditions.

Compared to global datasets such as STEAD, the CAIAG dataset is expected to be smaller in size. For this reason, it is primarily used for fine-tuning and evaluation rather than large-scale pretraining.

*[Figure 4.3: Map of seismic station distribution in the CAIAG network]*

### 4.2.4 Dataset Combination Strategy

Training a deep learning model on a single dataset may lead to overfitting to the characteristics of that dataset. Differences in instrument response, geological structure, and noise conditions can cause models trained on one dataset to perform poorly on another.

To address this issue, the present study adopts a multi-stage training strategy.

First, the model is trained on a combined dataset constructed from the STEAD and INSTANCE datasets. These datasets provide large-scale coverage of seismic signals from diverse regions and therefore allow the model to learn generalizable representations.

Second, the model is fine-tuned using the CAIAG dataset in order to adapt it to the regional seismic characteristics of Central Asia.

This approach follows a common paradigm in machine learning known as **transfer learning**, in which a model trained on a large dataset is subsequently adapted to a smaller but more specialized dataset.

## 4.3 Preprocessing Pipeline

Raw seismic waveform data must undergo several preprocessing steps before it can be used as input for neural networks. These steps transform the waveform into a standardized representation suitable for model training.

### 4.3.1 P-wave Detection and Window Extraction

Earthquake early warning systems rely on the detection of the initial P-wave arrival. Once the P-wave arrival time $t_P$ is known, a short segment of the waveform following the arrival is extracted.

For a waveform $x(t)$, the analysis window is defined as:

$$x(t), \quad t \in [t_P,\ t_P + W]$$

where $W$ is the window length. In this study, several window lengths are considered:

$$W \in \{3, 5, 8\} \text{ seconds}$$

These windows correspond to realistic time intervals available in operational early warning systems. P-wave detection can be performed using algorithms such as PhaseNet or other automated phase picking methods.

*[Figure 4.4: Example waveform with P-wave arrival and extracted analysis window]*

### 4.3.2 Signal Normalization

Seismic waveform amplitudes can vary significantly depending on factors such as distance from the epicenter and instrument sensitivity. To ensure numerical stability during neural network training, waveform amplitudes must be normalized.

One common approach is z-score normalization, defined as:

$$x' = \frac{x - \mu}{\sigma}$$

where $\mu$ is the mean of the signal and $\sigma$ is the standard deviation. Normalization ensures that the input signals have approximately zero mean and unit variance.

### 4.3.3 Spectrogram Computation

After window extraction and normalization, the waveform is converted into a time–frequency representation using the Short-Time Fourier Transform described in Chapter 3.

Let the STFT parameters be: FFT size $N_{\text{fft}}$, hop length $H$, and window function $w(n)$. The resulting STFT coefficients are used to compute the log-power spectrogram:

$$S[m, k] = \log\!\left(1 + |X[m, k]|^2\right)$$

For three-component seismic signals, the spectrogram is computed separately for each channel.

### 4.3.4 Input Tensor Construction

The spectrograms from the three seismic channels are stacked to form a three-dimensional tensor:

$$\mathbf{X} \in \mathbb{R}^{3 \times F \times T}$$

where $3$ corresponds to the number of seismic channels, $F$ is the number of frequency bins, and $T$ is the number of time frames. This tensor serves as the input to the convolutional neural network.

*[Figure 4.5: Example spectrogram representation of a seismic signal]*

## 4.4 Dataset Splitting

To evaluate model performance reliably, the dataset must be divided into separate subsets used for training, validation, and testing. The dataset is partitioned as follows:

- **Training set:** 70% of the data
- **Validation set:** 15% of the data
- **Test set:** 15% of the data

To avoid bias in the magnitude distribution, the splitting procedure is performed using stratified sampling based on magnitude bins. This ensures that earthquakes of different magnitudes are represented proportionally in each dataset subset.

### 4.4.1 Magnitude Distribution

Because earthquake magnitudes follow the Gutenberg–Richter law, the dataset contains many more small events than large ones. Consequently, the magnitude distribution of the dataset is heavily skewed toward lower magnitudes.

*[Figure 4.6: Histogram of earthquake magnitude distribution across the dataset]*

This imbalance must be considered during model training to prevent the network from learning a trivial solution that performs well on small earthquakes but poorly on larger ones.

## 4.5 Data Augmentation and Synthetic Data

The scarcity of high-magnitude earthquakes presents a challenge for machine learning models. One possible approach to addressing this issue is the use of synthetic seismic data.

Synthetic waveforms can be generated using simulation tools that model wave propagation through geological media. One example is the SYNTHOSEIS framework, which generates realistic seismic signals based on physical models.

Synthetic data can be used to increase the number of training samples corresponding to large earthquakes, improving the model's ability to generalize across magnitude ranges. However, synthetic data must be used cautiously, as simulated waveforms may not fully capture the complexity of real seismic signals.

## 4.6 Summary

This chapter described the datasets and preprocessing steps used in the proposed earthquake magnitude estimation system.

The training dataset combines large global waveform datasets (STEAD and INSTANCE) with regional seismic data from CAIAG. This multi-source strategy allows the model to learn generalizable features while also adapting to regional seismic characteristics.

The preprocessing pipeline converts raw seismic waveforms into spectrogram representations suitable for neural network input. Key steps include P-wave detection, window extraction, normalization, and time–frequency transformation.

The next chapter introduces the architecture of the proposed deep learning model and describes how these preprocessed inputs are used to estimate earthquake magnitude in near real time.