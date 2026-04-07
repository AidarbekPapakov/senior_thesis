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
- three-component recordings (vertical, north–south, east–west),
- waveform length of  ~60 seconds,
- sampling rate of 100 Hz,
- annotated P-wave and S-wave arrival times, and
- associated event metadata including earthquake magnitude and hypocenter location.

STEAD was chosen to be one of our primary data sources because of its size and high variability of the event locations, which grants us an access to earthquakes with representative sets of features.

However, as it is the case for the most seismic datasets, STEAD "suffers" from a noticable skew towards lower-magnitude earthquakes. Most events lay within the range of approximately magnitude 2.0 to 4.0, while earthquakes of magnitude >5.0 are relatively rare. Despite the fact this imbalance reflects the natural distribution of earthquakes, we still have to take care of large events' representation in the dataset as the main goal of this study is to understand to what extent the first seconds of P-waves and the magnitude are related. 

*[Figure 4.1: Example three-component seismic waveform from the STEAD dataset]*

### 4.2.2 The INSTANCE Dataset

As it is the case for STEAD, The INSTANCE dataset is also intended for machine learning experiments. The dataset developed by the Italian National Institute of Geophysics and Volcanology (INGV).

And just as the STEAD dataset, this one contains ~1.2 million waveform recordings as well, each with the following properties:

- three-component recordings (vertical, north–south, east–west),
- waveform duration of 120 seconds,
- sampling rate of 100 Hz,
- magnitude range approximately 0.0 to 6.5, and
- manually reviewed P-wave and S-wave arrival times.

As it is stated by the authors, the dataset underwent several preprocessing steps: 

1) removal of traces containing data gaps
2) trimming of waveform segments to consistent start times
3) resampling to a uniform sampling frequency 
4) removal of mean offsets and linear trends
5) computation of signal-to-noise ratios, and extraction of quality control metrics.

*[Figure 4.2: Example waveform segment from the INSTANCE dataset]*


### 4.2.4 Dataset Combination Strategy

Before we start the conversation on the datasets we have tested, we need to clarify one important thing. As it was mentioned, both datasets' metadata include the information on the type of phase picking: manual and automated. In this study, we only take into account the observations which P and S arrivals were labelled by the experts in the field. Although, it was mentioned in literature review that modern machine learning approaches manage phase picking quite well, we still decided to omit the observations automated labels due to potential introduction of bias into our training data. It is in general a good practice, for any machine learning problem, to aim to use the data which labels are fact-checked by the human experts even if the initial annotaion was not made by a human.

Moving to the datasets themselves, the authors of INSTANCE dataset provide a sample of the whole dataset that contains only 10,000 observations, which after our filtering is left with ~8,000 data instances. But as we have discussed with the STEAD dataset, this sample is heavily skewed towards lower bound of magnitude, which is not a bad thing, but in the context of our study is not quite the aim. Nevertheless, the proposed model was still tested on this sample. 

The following part might seem a bit controversial and counter-intuitive at first, but we will explain the choices we have made. To test out approach, we sampled a portion of data from both datasets in even proportions. Eventually, we have a dataset consisting of ~50,000 (~25,000 from INSTANCES and `25,000 from STEAD) actual events, each being represented by a 3-componenet waveforms with an assigned magnitude value. Here is the controversial part comes: instead of sampling waveforms from both datasets randomly we decided to deal with data imbalance right away. 

We have alredy risen the topic regarding the nature of earthquakes' magnitudes distribution, as both datasets contain only real recordings without any oversampling and synthetic data generation, the distribution of magnitudes is heavily skewed towards lower values (M2.0-3.0). As you can see on the histogram of INSTANCE's sample's magnitude distribution on fig. <TODO>.

Because our main goal is to estimate how well can be the relationship between the first several seconds of observations of P-waves and source's magnitude it is not ideal for us to center out focus at these high-density cluster of magnitude values. 

In machine learning, there are different ways to approach data imbalance issue, oversampling and undersampling. 
Oversampling means inflating the number of data instances in the deficit class, while undersampling represents the idea of removing the data instances from the classes with prevailing number of data.

<MAYBE>Oversampling with augmented samples</MAYBE>

Due to the lack of exepertise in the seismic field which could potentially introduce the bias in the final dataset during data synthesis, it was ultimately decided to undersample the other "classes". Finally, out dataset contains: 25,000 earthquake waveforms of magniture in the range M1.0-3.0, 25,000 in the range M3.0-5.0 and ~3,500 data instances (all the observations from both datasets) of M5.0>.


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