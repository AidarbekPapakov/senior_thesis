# Chapter 4 — Dataset and Preprocessing

## 4.1 Overview

Although deep learning has shown to excel at processing complex and high-structure data, it still requires large amounts of data in order to perform at a decent level. In the context of our study, we postulate that the training dataset must contain both the 3-component waveform recordings and the associated ground truth magnitude value for each event. To test our proposed approach the INSTANCE dataset was used.

## 4.2 Data Sources

### 4.2.1 The INSTANCE Dataset

The INSTANCE dataset is, as stated by the authors, intended for machine learning experiments. The dataset developed by the Italian National Institute of Geophysics and Volcanology (INGV).

The datasets contains ~1.2 million waveform recordings, each with the following properties:

- three-component recordings (vertical, north–south, east–west),
- waveform duration of 120 seconds,
- sampling rate of 100 Hz,
- magnitude range approximately 0.0 to 6.5.
- the type of phase picking: manual and automated.

As it is stated by the authors, the dataset underwent several preprocessing steps: 

1) removal of traces containing data gaps
2) trimming of waveform segments to consistent start times
3) resampling to a uniform sampling frequency 
4) removal of mean offsets and linear trends

*[Figure 4.2: Example waveform segment from the INSTANCE dataset]*


### 4.2.2 Dataset Filtration

We applied several filters to the original dataset to meet our criterias. First of all, each observation contains the units of measurements, consequently, only those observations measure in m/s were left. Secondly, although, it was mentioned that current machine learning approaches perform extremely well with the task of phase picking it was still decided to filter out the data with automatically picked phases because of the potential bias that could be introduced due to the lack of verbosity on the automatic part of phase picking. It is in general a good practice, for any machine learning problem, to aim to use the data which labels are fact-checked by the human experts even if the initial annotaion was not made by a human.

Moving to the dataset itself, the authors of INSTANCE dataset provide a sample of the whole dataset that contains only 10,000 observations, which after our filtering is left with ~8,000 data instances. But as we have discussed, this sample is heavily skewed towards lower bound of magnitude, which is not a bad thing, but in the context of our study does not suits us the best. Nevertheless, the proposed model was still tested on this sample. 

The following part might seem counter-intuitive at first, but we explain the choices we have made. To test out approach, we sampled the data in the following proportions: 

50,000 of earthguakes of magnitude in the range M1.0-3.0
50,000 of earthguakes of magnitude in the range M3.0-5.0
All the available earthquakes of magnitude >M5.0 

Instead of performing a random sampling on the original data, we have decided to shift our focus towards more moderate earthquakes by undersampling the classes of low magnitude earthquakes, because our main goal is to estimate how well can be the relationship between the first several seconds of observations of P-waves and source's magnitude it is not ideal for us to work only with highly dense cluster of low magnitude earthquakes. 

In machine learning, there is an alternative approach to data imbalancem called oversampling. Oversampling encapsulates the idea of inflating the number of data instances in the deficit class. Unfortunately, due to lack of expertise in synthesis of seismic data, we have decided to omit the idea of oversampling to exclude the risks of bias in our dataset. 

## 4.3 Preprocessing Pipeline

Although, the authors state the processes under which the waveform data undergo, the job is not finished for neural networks. The following steps transform the waveform into a standardized representation suitable for training our model.

### 4.3.1 P-wave Detection and Window Extraction

EEW systems rely on the initial P-wave arrival. Once the P-wave arrival time $t_P$ is known, a short time window following the arrival is processed.

For a waveform $x(t)$, the window to be processed is defined as follows:

$$x(t), \quad t \in [t_P,\ t_P + W]$$

where $W$ is the window length. In this study, several window lengths are considered:

$$W \in \{3, 5, 8, 10\} \text{ seconds}$$

Several time windows are selected to see the dependency of the learnt relation on the amount of data provided.

*[Figure 4.4: Example waveform with P-wave arrival and extracted analysis window]*

### 4.3.2 Signal Normalization

To have our waveforms suitalbe for a machine learning model, we normalize the incoming waveforms before passing them though STFT.
To do so, we apply the following normalization:

$$\hat{x}(t) = \frac{\tilde{x}(t) - \mu_{\tilde{x}}}{\max_{t}\left|\tilde{x}(t) - \mu_{\tilde{x}}\right| + \varepsilon}$$

Where $\tilde{x}(t)$ is the linearly detrended signal, $\mu_{\tilde{x}}$ is its channel-wise mean, and $\varepsilon = 10^{-8}$.

### 4.3.3 Spectrogram Computation

After window extraction and normalization, the waveform is converted into a time–frequency representation using the Short-Time Fourier Transform described in Chapter 3.

Let the STFT parameters be: FFT size $N_{\text{fft}}$, hop length $H$, and window function $w(n)$. The resulting STFT coefficients are used to compute the log-power spectrogram:

$$S[m, k] = \log\!\left(1 + |X[m, k]|^2\right)$$

For three-component seismic signals, the spectrogram is retrieved for each channel separately.

### 4.3.4 Input Tensor Construction

The spectrograms from the three seismic channels are stacked to form a three-dimensional tensor:

$$\mathbf{X} \in \mathbb{R}^{3 \times F \times T}$$

where $3$ is the number of seismic channels, $F$ is the number of frequency bins, and $T$ is the number of time frames. This tensor eventually is passed to the CNN as the input.

*[Figure 4.5: Example spectrogram representation of a seismic signal]*

## 4.4 Dataset Splitting

To evaluate model performance reliably, the dataset must be divided into separate subsets used for training, validation, and testing. The dataset is partitioned as follows:

- **Training set:** 70% of the data
- **Validation set:** 20% of the data
- **Test set:** 10% of the data

Each set serves its own function: 

1. Training set provides the data, that model updates its weights on via gradient descent.
2. Validation set serves the function of sanity checks during the training, needed to see how our model learns to generalize on the unseen data. During inference on validation set, not backpropogation is performed and weights are not updated.
3. Test set is the ultimate final test for a model to see what it has learnt over the epochs it has been trained on.

The split is performed using stratified sampling based on magnitude bins. This ensures that earthquakes of different magnitudes are represented proportionally in each subset.

## 4.5 Summary

This chapter described the data and preprocessing steps usedin the proposed earthquake magnitude estimation system.

The training dataset provides a sufficient amount of data for a stable model training. The preprocessing pipeline converts raw seismic waveforms into spectrograms suitable for out model input. Key steps involve window clipping, normalization, and time–frequency transformation.

The next chapter introduces the architecture of the proposed deep learning model and describes how these preprocessed inputs are used to estimate earthquake magnitude in near real time.