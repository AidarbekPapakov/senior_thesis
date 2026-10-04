# Earthquake Early Warning Systems — Notes

## 1. Terminology

### Geophysics related
- EWS / EEW (Early Warning System / Earthquake Early Warning)
- P-waves
- S-waves
- foreshocks
- aftershocks
- mainshock
- hypocenter
- epicenter
- magnitude
- MMI

### ML related
- CNNs, RNNs – main components of the model that are used in ML systems for EWS
- Train & Validation sets
- metrics and their explanation (Accuracy, precision, recall, beta score and etc.)
- Class disbalance and its implications
- Feature space

---

## 2. Several kinds of natural data

### ground-based (High quality data, but sparse)
- seismometers
- ground motion sensors

### space-based (Lots of noise and gaps in data)
- GNSS such as GPS and others

**TODO:** Decide which to use seismic or GNSS data for EWS

---

## 3. Synth data

- [Detailed answer how to generate seismic data (Late can be fed to any LM to generate the suggested method)](https://www.researchgate.net/post/How-to-generate-synthetic-data-for-seismic-channels-in-rocks)
- How it is generated?
- What are its limitations?

---

## 4. Current systems struggle from lack of high-magnitute data

---

## 5. Existing ML systems for EWS

### 5.0 PhaseNet  
https://arxiv.org/pdf/1803.03211

- Neural network that detects and localizes P-waves and S-waves
- basically a more robust and more general version of STA/LTA algorithm

**Quotes**
> Instead of using manually defined features, deep neural networks learn the features  
> from labeled data, both noise and signal, which proves a powerful advantage for complex  
> seismic waveforms

---

### 5.1 M-LARGE  
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB022703

- As stated by the authors, only useful for high-magnitude earthquakes (M7+), so it is not a good candidate for our purposes
- States as a presumption:  
  initial rupture signals (i.e., <5 s) do not provide the information of final magnitude,  
  as opposed to strongly deterministic scenarios (Olson & Allen, 2005; Wu & Zhao, 2006)  
  where the initial rupture signal for small or large magnitude events are fundamentally different.
- Describes in detail data synthesis, and links the synthesized data  
  https://zenodo.org/records/5015610
- Provides code implementation of the system  
  https://zenodo.org/records/4527253

**Quotes**
> Rather M-LARGE requires triggering, ostensibly by a seismic system  
> as is common in other GNSS algorithms (e.g., Crowell et al., 2018).  
> The noise in GNSS data is greater than that in seismic data and many algorithms  
> have been demonstrated for the detection of the onset of events using inertial recordings  
> (Perol et al., 2018; Ross et al., 2018; Zhu & Beroza, 2019) so a system that relies  
> on seismometers for triggering is still the most robust.

---

### 5.2 Real-time Earthquake Early Warning with Deep Learning  
Application to the 2016 Central Apennines, Italy Earthquake Sequence  
https://arxiv.org/pdf/2006.01332

**INVESTIGATE**
> First, we directly extract earthquake source parameters from  
> continuous waveform streams without phase picking, which  
> substantially decreases the response time for early warning.

- magnitude distribution of the training data is rather logarithimic with the peak at M2.5

**Quotes**
> For onsite warning, predominant periods and/or amplitudes of  
> the first few seconds of P waves at single stations are  
> utilized to evaluate source magnitude or ground shaking (5–10s).

> To comprehensively test the performance and robustness  
> of our system, we symmetrically analyze 500 testing  
> samples. To simulate real-time earthquake monitoring, we  
> randomly cut waveforms of the 500 testing samples to form  
> various time windows with different waveform lengths at  
> one or multiple stations. The results show that early warning  
> could be activated as early as 3–4 s after the first P phase if  
> only one or two stations receive earthquake signals. As  
> more data is received, the mean errors of epicentral location,  
> depth, and magnitude decrease from 8.8 km, 2.6 km, and  
> 0.4 to 3.7 km, 1.5 km, and 0.24, respectively.

---

### 5.3 QUITE SIMILAR APPROACH TO WHAT I'M PLANNING TO DO

Explainable deep learning for real-time prediction of uniform hazard spectral acceleration for on-site earthquake early warning  
https://academic.oup.com/gji/article/243/2/ggaf345/8244655?login=false

**Quotes**
> Jozinović et al. (2020, 2021) proposed an automated raw data-driven approach based on  
> a CNN architecture to predict ground motion intensity, including PGA, PGV and Sa  
> at a few periods of 0.3, 1 and 3s, using the initial seconds of arriving ground motions  
> (7, 10 or 15 s-long waveforms)

> Accurate automatic picking of P-wave arrivals is essential.  
> The ground motion recordings in the NGA-West2 data base do not begin with P-wave arrivals  
> and may include additional zeros or early station noise.  
> As a result, accurately picking the P waves in these ground motions is a critical step in developing the proposed DLUHS framework

> The window lengths are defined as starting from the origin of the P-wave arrival and extending 3, 5 and 8 s afterward.  
> This study automatically picks P-wave arrivals using the robust PPHASEPICKER algorithm developed by Kalkan (2016).

**Datasets**
- NGA-West2 (Magnitude range of 3.4 to 7.9)
- NIED K-NET
- KiK-net
- National Research Institute for Earth Science and Disaster Resilience
- USGS Earthquake Hazard Toolbox

---

### 5.4 Expanding the Role of GNSS in Seismic Monitoring  
https://insidegnss.com/expanding-the-role-of-gnss-in-seismic-modeling/

- Not a paper, but an artice
- discusses how GNSS data can be used for seismic monitoring
- Imediately shows that for <M6.0 earthquakes GNSS is almost useless

**Quotes**
> Existing inertial and geodetic networks were largely built and continue to operate independently.  
> Inclusion of both sensor types increases the density of ground motion observations.  
> Such a densification is particularly valuable in relatively sparser regions, such as Alaska,  
> but also adds redundancy and resilience to all existing overlapped networks.

> GNSS ambient position noise is predominantly the aggregate of timing effects of GNSS radio signal propagation through the atmosphere,  
> satellite and receiver oscillators and the antenna radio frequency environment.

---

### 5.5 CRED  
https://arxiv.org/pdf/1810.01965

- Solves the task by constructing a binary vector of 1 representing an earthquake and 0 otherwise as ground truth

**Quotes**
> Inputs into the network are spectrograms of three component seismograms.

**Regarding datasets**
> 550,000 30-second 3-component seismograms recorded by 889 broadband and short-period  
> stations in North California are used for the training of the network and its validation.

> Another half of the data set consists of seismic noise recorded by the same network stations.

> We process one month of continuous data recorded at station WHAR during August 2010.

---

### 5.6 Ai-Powered Based Real Time Earthquake Early Warning System Using Deep Learning

(Local file: Lalithavani2752025JERR136675.pdf)

- Made by Indian authors
- Basically just an overview
- proposes a CNN + MLP model for binary classification

---

### 5.7 Reliable Real-time Seismic Signal/Noise Discrimination with Machine Learning

- Tests 5 different classifiers
- CNN is 100x less fp and inference time is the same

---

### 5.8 A Machine-Learning Approach for Earthquake Magnitude Estimation  
https://arxiv.org/pdf/1911.05975  
https://github.com/smousavi05/MagNet

- same architecture already used
- not the best results honestly
- magnitudes barely above 3.0

---

### 5.9 A NOVEL APPROACH FOR EARTHQUAKE EARLY WARNING SYSTEM DESIGN USING DEEP LEARNING TECHNIQUES  
https://arxiv.org/pdf/2101.06517

- MFCC → CNN → LSTM → MLP
- classification task (P-wave, S-wave, Noise)
- whole waveform is used as input (Why I have no idea honestly)
- deep dive into hardware

---

### 5.10 Seismic Signal Denoising and Decomposition Using Deep Neural Networks  
https://arxiv.org/pdf/1811.02695

- U-net like architecture
- STA/LTA on denoised signals works better

---

### 5.11 THE TRANSFORMER EARTHQUAKE ALERTING MODEL  
https://arxiv.org/pdf/2009.06316

- CNN + transformer
- raw waveform as input
- PGA PDF as output

---

### 5.12 A global-scale database of seismic phases  
https://arxiv.org/pdf/2505.18874

- basically a dataset

---

### 5.13 Generalized Neural Networks for Real-Time EEW  
https://arxiv.org/pdf/2312.15218

- 3-component waveform input
- Fully Convolutional Network
- predicts magnitude PDF

---

## Notes

- Most papers use the whole waveform as input (P + S)
- not sustainable for EEW
- need early estimation
- need to put attention to synthetic data for large earthquakes (M5.0+)
- spectrogram as denoiser and feature extractor
- test on overly saturated data (M7.0+)
- might add confidence intervals by retraining model with different seeds/data splits
and make a pdf oe common metrics (RMSE, MAE, R^2, Pearson's corr.)
---

## 6. Things to mention

- Mel spectrograms vs MFCC  
  https://arxiv.org/pdf/1706.07156
- CNNs, MLPs
- Gradient Descent, Backpropagation, Loss functions
- Dropout
- Metrics
- Class disbalance
- Possible MLE for synthetic data

---

## 7. Data

### CAIAG
Central-Asian Institute for Applied Geosciences

- Open access seismic data in Kyrgyzstan

Links:
- https://www.caiag.kg/ru/nauchnaya-infrastruktura/informatsionnye-sistemy/sdss
- https://www.caiag.kg/ru/nauchnaya-infrastruktura/informatsionnye-sistemy/bazy-sejsmicheskikh-dannykh


### Python package with different utils, one of which is to retrieve data
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://maihao14.github.io/QuakeLabeler/tutorials.html

### Python package with different utils, similar to the above one. Also provides datasets
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://github.com/seisbench/seisbench?tab=readme-ov-file


### STanford EArthquake Dataset (STEAD):A Global Data Set of Seismic Signals for AI
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://github.com/smousavi05/STEAD

### LEN-DB - Local earthquakes detection: a benchmark dataset of 3-component seismograms built on a global scale
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://zenodo.org/records/3648232


### INSTANCE is a dataset of seismic waveforms data and associated metadata suited for analysis based on machine learning
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://www.pi.ingv.it/banche-dati/instance/

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;More about INSTANCE:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"it includes events in the magnitude range between 0.0 and 6.5  "

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"All the waveform traces have a length of 120 s, are sampled at 100 Hz"

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"the manual picking of the arrival phases is routinely performed by a group of about 20 INGV highly trained staff personnel who also review the hypocenter locations and magnitude determination before bulletin publication"

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"This part of our data assembling procedure targets the preparation of the digital counts waveform traces. It includes the following steps:"

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"removal of traces containing data gaps (i.e., missing data);
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;trimming the waveform trace to the nearest sample to the start time;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;120 s trace windowing;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;removal of mean and linear trends from the data;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;resampling at 100 Hz;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;calculation of the signal-to-noise ratio;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;extraction of the data quality metrics."


### SYNTHOSEIS

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Python package to generate seismic data (https://github.com/sede-open/synthoseis)


### Other
- https://www.sciencebase.gov/catalog/item/5aa40985e4b0b1c392eaaeee

---

## 8. Current situation in Kyrgyzstan

- PRESTo is used  
  https://www.prestoews.org/about.php
- Bayesian magnitude estimation

Papers:
- 2008 (FREE): https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2007JB005386
- 2015 (PAID): https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(15)00050-9
- 2021 (FREE): https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020JB020359
