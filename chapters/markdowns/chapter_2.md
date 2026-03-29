# Chapter 2 — Background and Literature Review

## 2.1 Fundamentals of Seismology

Before diving into the proposed method and to have a grasp understanding of what is going to be discussed, we are going to get familiar with several terms related to seismology.

### 2.1.1 Earthquake Generation

Essentially, earthquakes are the result of sudden release of accumulated strain energy inside of the Earth's crust. Tectonic stresses gradually deform rocks along geological faults until the stress exceeds the frictional resistance preventing slip. By that time, rapid rupture propagates along the fault plane, releasing the energy storen in the form of seismic waves.

The location inside the Earth where the rupture itself starts is called the **hypocenter** (or focus), whereas the projection of this point onto the Earth's surface is called the **epicenter**. Depending on the size of the earthquake, the rupture process might propagate along the fault plane over several seconds or even longer.

Earthquakes might consist of several stages. The main event is called the **mainshock**, smaller earthquakes occurring before and after it are known as **foreshocks** and **aftershocks**, respectively.

### 2.1.2 Seismic Wave Types

There are several distinguishable types of waves considered, the primary two are:

**P-waves** (primary waves) and **S-waves** (secondary waves).

P-waves are faster, meaninig they're are first to reach the Earth's surface, despite that, they are not as desctructive and dangerous as the S-waves.
The velocity of P-waves might range approximately from 5 to 8 km/s in the Earth's crust.

On the other hand, S-waves travel notably slower than P-waves, carrying more threat due to their more destructive nature. Their velocity might range from 3 to 4 km/s.

And this difference in velocities lays the ground for most early warning systems deployed around the world. The small window allows for issuing a warning before the destructive waves reach the vulnerable areas.

### 2.1.3 Earthquake Magnitude and Intensity

The scale of an earthquake can be measured in several ways. Historically, the Richter magnitude scale was used to estimate earthquake size based on the logarithm of the maximum amplitude recorded by a seismograph. However, this scale saturates for large earthquakes and is no longer widely used for scientific analysis.

Modern seismology instead relies on the **moment magnitude scale** $M_w$, which is derived from the seismic moment:

$$M_0 = \mu A D$$

where

- $\mu$ is the shear modulus of the rock,
- $A$ is the rupture area, and
- $D$ is the average slip along the fault.

The moment magnitude is then defined as:

$$M_w = \frac{2}{3} \log_{10}(M_0) - 6.07$$

This provides a consistent measure of earthquake scale across a range of magnitudes.

It is also important to distinguish magnitude from **intensity**, which measures how severe ground shaking is at a certain place. Modified Mercalli Intensity (MMI) scale is one of the ways how to measure intensity. It ranges from I (not felt) to XII (complete destruction).

Due to the fact that intensity is highly dependent on the place location, ground soil and stability, early warning systems usually try to estimate earthquake magnitude and location.

## 2.2 Classical Earthquake Early Warning Systems

For the most part of their existance, EEW systems were built on the base of signal processing methods and empirical observed by the experts in the field.

Two common approaches are: 

1) energy-based detection algorithms 
2) probabilistic parameter estimation frameworks.

### 2.2.1 STA/LTA Detection

The one method, which is used widely, but is quite simplistic at the same time, is the Short-Term Average / Long-Term Average (STA/LTA) algorithm.

The method calculates the ratio between two moving averages of signal amplitude. The short-term average reflects recent signal energy, while the long-term average represents the background noise level. The detection statistic is then defined as:

$$R(t) = \frac{\text{STA}(t)}{\text{LTA}(t)}$$

When this ratio exceeds a predefined threshold [set by the experts in the field], the system notifies about the seismic event.

The STA/LTA algorithm is basic, computationally simple and can operate in real time. This is why it has been used in seismic monitoring systems.

However, this algorithm has certain limitations. STA/LTA primarily detects abrupt increases in signal amplitude but provides little to no information about the physical properties of the earthquake, such as its magnitude or rupture dynamics. It is also noteworthy that the approach is sensitive to noise and may produce false positive results in environments with strong background vibrations.

### 2.2.2 Bayesian EEW Systems

Another paradigm in EEW is utilizing statistical-probabilistic models to estimate earthquake parameters from observations recorded at multiple stations.

One example of such a system is the **PRESTo** (PRobabilistic and Evolutionary early warning SysTem) framework, which is deployed on the terriotory of Kyrgyzstan. <TODO: dive deeper into bayesian method> PRESTo combines Bayesian inference with empirical ground motion prediction equations to estimate earthquake magnitude and location as new observations become available. <TODO>

In such systems, the posterior distribution of earthquake properties is updated as soon as other stations detect seismic signals. This approach allows to converge to a reasonable estimation of parameters' distributions as more data comes.

While this system improves estimation accuracy compared to simple signal processing algorithms, they still rely on manually tuned features and require substantial amount of time to gather data to approach near-optimal estimation. Consequently, it may struggle to output reliable magnitude estimates using only the earliest observations of seismic data.

## 2.3 Data Sources for Seismic Machine Learning

Machine learning, notoriosly, requires much data to be able to produce adequate and meaningful results. EEW is no exception; large volumes of labeled seismic data are needed to train a good machine learning model. These data is typically obtained from two types of sources: ground-based seismic sensors and satellite-based geodetic systems.

### 2.3.1 Ground-Based Seismic Sensors

The most common seismic monitoring instruments are seismometers and accelerometers installed at ground stations. These instruments measure ground motion along three orthogonal axes, producing three-component waveform recordings. In this work, ground-based sensors will be our main source of data. The reason behind that is the fact that ground-based seismic sensors provide high-quality data. They are capable of detecting subtle ground motions associated with small earthquakes, which are take up the largest portion of all earthquakes.

### 2.3.2 GNSS-Based Observations

While not directly used in this work, Global Navigation Satellite Systems (GNSS) still deserve to be mentioned. GNSS receivers measure ground displacement directly by tracking changes in satellite signal timing.

In fact, GNSS measurements can complement seismic data by providing accurate estimates of large ground displacements associated with major earthquakes, that ground-based methods mighst struggle due to magnitude oversaturation.

However, GNSS measurements are significantly noisier than seismic recordings. As a result, GNSS-based monitoring is effective only for very large earthquakes. Studies have shown that the systems that employ only GNSS suffer from inability to detect smaller amd medium eqrthquakes (M6.0<) as the signals received often contain insufficient information to do so.

Consequently, most EEW systems rely primarily on seismic sensors for early detection and parameter estimation, while GNSS measurements may serve as complementary data sources for very large events.

## 2.4 Machine Learning Approaches to Seismic Monitoring

With the exponential development of deep learning over the past decade, researchers and scientists started exploring the potential applications of machine learning to seismic signal analysis. Neural networks, in particular, have demonstrated strong performance in tasks such as seismic phase picking, earthquake detection, magnitude estimation, and ground motion prediction.

Several representative systems are reviewed below.

### 2.4.1 Machine Learning advances and achievements

Several studies have been moving towards this direction, but the one to stand out among the others is the one that introduced **PhaseNet**, a deep neural network designed for automatic detection of P-wave and S-wave arrivals. The architecture is based on a convolutional encoder–decoder structure similar to the U-Net architecture widely used in image segmentation.

Instead of relying on manually designed signal features, PhaseNet processes raw seismic waveforms and learns representations directly from labeled training data. The network produces probability estimates for P-wave and S-wave arrivals at each time step.

PhaseNet has shown to outperform traditional STA/LTA-based methods in both detection accuracy and robustness to noise. 

PhaseNet represents a common trend in seismology: replacing manually engineered signal processing pipelines with data-driven models capable of learning complex patterns in seismic signals.

Another similar work, is the **CRED** system, which represents another deep learning approach for earthquake detection. In contrast to models operating directly on waveforms, CRED uses spectrograms of three-component seismic observations.

The network itself is composed of CNNs and RNNs, from the premise of spectrograms being a verbose spatial-temporal representaion of seismic waves. The system is then trained to solve the problem of binary classification that outputs a vector of binary values, with 1 representing an earthquake and 0 the absence of earthquake. Training is performed on a large dataset of labeled seismic events and noise recordings.

This particular work shares a lot with what we have done, but the task solved is essentially different from what we have set to accomplish because the task of binary classification is significantly easier than estimating a scalar value.

**M-LARGE** is one the systems that utilizes the data received from GNSS sensors that we have discussed before, thus, concluded by the authors themselves, is practically useless to the low to meduium eartquakes, that are the biggest of all earthquakes.

Nevertheless, the proposed model performed exceptionnally well on synthetically generated high-magnitude data, which still indicate the fact that deep learning models might excell at capturing complex patterns in seismic data.

Another direction of research focuses on directly earthquake parameters based on raw waveform data without explicitly preprocessing the signals.

Multiple approaches have been utilized in that regard. Some researchers concatenated three-component waveforms into a one matrix and passed it to CNNs, other used a single waveform as the input to RNN-based architecture.

Such studies try to make these models learn the relevant features from the raw data itself.

Another advance, motiavted by the architectures that dominated in the tasks of Computer Vision and Natural Language processing, is attention-based architectures.
The Transformer Earthquake Alerting Model (TEAM) combines convolutional layers with transformer-based attention mechanisms to estimate probability distributions of peak ground acceleration (PGA). Transformers allow the model to capture long-range temporal dependencies within seismic waveforms. Although these models demonstrate promising performance, their computational complexity can pose challenges for real-time deployment in EEW systems.

Several models have been proposed specifically for earthquake magnitude estimation. For example, the **MagNet** estimates earthquake magnitude from seismic waveforms. While the model demonstrates reasonable performance for small earthquakes (0.5-2.0M), its accuracy tends to decrease significantly for larger events due to limited training data skewed to lower bounds of earthquake's magnitude.

Another way deep learning has found its place in is signal denoising. Weiqiang Zhu and  S. Mostafa Mousavi adapted the U-net like architecture to build a model that successfully manages to remove the noise from the incoming signals. The extrincic metric of their was the increased performance of STA/LTA method applied to denosed signals over the original waveform. 

## 2.5 Limitations of Existing Approaches

Despite significant progress in machine learning based seismic monitoring, certain limitations persist.

**Dependence on Full Waveforms** - A significant amount of studies utilize the full waveform, including both P-waves and S-waves. While this, obviously, improves models' metrics, it, subsequently, reduces the relevance of such systems for real-time early warning, where only the earliest seconds of data are available.

**Data Imbalance** - Earthquake magnitudes follow the Gutenberg–Richter law, which states that the number of earthquakes decreases exponentially with increasing magnitude. As a result, large earthquakes are extremely rare compared to small ones. This imbalance leads to training datasets heavily dominated by low-magnitude events, making it difficult for machine learning models to learn reliable representations of large earthquakes. 

Although, some of the mentioned works have tried utilizing synthetically generated waveforms based on the groundworks of seismologists, who made it possible to approximately emulate the signals produced by earthquakes.

**Underutilization of Signal Processing** - Most proposed architectures rely solely on raw inputs instead of utilizing widely known signal processing transformations, such as Mel Frequency Cepstral Coefficients (MFCCs) and Spectograms.

## 2.6 Research Gap

Despite illustrating the broad appication of machine learning techniques in seismology, the literature reviewed suggests several opportunities for further research.

First, relatively few studies focus explicitly on estimating earthquake magnitude using only the earliest seconds of P-wave data. Most approaches rely on longer waveform segments that might include S-wave observations.

Second, many architectures either miss out on spatial representations or temporal modeling or feature engineering using one, two and almost never all three techniques at the same time, and those systems utilizing all three usually perfrom a much simplier task of classifying events, rather than estimating earthquakes parameters.

Finally, the scarcity of high-magnitude training data remains a major challenge. Techniques such as synthetic data generation and targeted sampling strategies might help mitigate this limitation.

## 2.7 Summary

This chapter reviewed the physical principles underlying earthquake generation and seismic wave propagation, as well as the traditional methods used in earthquake early warning systems.

Classical EEW algorithms rely on signal processing techniques and manually tuned parameters that often require high expertise in the field. These methods struggle to estimate earthquake magnitude accurately within the first few seconds of an event.

Recent advances in deep learning have introduced new approaches for seismic signal analysis, including phase detection, signal denoising, event classification, and ground motion prediction. These models show the potential of machine learning based methods.

However, several challenges remain, including dataset imbalance, reliance on full waveform data, and limited exploration of architectures combining spatial and temporal feature extraction.

These limitations motivate the methodology proposed in this thesis, which integrates spectrograms, convolutional neural networks and recurrent neural networks to estimate earthquake magnitude from early P-wave spectrograms derived from multi-source seismic datasets.