# Chapter 2 — Background and Literature Review

## 2.1 Fundamentals of Seismology

Understanding earthquake early warning systems requires a brief overview of the physical processes underlying seismic events and the signals recorded by monitoring stations.

### 2.1.1 Earthquake Generation

Earthquakes occur due to the sudden release of accumulated elastic strain energy within the Earth's crust. Tectonic stresses gradually deform rocks along geological faults until the stress exceeds the frictional resistance preventing slip. At that point, rapid rupture propagates along the fault plane, releasing stored energy in the form of seismic waves.

The point inside the Earth where rupture initiates is called the **hypocenter** (or focus), while the projection of this point onto the Earth's surface is called the **epicenter**. The rupture process typically propagates along the fault plane over several seconds or longer, depending on the size of the earthquake.

Earthquake sequences often consist of several stages. The main rupture event is referred to as the **mainshock**, while smaller earthquakes occurring before and after it are known as **foreshocks** and **aftershocks**, respectively. These events reflect the redistribution of stress within the surrounding crust.

### 2.1.2 Seismic Wave Types

When an earthquake occurs, several types of seismic waves propagate through the Earth's interior and along its surface.

The two primary body waves are **P-waves** (primary waves) and **S-waves** (secondary waves).

P-waves are compressional waves that propagate by alternately compressing and expanding the material through which they travel. Because they involve particle motion parallel to the direction of propagation, P-waves can travel through both solid and fluid materials. Their velocities typically range from approximately 5 to 8 km/s in the Earth's crust.

S-waves, in contrast, are shear waves that involve particle motion perpendicular to the direction of propagation. These waves cannot propagate through fluids and travel more slowly than P-waves, typically at velocities of 3 to 4 km/s.

This difference in velocity forms the physical basis of earthquake early warning systems. Since P-waves arrive before the more destructive S-waves, the detection of the first P-wave arrivals provides a small time window in which warnings can be issued before strong shaking begins.

In addition to body waves, earthquakes also generate **surface waves**, which travel along the Earth's surface and often produce the largest ground displacements responsible for structural damage.

### 2.1.3 Earthquake Magnitude and Intensity

The size of an earthquake is quantified using several magnitude scales. Historically, the Richter magnitude scale was used to estimate earthquake size based on the logarithm of the maximum amplitude recorded by a seismograph. However, this scale saturates for large earthquakes and is no longer widely used for scientific analysis.

Modern seismology instead relies on the **moment magnitude scale** $M_w$, which is derived from the seismic moment:

$$M_0 = \mu A D$$

where

- $\mu$ is the shear modulus of the rock,
- $A$ is the rupture area, and
- $D$ is the average slip along the fault.

The moment magnitude is then defined as:

$$M_w = \frac{2}{3} \log_{10}(M_0) - 6.07$$

This scale provides a consistent measure of earthquake size across a wide range of magnitudes.

It is important to distinguish magnitude from **intensity**, which measures the severity of ground shaking at a particular location. One commonly used intensity scale is the Modified Mercalli Intensity (MMI) scale, which ranges from I (not felt) to XII (complete destruction). Intensity varies with distance from the epicenter and local geological conditions.

Earthquake early warning systems typically attempt to estimate earthquake magnitude and location in order to predict expected ground motion intensity at various locations.

## 2.2 Classical Earthquake Early Warning Systems

Before the introduction of machine learning approaches, EEW systems relied primarily on deterministic signal processing methods and empirical relationships derived from seismological observations.

Two widely used approaches include energy-based detection algorithms and probabilistic parameter estimation frameworks.

### 2.2.1 STA/LTA Detection

One of the most widely used earthquake detection techniques is the Short-Term Average / Long-Term Average (STA/LTA) algorithm.

The STA/LTA method monitors the ratio between two moving averages of signal amplitude. The short-term average reflects recent signal energy, while the long-term average represents the background noise level. The detection statistic is defined as:

$$R(t) = \frac{\text{STA}(t)}{\text{LTA}(t)}$$

When this ratio exceeds a predefined threshold, the system identifies a possible seismic event.

The STA/LTA algorithm is computationally simple and can operate in real time. For this reason, it remains widely used in operational seismic monitoring systems.

However, the algorithm has important limitations. STA/LTA primarily detects abrupt increases in signal amplitude but provides little information about the physical characteristics of the earthquake, such as its magnitude or rupture dynamics. Furthermore, it is sensitive to noise and may produce false detections in environments with strong background vibrations.

### 2.2.2 Bayesian EEW Systems

More advanced EEW systems incorporate probabilistic models to estimate earthquake parameters from observations recorded at multiple stations.

One example is the **PRESTo** (PRobabilistic and Evolutionary early warning SysTem) framework. PRESTo combines Bayesian inference with empirical ground motion prediction equations to estimate earthquake magnitude and location as new observations become available.

In Bayesian EEW systems, the posterior distribution of earthquake parameters is updated as additional stations detect seismic signals. This approach allows uncertainty to be explicitly modeled and progressively reduced as more data becomes available.

While probabilistic systems improve estimation accuracy compared to simple signal processing algorithms, they still rely on manually engineered features extracted from the waveform. These features may include peak ground displacement, dominant period, or signal amplitude ratios.

A key limitation of such features is that they often require observing a substantial portion of the waveform, including both P-waves and S-waves. Consequently, classical systems may struggle to produce reliable magnitude estimates using only the earliest seconds of seismic data.

## 2.3 Data Sources for Seismic Machine Learning

Machine learning approaches to EEW rely on large volumes of labeled seismic data. These data are typically obtained from two types of sensor networks: ground-based seismic sensors and satellite-based geodetic systems.

### 2.3.1 Ground-Based Seismic Sensors

The most common seismic monitoring instruments are seismometers and accelerometers installed at ground stations. These instruments measure ground motion along three orthogonal axes, producing three-component waveform recordings.

Ground-based seismic sensors provide high-quality data with relatively high signal-to-noise ratios. They are capable of detecting subtle ground motions associated with small earthquakes.

However, seismic stations are relatively expensive to install and maintain. As a result, global seismic networks are spatially sparse compared to satellite-based observation systems.

Despite this limitation, ground-based sensors remain the primary data source for most EEW systems due to their high sensitivity and temporal resolution.

### 2.3.2 GNSS-Based Observations

An alternative source of geophysical data is provided by Global Navigation Satellite Systems (GNSS) such as GPS. GNSS receivers can measure ground displacement directly by tracking changes in satellite signal timing.

In principle, GNSS measurements can complement seismic data by providing accurate estimates of large ground displacements associated with major earthquakes.

However, GNSS measurements are significantly noisier than seismic recordings. Ambient GNSS position noise arises from several factors, including atmospheric signal propagation delays, receiver clock errors, and environmental interference.

As a result, GNSS-based monitoring is generally effective only for very large earthquakes. Studies have shown that GNSS signals often contain insufficient information to detect earthquakes below approximately magnitude 6.0. For smaller events, the displacement signals are typically indistinguishable from background noise.

Consequently, most EEW systems rely primarily on seismic sensors for early detection and parameter estimation, while GNSS measurements may serve as complementary data sources for very large events.

## 2.4 Machine Learning Approaches to Seismic Monitoring

Over the past decade, machine learning techniques—particularly deep learning—have been increasingly applied to seismic signal analysis. Neural networks have demonstrated strong performance in tasks such as seismic phase picking, earthquake detection, magnitude estimation, and ground motion prediction.

Several representative systems are reviewed below.

### 2.4.1 PhaseNet

PhaseNet is a deep neural network designed for automatic detection of P-wave and S-wave arrivals. The architecture is based on a convolutional encoder–decoder structure similar to the U-Net architecture widely used in image segmentation.

Instead of relying on manually designed signal features, PhaseNet processes raw seismic waveforms and learns representations directly from labeled training data. The network produces probability estimates for P-wave and S-wave arrivals at each time step.

This approach has been shown to outperform traditional STA/LTA-based methods in both detection accuracy and robustness to noise.

PhaseNet illustrates a broader trend in seismology: replacing manually engineered signal processing pipelines with data-driven models capable of learning complex patterns in seismic signals.

### 2.4.2 CRED: Convolutional Recurrent Earthquake Detector

The CRED system represents another deep learning approach for earthquake detection. In contrast to models operating directly on waveforms, CRED uses spectrogram representations derived from three-component seismic recordings.

The network architecture combines convolutional layers with recurrent layers, enabling it to capture both spectral patterns and temporal dependencies in the data.

The system is trained as a binary classifier that outputs a vector indicating the probability that an earthquake signal is present at each time step. Training is performed on a large dataset of labeled seismic events and noise recordings.

By operating on spectrogram inputs, CRED effectively converts seismic waveform analysis into a pattern recognition problem similar to image processing.

### 2.4.3 Direct Parameter Estimation from Waveforms

Another line of research focuses on directly estimating earthquake source parameters from continuous waveform streams without explicit phase picking.

For example, deep learning models have been developed to infer earthquake location and magnitude directly from raw waveform segments. These models attempt to bypass traditional signal processing steps by allowing neural networks to learn the relevant features automatically.

Such approaches have demonstrated the ability to produce early estimates of earthquake parameters within a few seconds after P-wave detection. As additional waveform data becomes available, the accuracy of these estimates improves.

This paradigm represents a shift from modular processing pipelines toward end-to-end learning systems.

### 2.4.4 Transformer-Based EEW Models

More recently, attention-based architectures have been explored for seismic monitoring tasks.

The Transformer Earthquake Alerting Model (TEAM) combines convolutional layers with transformer-based attention mechanisms to estimate probability distributions of peak ground acceleration (PGA). Transformers allow the model to capture long-range temporal dependencies within seismic waveforms.

Although these models demonstrate promising performance, their computational complexity can pose challenges for real-time deployment in EEW systems.

### 2.4.5 Magnitude Estimation Networks

Several neural network models have been proposed specifically for earthquake magnitude estimation.

For example, the MagNet architecture uses deep learning to estimate earthquake magnitude from seismic waveforms. While the model demonstrates reasonable performance for small earthquakes, its accuracy tends to degrade for larger events due to limited training data at high magnitudes.

More recent work has explored fully convolutional architectures that predict probability distributions over possible magnitudes rather than a single scalar estimate.

These approaches highlight the ongoing challenge of magnitude estimation in imbalanced datasets dominated by small events.

## 2.5 CNN-Based Ground Motion Prediction from Early Waveforms

One of the approaches most closely related to the methodology proposed in this thesis is the deep learning framework introduced by Jozinović and colleagues.

Their method predicts ground motion intensity metrics—such as peak ground acceleration (PGA) and peak ground velocity (PGV)—using only the initial seconds of seismic waveforms following P-wave arrival.

The model uses convolutional neural networks to process short waveform segments ranging from approximately 7 to 15 seconds in duration. By analyzing these early signals, the network attempts to predict the eventual ground motion intensity that will occur later in the event.

This approach demonstrates that deep learning models can extract useful predictive information from the earliest portions of seismic signals.

However, the primary output of this system is ground motion intensity rather than earthquake magnitude. In contrast, the present thesis focuses specifically on estimating the final magnitude of the earthquake using early waveform data.

## 2.6 Limitations of Existing Approaches

Despite significant progress in machine-learning-based seismic monitoring, several limitations remain.

**Dependence on Full Waveforms** — Many existing systems analyze the complete waveform, including both P-waves and S-waves. While this improves accuracy, it reduces the usefulness of the system for real-time early warning, where only the earliest seconds of data are available.

**Data Imbalance** — Earthquake magnitudes follow the Gutenberg–Richter law, which states that the number of earthquakes decreases exponentially with increasing magnitude. As a result, large earthquakes are extremely rare compared to small ones. This imbalance leads to training datasets heavily dominated by low-magnitude events, making it difficult for machine learning models to learn reliable representations of large earthquakes.

**Limited Dataset Integration** — Many studies train models on a single dataset from a specific region. However, seismic waveforms vary significantly across geological environments. Combining multiple datasets may improve model generalization but introduces challenges related to data normalization and distribution shifts.

**Underutilization of Temporal Modeling** — Some existing architectures rely solely on convolutional networks operating on spectrograms or waveform segments. While CNNs are effective at capturing spatial patterns in time–frequency representations, they may not fully capture the temporal evolution of seismic signals. Incorporating recurrent architectures such as LSTMs provides a natural mechanism for modeling sequential dependencies in waveform data.

## 2.7 Research Gap

The literature reviewed above suggests several opportunities for further research.

First, relatively few studies focus explicitly on estimating earthquake magnitude using only the earliest seconds of P-wave data. Most approaches rely on longer waveform segments that include S-wave information.

Second, many architectures either focus on spatial feature extraction using convolutional networks or temporal modeling using recurrent networks, but fewer approaches combine both mechanisms in a unified architecture optimized for magnitude estimation.

Third, the integration of multiple seismic datasets spanning different geographic regions remains underexplored. Training models on combined global datasets and then fine-tuning them for regional deployment may improve performance in practical EEW applications.

Finally, the scarcity of high-magnitude training data remains a major challenge. Techniques such as synthetic data generation and targeted sampling strategies may help mitigate this limitation.

## 2.8 Summary

This chapter reviewed the physical principles underlying earthquake generation and seismic wave propagation, as well as the traditional methods used in earthquake early warning systems.

Classical EEW algorithms rely on signal processing techniques and empirical models that often require observing a substantial portion of the seismic waveform. These methods struggle to estimate earthquake magnitude accurately within the first few seconds of an event.

Recent advances in deep learning have introduced new approaches for seismic signal analysis, including neural networks for phase detection, event classification, and ground motion prediction. These models demonstrate the potential of data-driven methods to extract complex patterns from seismic waveforms.

However, several challenges remain, including dataset imbalance, reliance on full waveform data, and limited exploration of architectures combining spatial and temporal feature extraction.

These limitations motivate the methodology proposed in this thesis, which integrates convolutional neural networks and recurrent neural networks to estimate earthquake magnitude from early P-wave spectrograms derived from multi-source seismic datasets.