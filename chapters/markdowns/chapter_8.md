# Chapter 8 — Conclusion

## 8.1 Summary of the Work

This thesis addressed the problem of earthquake magnitude estimation in the context of earthquake early warning systems. The main objective was to develop a deep learning model capable of predicting earthquake magnitude using only the first few seconds of seismic waveform data after P-wave arrival.

To achieve this, the following steps were performed:

1. A review of existing machine learning approaches for earthquake early warning was conducted.
2. Mathematical foundations of signal processing and deep learning were established.
3. A preprocessing pipeline was developed to convert seismic waveforms into spectrogram representations.
4. A CNN–LSTM–MLP architecture was proposed for magnitude estimation.
5. An experimental framework was designed and used to evaluate model performance across multiple observation windows.

## 8.2 Key Findings

Based on the experimental results, the following conclusions can be drawn.

1. **Early seismic signals contain meaningful information about earthquake magnitude.** Using only the first 3 seconds of P-wave data, the proposed model achieves MAE = 0.3699 and $R^2$ = 0.6483. With 10 seconds of data, performance improves to MAE = 0.3152 and $R^2$ = 0.7478.
2. **Convolutional and recurrent networks model spectrogram dynamics well.** The combination is capable of modeling complex spatio-temporal data such as earthquake waveform spectrograms. Spearman's rank correlation reaches 0.84 for the 10-second window, indicating that the model preserves the ordering of magnitudes well across the test set.
3. **Longer windows help, with diminishing returns.** Most of the magnitude-relevant information accessible to a single-station model is present in the first few seconds, suggesting a favorable trade-off between accuracy and warning latency.
4. **The learned embeddings encode magnitude.** PCA projection of the learned embeddings reveals a clear magnitude gradient along the first principal component, indicating that the CNN–LSTM backbone learns representations in which magnitude is one of the dominant axes of variation.

## 8.3 Contributions

The main contributions of this work are:

- the development of a spectrogram-based preprocessing pipeline for seismic data;
- the design of a hybrid CNN–LSTM–MLP architecture for early magnitude estimation;
- the integration of the INSTANCE seismic dataset into a unified training framework;
- the formulation of a structured experimental methodology for evaluating EEW models across multiple observation windows;
- an empirical demonstration that meaningful magnitude estimates can be obtained from as little as 3 seconds of single-station P-wave data.

## 8.4 Future Work

Several directions for future research can be identified.

- **Architecture:** transformer-based models, more sophisticated attention mechanisms for temporal modeling, and lightweight architectures for faster inference.
- **Data:** additional regional datasets, and data enrichment via synthetic data generation for larger earthquakes, with appropriate modeling to avoid data contamination.
- **Software:** model quantization to improve inference speed on microcontrollers that lack a dedicated graphics processing unit and can perform only integer-based arithmetic.
- **Methodology:** comparing the proposed model against a multi-station baseline and against established prior approaches (such as MagNet and TEAM) on a common evaluation protocol, which would strengthen the empirical case for spectrogram-based architectures in EEW.

## 8.5 Final Remarks

Earthquake early warning remains a challenging problem due to the inherent uncertainty and variability of seismic processes. While deep learning approaches cannot solve this problem outright, there is significant unexplored territory at the intersection of deep learning and seismology. The results presented in this thesis suggest that spectrogram-based hybrid architectures represent a productive line of investigation for this field.
