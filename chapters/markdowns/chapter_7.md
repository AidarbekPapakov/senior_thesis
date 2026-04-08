# Chapter 7 — Discussion

## 7.1 Overview

This chapter interprets the results obtained from the experiments described in Chapter 6 and evaluates the effectiveness of the proposed CNN–LSTM model for earthquake magnitude estimation.

The discussion focuses on three main aspects: model performance and predictive capability, impact of architectural and data-related choices, and limitations of the proposed approach.

Where numerical results are required, placeholders are used and should be replaced once the experiments are completed.

## 7.2 Model Performance Analysis

### 7.2.1 Overall Performance

The overall performance of the model is evaluated using the metrics defined in Chapter 3: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), coefficient of determination ($R^2$), and Pearson correlation coefficient.

Based on the experimental results (see Chapter 6), the model achieved:

| Metric | Value |
|--------|-------|
| RMSE | [value] |
| MAE | [value] |
| $R^2$ | [value] |
| Pearson $r$ | [value] |

These results indicate that the model is capable of [insert interpretation: e.g., "accurately capturing the relationship between early waveform features and earthquake magnitude"].

If the correlation coefficient is high (e.g., $r > 0.8$), this suggests that the model successfully learns a strong linear relationship between predicted and true magnitudes.

### 7.2.2 Error Characteristics

The distribution of prediction errors provides additional insight into model behavior.

From the error histogram (Figure [X]), the model tends to [underestimate / overestimate] magnitudes in the range [range], and the variance of errors increases for [larger / smaller] magnitudes. This behavior is consistent with the imbalance in the dataset, where larger earthquakes are underrepresented.

### 7.2.3 Performance on High-Magnitude Events

A key requirement of earthquake early warning systems is reliable performance on large earthquakes.

For events with magnitude $M \geq 5.0$, the model achieved:

| Metric | Value |
|--------|-------|
| RMSE | [value] |
| MAE | [value] |

If performance degrades in this regime, it may indicate that the training dataset lacks sufficient high-magnitude examples, or that the model struggles to extrapolate beyond the dominant magnitude range. This limitation is expected due to the Gutenberg–Richter distribution discussed earlier.

## 7.3 Impact of Model Architecture

### 7.3.1 CNN–LSTM vs CNN-only

The ablation study comparing CNN-only and CNN–LSTM architectures shows that:

| Architecture | RMSE |
|--------------|------|
| CNN-only | [value] |
| CNN–LSTM | [value] |

If the CNN–LSTM model performs better, this confirms that temporal modeling improves magnitude estimation. This result suggests that the temporal evolution of seismic signals contains important information that cannot be captured by purely spatial feature extraction.

### 7.3.2 Effect of Window Length

The experiment on different observation window lengths shows the following trend:

| Window | RMSE |
|--------|------|
| 3 s | [value] |
| 5 s | [value] |
| 8 s | [value] |

In general, increasing the window length leads to [improved / marginally improved / unchanged] performance. This is expected because longer windows contain more information about the rupture process. However, this improvement comes at the cost of increased warning latency.

Thus, there exists a trade-off between prediction accuracy and response time, which is critical for practical EEW systems.

## 7.4 Impact of Dataset Choice

### 7.4.1 Multi-Dataset Training

The comparison between single-dataset and multi-dataset training shows that:

| Training Data | RMSE |
|---------------|------|
| STEAD only | [value] |
| STEAD + INSTANCE | [value] |

If performance improves when combining datasets, this indicates that the model benefits from increased data diversity and that learned representations generalize better across different seismic conditions.

### 7.4.2 Regional Fine-Tuning

Fine-tuning on CAIAG data results in:

| Stage | RMSE |
|-------|------|
| Before fine-tuning | [value] |
| After fine-tuning | [value] |

Improvement in performance suggests that regional adaptation is important due to differences in geological structure, sensor characteristics, and noise conditions. If improvement is minimal, it may indicate that the global model already generalizes well.

## 7.5 Limitations of the Approach

Despite promising results, the proposed method has several limitations.

### 7.5.1 Data Imbalance

The most significant limitation is the imbalance in earthquake magnitudes. Because large earthquakes are rare, the model is trained primarily on small events. This can lead to underestimation of large magnitudes and reduced reliability for critical events.

### 7.5.2 Dependence on P-wave Picking

The model relies on accurate detection of P-wave arrival times. Errors in P-wave picking can result in incorrect window extraction and degraded model performance. In real-world systems, this dependency introduces an additional source of uncertainty.

### 7.5.3 Limited Early Information

The model operates on very short time windows (3–8 seconds), which inherently limits the available information. As noted in previous studies, early rupture signals may not fully determine the final earthquake magnitude. This creates a fundamental limitation independent of model architecture.

### 7.5.4 Synthetic Data Limitations

If synthetic data is used to augment large-magnitude events, discrepancies between simulated and real signals may introduce bias.

## 7.6 Comparison with Existing Approaches

Compared to traditional EEW methods based on empirical relationships, the proposed approach offers direct mapping from waveform to magnitude, reduced reliance on handcrafted features, and potential for real-time inference.

Compared to other deep learning approaches, using spectrograms allows better noise robustness, and combining CNN and LSTM captures both spatial and temporal features. However, some modern approaches use transformers or fully convolutional architectures, which may offer competitive performance.

## 7.7 Summary

This chapter analyzed the experimental results and discussed the strengths and limitations of the proposed CNN–LSTM model.

The findings suggest that early seismic signals contain sufficient information for approximate magnitude estimation, though performance depends strongly on dataset composition and model design.